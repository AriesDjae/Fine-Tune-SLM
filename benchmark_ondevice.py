"""
benchmark_ondevice.py — RQ3: kelayakan deploy offline di HP kelas bawah (CPU-only).

PERTANYAAN YANG DIJAWAB. RQ2 menunjukkan 0.8B meraih ~96% mutu 4B dengan 1/5 parameter.
Itu baru bermakna bila disertai angka perangkat: **apakah 4B benar-benar tidak muat, dan
apakah 0.8B benar-benar muat, di HP dengan RAM 3-4 GB?** Skrip ini mengukurnya.

YANG DIUKUR (per model, GGUF Q4_K_M, llama.cpp murni CPU, n_gpu_layers=0):
  - PUNCAK RAM saat MEMUAT bobot          -> menentukan lolos/tidak gerbang RAM
  - PUNCAK RAM saat INFERENSI (KV cache)  -> puncak sebenarnya saat dipakai
  - latensi & throughput (token/detik), mean +- std
  - ukuran berkas GGUF
  - overhead lapisan red-flag (klaim '~0 ms' DIBUKTIKAN, bukan diasumsikan)

GERBANG RAM. Sasaran deploy = HP masyarakat umum, bukan flagship. Anggaran default 3 GB
dan 4 GB TOTAL, dengan asumsi konservatif bahwa sistem operasi + aplikasi lain memakai
sebagian (lihat `SISA_UNTUK_APP`). Perangkat uji WAJIB disebut di metodologi; angka dari
laptop hanya PROKSI — ulangi di HP nyata via Termux bila memungkinkan.

PERTANYAAN UJI diambil dari test set NYATA (keluhan pasien berbahasa Indonesia), bukan
kalimat karangan, supaya panjang prompt dan perilaku generasi mewakili pemakaian
sesungguhnya. Sekitar sepertiga sengaja dipilih dari kasus yang memicu red-flag.

Pemakaian:

    python benchmark_ondevice.py --gguf outputs/gguf/Qwen3.5-0.8B.Q4_K_M.gguf --label 0.8b
    python benchmark_ondevice.py --gguf outputs/gguf/Qwen3.5-2B.Q4_K_M.gguf   --label 2b
    python benchmark_ondevice.py --gguf outputs/gguf/Qwen3.5-4B.Q4_K_M.gguf   --label 4b
    python benchmark_ondevice.py --summarize results_bench

Riwayat: versi Pivot-3 skrip ini menguji hipotesis Gemma 4 PLE vs Qwen-2B dan sudah tidak
berlaku; ditulis ulang penuh untuk studi scaling satu keluarga (Pivot 6).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import statistics
import sys
import threading
import time

try:
    import psutil
except ImportError:                                  # Termux/Android: psutil menolak dipasang
    psutil = None                                    # ("platform android is not supported")


def _rss_bytes() -> int:
    """RSS proses ini. /proc dipakai bila psutil tak ada — di Android itu satu-satunya jalan,
    dan angkanya berasal dari sumber yang sama dengan yang dibaca psutil di Linux."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss
    with open("/proc/self/status", encoding="utf-8") as f:
        for baris in f:
            if baris.startswith("VmRSS:"):
                return int(baris.split()[1]) * 1024
    raise RuntimeError("tidak bisa membaca RSS: psutil tak ada dan /proc/self/status tak terbaca")


def _meminfo_bytes(kunci: str) -> int:
    with open("/proc/meminfo", encoding="utf-8") as f:
        for baris in f:
            if baris.startswith(kunci + ":"):
                return int(baris.split()[1]) * 1024
    return 0


def _ram_total_bytes() -> int:
    return psutil.virtual_memory().total if psutil is not None else _meminfo_bytes("MemTotal")


def _ram_available_bytes() -> int:
    return psutil.virtual_memory().available if psutil is not None else _meminfo_bytes("MemAvailable")

for _aliran in (sys.stdout, sys.stderr):
    try:
        _aliran.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

TEST_FILE = os.path.join("Data", "processed_id", "test.jsonl")
SEED = 42

# Arsitektur — semua dense, semua parameter aktif tiap inferensi (studi scaling satu
# keluarga; tak ada MoE/PLE seperti pivot sebelumnya).
ARCH = {
    "0.8b": {"family": "Qwen3.5-0.8B", "total_params_B": 0.8},
    "2b": {"family": "Qwen3.5-2B", "total_params_B": 2.0},
    "4b": {"family": "Qwen3.5-4B", "total_params_B": 4.0},
}

# Anggaran RAM perangkat sasaran. SISA_UNTUK_APP = taksiran konservatif berapa yang
# benar-benar tersedia untuk satu aplikasi setelah OS Android + aplikasi latar.
# 3-4 GB = perangkat sasaran naskah. 6 dan 8 GB ikut dihitung sebagai pembanding
# kelas menengah, supaya pembaca tahu di titik mana 2B dan 4B mulai muat.
GERBANG = {"HP 3 GB": 3.0, "HP 4 GB": 4.0, "HP 6 GB": 6.0, "HP 8 GB": 8.0}
SISA_UNTUK_APP = 0.55

# System prompt PERSIS seperti di data latih/uji (seragam untuk seluruh 2998 baris).
# Wajib sama: panjangnya (~50 token) ikut menentukan waktu prefill, jadi memakai kalimat
# karangan sendiri akan membuat latensi yang dilaporkan tidak mewakili pemakaian nyata.
#
# Catatan temuan: prompt ini secara EKSPLISIT meminta disclaimer ("Selalu ingatkan..."),
# dan baseline mematuhinya (disclaimer 68-93%) sementara model fine-tuned 0,0% —
# fine-tuning mengikis kepatuhan pada instruksi sistem, bukan sekadar mengubah gaya.
SYSTEM_PROMPT = (
    "Anda adalah asisten informasi kesehatan berbahasa Indonesia. Berikan penjelasan "
    "yang jelas, akurat, dan hati-hati berdasarkan pengetahuan klinis yang mapan. "
    "Selalu ingatkan bahwa informasi ini bukan pengganti konsultasi langsung dengan dokter."
)


class PuncakRSS:
    """Sampler RSS latar -> menangkap PUNCAK memori (mis. saat KV cache tumbuh)."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = _rss_bytes()
        self._run = False
        self._t = None

    def reset(self):
        self.peak = _rss_bytes()

    def start(self):
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> float:
        self._run = False
        if self._t:
            self._t.join()
        return self.peak / 1e9  # GB


def ambil_pertanyaan(n: int, berkas: str | None = None) -> list[str]:
    """Keluhan pasien NYATA dari test set; ~1/3 dari kasus yang memicu red-flag.

    `berkas` = JSON berisi daftar pertanyaan yang sudah dipilih sebelumnya. Dipakai saat
    menjalankan benchmark di HP: `Data/` tidak ikut ke perangkat (aturan lisensi yang sama
    dengan bank jawaban), dan lagipula perbandingan laptop vs HP hanya sah bila pertanyaannya
    PERSIS sama. Dump-nya dibuat di laptop dengan --simpan-pertanyaan.
    """
    if berkas:
        with open(berkas, encoding="utf-8") as f:
            daftar = json.load(f)
        if len(daftar) < n:
            raise SystemExit(f"{berkas} hanya berisi {len(daftar)} pertanyaan, diminta {n}.")
        return daftar[:n]
    if not os.path.exists(TEST_FILE):
        raise SystemExit(
            f"Tidak ada {TEST_FILE}. Di HP, pakai --pertanyaan hasil --simpan-pertanyaan "
            "dari laptop supaya pertanyaannya sama persis.")
    semua = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            msg = json.loads(baris)["messages"]
            semua.append(next(m["content"] for m in msg if m["role"] == "user"))

    try:
        from redflag import detect
        gawat = [q for q in semua if detect(q).terpicu]
    except Exception:
        gawat = []

    rng = random.Random(SEED)
    n_gawat = min(len(gawat), max(1, n // 3))
    pilih = rng.sample(gawat, n_gawat) if gawat else []
    sisa = [q for q in semua if q not in set(pilih)]
    pilih += rng.sample(sisa, max(0, n - len(pilih)))
    rng.shuffle(pilih)
    return pilih


def bangun_prompt(q: str) -> str:
    """ChatML Qwen3.5 dengan blok <think> KOSONG — sama persis seperti saat training.

    Template Qwen3.5 menyisipkan <think></think> walau enable_thinking=False; menyamakan
    prefix ini membuat train == inferensi, sehingga latensi dan keluaran mewakili model
    yang sebenarnya dilatih.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _cpu() -> str:
    return platform.processor() or platform.machine() or "tak diketahui"


def _ram_tersedia_GB() -> float:
    """RAM yang BENAR-BENAR bisa dipakai saat ini, bukan taksiran dari RAM total.

    Di Android, RAM nominal 8 GB sering hanya menyisakan ~3 GB karena OS dan aplikasi
    latar. Angka ini dicatat pada setiap run supaya gerbang kelayakan dapat dilaporkan
    dengan ketersediaan TERUKUR, bukan hanya dengan asumsi SISA_UNTUK_APP.
    """
    return round(_ram_available_bytes() / 1e9, 2)


def benchmark(args) -> None:
    from llama_cpp import Llama

    rss0 = _rss_bytes() / 1e9
    sampler = PuncakRSS()
    pertanyaan = ambil_pertanyaan(args.n_queries, args.pertanyaan)

    # ---- MEMUAT (CPU murni) ----
    sampler.reset()
    sampler.start()
    t0 = time.time()
    llm = Llama(model_path=args.gguf, n_ctx=args.n_ctx, n_gpu_layers=0,
                n_threads=args.threads, verbose=False)
    waktu_muat = time.time() - t0
    puncak_muat = sampler.stop()
    rss_setelah_muat = proc.memory_info().rss / 1e9

    # ---- pemanasan (tak diukur) ----
    llm.create_completion(bangun_prompt(pertanyaan[0]), max_tokens=8, temperature=0.0)

    # ---- INFERENSI ----
    sampler.reset()
    sampler.start()
    lat, tps, n_tok = [], [], []
    for i, q in enumerate(pertanyaan, 1):
        t = time.time()
        o = llm.create_completion(bangun_prompt(q), max_tokens=args.max_new_tokens,
                                  temperature=0.0, top_k=1)
        dt = time.time() - t
        ntok = o.get("usage", {}).get("completion_tokens") or args.max_new_tokens
        lat.append(dt)
        n_tok.append(ntok)
        tps.append(ntok / dt if dt > 0 else 0.0)
        print(f"  {i}/{len(pertanyaan)}  {dt:6.2f}s  {tps[-1]:5.1f} tok/s   ", end="\r")
    print()
    puncak_infer = sampler.stop()

    # ---- overhead lapisan red-flag ----
    overhead_ms = None
    try:
        from redflag import detect
        t = time.time()
        for q in pertanyaan:
            detect(q)
        overhead_ms = (time.time() - t) / len(pertanyaan) * 1000
    except Exception:
        pass

    gerbang = {}
    for nama, total in GERBANG.items():
        tersedia = total * SISA_UNTUK_APP
        gerbang[nama] = {
            "ram_total_GB": total,
            "diasumsikan_tersedia_GB": round(tersedia, 2),
            "puncak_infer_GB": round(puncak_infer, 2),
            "lolos": bool(puncak_infer <= tersedia),
        }

    out = {
        "label": args.label,
        "gguf": os.path.basename(args.gguf),
        "kuantisasi": "Q4_K_M",
        "arsitektur": ARCH.get(args.label, {}),
        "perangkat": {
            "cpu": _cpu(), "threads": args.threads,
            "ram_total_GB": round(_ram_total_bytes() / 1e9, 1),
            "ram_tersedia_saat_mulai_GB": _ram_tersedia_GB(),
            "os": f"{platform.system()} {platform.release()}",
            "CATATAN": "angka dari mesin ini PROKSI; ulangi di HP nyata via Termux",
        },
        "ukuran_berkas_GB": round(os.path.getsize(args.gguf) / 1e9, 3),
        "ram_GB": {
            "dasar": round(rss0, 3),
            "setelah_muat": round(rss_setelah_muat, 3),
            "puncak_muat": round(puncak_muat, 3),
            "puncak_infer": round(puncak_infer, 3),
            "bobot_saja": round(rss_setelah_muat - rss0, 3),
        },
        "waktu": {
            "muat_s": round(waktu_muat, 2),
            "latensi_s_mean": round(statistics.mean(lat), 2),
            "latensi_s_std": round(statistics.pstdev(lat), 2),
            "tok_per_s_mean": round(statistics.mean(tps), 2),
            "tok_per_s_std": round(statistics.pstdev(tps), 2),
            "token_keluaran_mean": round(statistics.mean(n_tok), 1),
            "n_queries": len(pertanyaan),
            "max_new_tokens": args.max_new_tokens,
            "n_ctx": args.n_ctx,
        },
        "redflag_overhead_ms_per_pertanyaan": (
            round(overhead_ms, 3) if overhead_ms is not None else None
        ),
        "gerbang_ram": gerbang,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"bench_{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _cetak_satu(out)
    print(f"\nDisimpan -> {path}")


def _cetak_satu(o: dict) -> None:
    print(f"\n=== {o['label']}  ({o['arsitektur'].get('family', '?')}, {o['kuantisasi']}) ===")
    print(f"  berkas GGUF        : {o['ukuran_berkas_GB']:.2f} GB")
    r = o["ram_GB"]
    print(f"  RAM bobot saja     : {r['bobot_saja']:.2f} GB")
    print(f"  RAM puncak (muat)  : {r['puncak_muat']:.2f} GB")
    print(f"  RAM puncak (infer) : {r['puncak_infer']:.2f} GB   <- penentu gerbang")
    w = o["waktu"]
    print(f"  waktu muat         : {w['muat_s']:.2f} s")
    print(f"  latensi/pertanyaan : {w['latensi_s_mean']:.2f} +- {w['latensi_s_std']:.2f} s")
    print(f"  throughput         : {w['tok_per_s_mean']:.1f} +- {w['tok_per_s_std']:.1f} tok/s")
    if o.get("redflag_overhead_ms_per_pertanyaan") is not None:
        print(f"  overhead red-flag  : {o['redflag_overhead_ms_per_pertanyaan']:.3f} ms/pertanyaan")
    print("  GERBANG RAM:")
    for nama, g in o["gerbang_ram"].items():
        print(f"    {nama:8s} (tersedia ~{g['diasumsikan_tersedia_GB']:.2f} GB) -> "
              f"{'LOLOS' if g['lolos'] else 'GAGAL'}")


def ringkas(dir_hasil: str) -> None:
    berkas = sorted(glob.glob(os.path.join(dir_hasil, "bench_*.json")))
    if not berkas:
        raise SystemExit(f"Tak ada bench_*.json di {dir_hasil}")
    data = [json.load(open(b, encoding="utf-8")) for b in berkas]
    urut = {"0.8b": 0, "2b": 1, "4b": 2}
    data.sort(key=lambda d: urut.get(d["label"], 9))
    d0 = data[0]

    print(f"\nRQ3 — TRADE-OFF ON-DEVICE (CPU, Q4_K_M, n={d0['waktu']['n_queries']} pertanyaan)")
    print(f"Perangkat: {d0['perangkat']['cpu']} | {d0['perangkat']['threads']} thread "
          f"| RAM {d0['perangkat']['ram_total_GB']} GB | {d0['perangkat']['os']}\n")
    # Gerbang dihitung ulang di sini dari puncak RAM yang tersimpan, jadi menambah kelas
    # perangkat baru di GERBANG tidak menuntut benchmark diulang.
    kelas = sorted(GERBANG.items(), key=lambda kv: kv[1])
    kepala = "".join(f"{nama.replace('HP ', ''):>8s}" for nama, _ in kelas)
    print(f"{'model':8s}{'GGUF':>9s}{'RAM infer':>12s}{'muat':>9s}"
          f"{'latensi':>11s}{'tok/s':>9s}{kepala}")
    for d in data:
        puncak = d["ram_GB"]["puncak_infer"]
        sel = "".join(f"{('ya' if puncak <= tot * SISA_UNTUK_APP else 'TIDAK'):>8s}"
                      for _, tot in kelas)
        print(f"{d['label']:8s}{d['ukuran_berkas_GB']:8.2f}G{puncak:11.2f}G"
              f"{d['waktu']['muat_s']:8.1f}s{d['waktu']['latensi_s_mean']:10.1f}s"
              f"{d['waktu']['tok_per_s_mean']:9.1f}{sel}")
    print(f"  (tersedia untuk aplikasi = {SISA_UNTUK_APP:.0%} RAM total: "
          + ", ".join(f"{nama} -> {tot * SISA_UNTUK_APP:.2f} GB" for nama, tot in kelas) + ")")
    ov = [d.get("redflag_overhead_ms_per_pertanyaan") for d in data if d.get("redflag_overhead_ms_per_pertanyaan")]
    if ov:
        print(f"\nOverhead lapisan red-flag: {statistics.mean(ov):.3f} ms/pertanyaan "
              f"(dibanding latensi model dalam DETIK -> praktis nol, terbukti bukan diasumsikan)")
    print("\nCATATAN: angka dari perangkat di atas adalah PROKSI untuk HP kelas bawah.")
    print("Sebutkan spesifikasi perangkat uji di metodologi; ulangi via Termux bila bisa.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="path berkas GGUF Q4_K_M")
    ap.add_argument("--label", choices=sorted(ARCH), help="ukuran model")
    ap.add_argument("--n_queries", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n_ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out_dir", default="results_bench")
    ap.add_argument("--summarize", metavar="DIR", help="ringkas semua bench_*.json di DIR")
    ap.add_argument("--pertanyaan", metavar="JSON",
                    help="pakai daftar pertanyaan dari berkas (untuk run di HP)")
    ap.add_argument("--simpan-pertanyaan", metavar="JSON", dest="simpan_pertanyaan",
                    help="tulis pertanyaan yang terpilih lalu keluar (tanpa memuat model)")
    args = ap.parse_args()

    if args.simpan_pertanyaan:
        # Dijalankan di laptop: menuliskan pertanyaan yang SAMA dengan yang dipakai run
        # laptop, supaya run HP bisa dibandingkan baris per baris.
        daftar = ambil_pertanyaan(args.n_queries)
        with open(args.simpan_pertanyaan, "w", encoding="utf-8") as f:
            json.dump(daftar, f, ensure_ascii=False, indent=1)
        print(f"{len(daftar)} pertanyaan -> {args.simpan_pertanyaan}")
        return

    if args.summarize:
        ringkas(args.summarize)
    elif args.gguf and args.label:
        if not os.path.exists(args.gguf):
            raise SystemExit(f"Tak ada {args.gguf}")
        benchmark(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
