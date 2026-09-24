"""
demo_chat.py — peraga RQ3: chatbot medis offline di HP, jawaban muncul kata demi kata.

Untuk demo langsung (ke dosen, di Termux). Memakai jalur yang SAMA dengan benchmark:
prompt `bangun_prompt`, pengaman pengulangan DRY+, lapisan red-flag. Bedanya, skrip ini
menampilkan jawaban secara streaming dan tidak mengukur apa pun untuk naskah — angka
RQ3 tetap dari `benchmark_ondevice.py`.

    python demo_chat.py --ukuran 0.8b                 # ketik pertanyaan sendiri
    python demo_chat.py --ukuran 0.8b --contoh 3      # 3 pertanyaan dari pertanyaan_bench.json
    python demo_chat.py --gguf path/model.gguf        # berkas GGUF di lokasi lain

Ketik `keluar` (atau Ctrl+C) untuk berhenti.
"""
from __future__ import annotations

import argparse
import json
import os
import time

# Impor ini sekaligus meluruskan sys.platform "android" -> "linux" sebelum llama_cpp.
from benchmark_ondevice import _ram_available_bytes, _rss_bytes, bangun_prompt

TEBAL, REDUP, MERAH, HIJAU, BIRU, POLOS = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[0m")

NAMA_GGUF = {"0.8b": "Qwen3.5-0.8B", "2b": "Qwen3.5-2B", "4b": "Qwen3.5-4B"}


def cari_gguf(ukuran: str) -> str:
    nama = f"{NAMA_GGUF[ukuran]}.Q4_K_M.gguf"
    for folder in ("~/models", "~/storage/downloads/models", "outputs/gguf"):
        p = os.path.expanduser(os.path.join(folder, nama))
        if os.path.exists(p):
            return p
    raise SystemExit(f"{nama} tidak ditemukan di ~/models, ~/storage/downloads/models, "
                     "atau outputs/gguf. Pakai --gguf untuk menunjuk berkasnya.")


def jawab(llm, q: str) -> dict:
    from pengaman_decoding import dry_plus
    from redflag import detect

    rf = detect(q)
    if rf.terpicu:
        print(f"\n{MERAH}{TEBAL}⚠  TANDA BAHAYA terdeteksi: {rf.ringkas()}{POLOS}")
        print(f"{MERAH}Segera ke IGD / fasilitas kesehatan terdekat. "
              f"Jawaban di bawah hanya informasi awal.{POLOS}")

    p = bangun_prompt(q)
    kw, rapikan = dry_plus(llm, p)
    print(f"\n{HIJAU}{TEBAL}Asisten:{POLOS} ", end="", flush=True)
    t0 = time.time()
    t_pertama, potongan, n_tok = None, [], 0
    for o in llm.create_completion(p, max_tokens=None, temperature=0.0, top_k=1,
                                   stream=True, **kw):
        s = o["choices"][0]["text"]
        if s and t_pertama is None:
            t_pertama = time.time() - t0
        potongan.append(s)
        n_tok += 1
        print(s, end="", flush=True)
    dt = time.time() - t0
    teks = "".join(potongan)
    akhir = rapikan(teks)
    if akhir != teks:
        print(f"\n{REDUP}[pengulangan terdeteksi — jawaban dipotong di titik itu]{POLOS}")

    print(f"\n\n{REDUP}— {n_tok} token · {dt:.1f} dtk · {n_tok / dt:.1f} token/dtk · "
          f"token pertama {t_pertama or 0:.1f} dtk · RAM proses {_rss_bytes() / 1e9:.2f} GB"
          f"{POLOS}")
    return {"pertanyaan": q, "jawaban": akhir, "redflag": rf.ringkas(),
            "token": n_tok, "detik": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukuran", choices=sorted(NAMA_GGUF), default="0.8b")
    ap.add_argument("--gguf", help="path GGUF (bawaan: dicari dari --ukuran)")
    ap.add_argument("--contoh", type=int, default=0, metavar="N",
                    help="jawab N pertanyaan dari --pertanyaan lalu lanjut mode ketik")
    ap.add_argument("--pertanyaan", default="pertanyaan_bench.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--simpan", metavar="JSON", help="simpan tanya-jawab demo ke berkas ini")
    args = ap.parse_args()

    from llama_cpp import Llama

    gguf = args.gguf or cari_gguf(args.ukuran)
    print(f"{BIRU}Memuat {os.path.basename(gguf)} "
          f"({os.path.getsize(gguf) / 1e9:.2f} GB) · {args.threads} thread · "
          f"RAM bebas {_ram_available_bytes() / 1e9:.2f} GB ...{POLOS}")
    t = time.time()
    llm = Llama(model_path=gguf, n_ctx=1024, n_gpu_layers=0,
                n_threads=args.threads, verbose=False)
    print(f"{BIRU}Siap dalam {time.time() - t:.1f} dtk · RAM proses "
          f"{_rss_bytes() / 1e9:.2f} GB · tanpa internet{POLOS}")
    print(f"{REDUP}Informasi & triase awal, BUKAN diagnosis.{POLOS}")

    riwayat = []
    antre = []
    if args.contoh:
        with open(args.pertanyaan, encoding="utf-8") as f:
            antre = json.load(f)[:args.contoh]

    try:
        while True:
            if antre:
                q = antre.pop(0)
                print(f"\n{TEBAL}Pertanyaan:{POLOS} {q}")
            else:
                q = input(f"\n{TEBAL}Pertanyaan (ketik 'keluar'):{POLOS} ").strip()
                if not q:
                    continue
                if q.lower() in {"keluar", "exit", "quit"}:
                    break
            riwayat.append(jawab(llm, q))
    except (KeyboardInterrupt, EOFError):
        print()

    if args.simpan and riwayat:
        with open(args.simpan, "w", encoding="utf-8") as f:
            json.dump(riwayat, f, ensure_ascii=False, indent=1)
        print(f"{REDUP}tersimpan: {args.simpan} ({len(riwayat)} tanya-jawab){POLOS}")


if __name__ == "__main__":
    main()
