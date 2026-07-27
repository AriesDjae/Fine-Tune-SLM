"""
judge_run.py — SATU perintah untuk menjalankan penilaian Tier 2 setelah key ditempel.

Anda cukup:
  1. Tempel API key OpenAI ke file `openai.key` (ganti baris placeholder).
  2. Jalankan salah satu:

     python judge_run.py            # PILOT 20 item + contoh hasil, lalu BERHENTI (murah,
                                    # untuk cek mutu penilaian sebelum keluar banyak biaya)
     python judge_run.py --full     # pilot -> lanjut ke 200 -> ringkasan (sekali jalan)

Opsi lain: --model-key {0.8b,2b,4b} (default 0.8b) · --judge-model gpt-4o (default) ·
--provider {openai,openrouter} · --pilot N (default 20).

Semua aman diputus (checkpoint per item); menjalankan lagi akan melanjutkan, bukan mengulang.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from llm_judge_openai import PROVIDER, _api_key, jalankan
from llm_judge_tier2 import OUT_DIR, bangun_sampel, ckpt_path, ringkas


def _cek_kunci(provider: str) -> None:
    _, env_key, key_file, prefiks = PROVIDER[provider]
    if _api_key(env_key, key_file, prefiks) is None:
        print("=" * 68)
        print(f"  KUNCI API BELUM ADA untuk provider '{provider}'.")
        print(f"  Buka file '{key_file}', ganti baris placeholder dengan API key Anda")
        print(f"  (diawali {' / '.join(prefiks)}), simpan, lalu jalankan perintah ini lagi.")
        print(f"  Alternatif: set variabel lingkungan {env_key}.")
        print("=" * 68)
        sys.exit(1)


def _readout(model_key: str, n: int, batas: int, strip: bool = False, tag: str = "") -> None:
    """Cetak beberapa contoh hasil pilot agar mutu penilaian bisa dilihat mata."""
    ckpt = ckpt_path(model_key, tag)
    if not os.path.exists(ckpt):
        return
    hasil = {}
    with open(ckpt, encoding="utf-8") as f:
        for baris in f:
            try:
                r = json.loads(baris)
                hasil[r["idx"]] = r
            except Exception:
                pass
    sampel = {it["idx"]: it for it in bangun_sampel(model_key, n, strip=strip)}

    print(f"\n{'='*68}\nCONTOH HASIL PILOT (untuk cek mutu — kode baseline/ft dibuka di sini saja):\n")
    ditampilkan = 0
    for idx, r in list(hasil.items())[:5]:
        it = sampel.get(idx)
        if not it:
            continue
        tanya = " ".join(it["pertanyaan"].split())[:100]
        print(f"[idx {idx}] {tanya}")
        for slot in ("A", "B"):
            v = r["nilai"][slot]
            kondisi = r["kunci"][slot]
            print(f"   {slot} ({kondisi:9s}): aman={v['aman']:5s} akurat={v['akurat']:8s} "
                  f"menyesatkan={v['menyesatkan']:5s} | {v.get('alasan','')[:70]}")
        pilih = r["nilai"]["lebih_baik"]
        lbl = r["kunci"].get(pilih, "seri") if pilih in ("A", "B") else "seri"
        print(f"   -> lebih baik: {pilih} ({lbl})\n")
        ditampilkan += 1
    if ditampilkan == 0:
        print("  (belum ada hasil — semua item pilot gagal? cek pesan error di atas)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="jalankan penuh 200 lalu ringkas (bukan pilot saja)")
    ap.add_argument("--model-key", default="0.8b", choices=["0.8b", "2b", "4b"])
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDER))
    ap.add_argument("--pilot", type=int, default=20, help="jumlah item pilot (default 20)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show-reference", action="store_true",
                    help="sodorkan referensi dokter (default TIDAK; lihat catatan objektivitas)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="batas token keluaran; default per-penyedia (700, 3000 utk gemini "
                         "yang model penalarannya ikut memakai anggaran ini)")
    ap.add_argument("--tag", default="",
                    help="penanda run terpisah (mis. 'strip', 'gpt5'); kosong = run utama")
    ap.add_argument("--strip-disclaimer", action="store_true",
                    help="ABLATION: buang kalimat disclaimer dari KEDUA kondisi sebelum dinilai")
    args = ap.parse_args()

    _cek_kunci(args.provider)

    # Jaga-jaga: run non-utama WAJIB bertag, supaya tak mencemari checkpoint run utama
    # yang hasilnya sudah dipakai (results/llm_judge/tier2_<key>.jsonl).
    if (args.strip_disclaimer or args.judge_model != ap.get_default("judge_model")) and not args.tag:
        raise SystemExit(
            "Run ini beda kondisi dari run utama (strip-disclaimer dan/atau penilai lain).\n"
            "Beri --tag agar hasilnya masuk berkas terpisah, mis. --tag strip / --tag gpt5."
        )

    ekstra = dict(strip=args.strip_disclaimer, tag=args.tag)
    if not args.full:
        print(f">> PILOT {args.pilot} item dengan {args.provider}:{args.judge_model}...")
        jalankan(args.model_key, args.n, args.pilot, args.provider, args.judge_model,
                 args.show_reference, args.max_tokens, **ekstra)
        _readout(args.model_key, args.n, args.pilot, args.strip_disclaimer, args.tag)
        print(f"{'='*68}")
        print("PILOT SELESAI. Kalau hasil di atas masuk akal, jalankan penuh:")
        tambahan = (" --strip-disclaimer" if args.strip_disclaimer else "") + \
                   (f" --tag {args.tag}" if args.tag else "")
        print(f"   python judge_run.py --full --model-key {args.model_key} "
              f"--judge-model {args.judge_model}{tambahan}")
        print(f"({args.pilot} item pilot tak akan diulang — resume otomatis.)")
        return

    print(f">> RUN PENUH s/d {args.n} item dengan {args.provider}:{args.judge_model}...")
    jalankan(args.model_key, args.n, None, args.provider, args.judge_model,
             args.show_reference, args.max_tokens, **ekstra)
    print(f"\n{'='*68}\nRINGKASAN:\n")
    ringkas(args.model_key, args.tag)


if __name__ == "__main__":
    sys.exit(main())
