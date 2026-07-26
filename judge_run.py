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
from llm_judge_tier2 import OUT_DIR, bangun_sampel, ringkas


def _cek_kunci(provider: str) -> None:
    _, env_key, key_file = PROVIDER[provider]
    if _api_key(env_key, key_file) is None:
        print("=" * 68)
        print(f"  KUNCI API BELUM ADA untuk provider '{provider}'.")
        print(f"  Buka file '{key_file}', ganti baris placeholder dengan API key Anda")
        print(f"  (diawali 'sk-'), simpan, lalu jalankan perintah ini lagi.")
        print(f"  Alternatif: set variabel lingkungan {env_key}.")
        print("=" * 68)
        sys.exit(1)


def _readout(model_key: str, n: int, batas: int) -> None:
    """Cetak beberapa contoh hasil pilot agar mutu penilaian bisa dilihat mata."""
    ckpt = os.path.join(OUT_DIR, f"tier2_{model_key}.jsonl")
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
    sampel = {it["idx"]: it for it in bangun_sampel(model_key, n)}

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
    ap.add_argument("--max-tokens", type=int, default=700)
    args = ap.parse_args()

    _cek_kunci(args.provider)

    if not args.full:
        print(f">> PILOT {args.pilot} item dengan {args.provider}:{args.judge_model}...")
        jalankan(args.model_key, args.n, args.pilot, args.provider, args.judge_model,
                 args.show_reference, args.max_tokens)
        _readout(args.model_key, args.n, args.pilot)
        print(f"{'='*68}")
        print("PILOT SELESAI. Kalau hasil di atas masuk akal, jalankan penuh:")
        print(f"   python judge_run.py --full --model-key {args.model_key} --judge-model {args.judge_model}")
        print("(20 item pilot tak akan diulang — resume otomatis.)")
        return

    print(f">> RUN PENUH s/d {args.n} item dengan {args.provider}:{args.judge_model}...")
    jalankan(args.model_key, args.n, None, args.provider, args.judge_model,
             args.show_reference, args.max_tokens)
    print(f"\n{'='*68}\nRINGKASAN:\n")
    ringkas(args.model_key)


if __name__ == "__main__":
    sys.exit(main())
