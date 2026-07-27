"""
hitung_bertscore_p6.py — hitung ulang BERTScore Pivot 6 dari prediksi tersimpan.

KENAPA ADA. Nilai BERTScore P6 hanya tercatat di CLAUDE.md tanpa berkas sumber:
seluruh notebook di `notebooks/` tersimpan dengan 0 output, dan tidak ada JSON
hasil di `results/`. Prediksi mentah keenam konfigurasi MASIH ADA
(`results/preds_p6/preds/`), jadi BERTScore bisa dihitung ulang tanpa perlu
membangkitkan jawaban lagi — tanpa H100, cukup GPU lokal.

GERBANG KESELARASAN (penting, jangan dihapus). `eval.py:evaluate()` MENGACAK
sampel (`random.shuffle`) sebelum memotong ke `n_eval`. README preds mengklaim
`preds[i]` sejajar dengan baris ke-i `test.jsonl`. Kedua hal itu tidak otomatis
konsisten. Kalau urutannya ternyata teracak, memasangkan preds[i] dengan
referensi baris ke-i menghasilkan skor yang SALAH tapi tetap terlihat wajar.
Karena itu skrip menghitung dulu token-F1 dan ROUGE-L dengan pemasangan
urutan-berkas, lalu MEMBANDINGKANNYA dengan nilai yang sudah terverifikasi di
`results/eval_truncation.json`. BERTScore hanya dihitung bila selisihnya di
bawah toleransi.

Pemakaian (WAJIB venv GPU — paket torch & bert-score ada di sana):

    .venv-gpu/Scripts/python.exe hitung_bertscore_p6.py
    .venv-gpu/Scripts/python.exe hitung_bertscore_p6.py --periksa-saja   # gerbang saja

Keluaran: results/bertscore_p6.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Konsol Windows default cp1252 -> karakter Δ pada laporan bikin crash.
# Sama seperti skrip judge, stdout dipaksa UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chat_format import clean_greeting          # noqa: E402
from eval import rougeL, token_f1               # noqa: E402

TEST_FILE = ROOT / "Data" / "processed_id" / "test.jsonl"
PREDS_DIR = ROOT / "results" / "preds_p6" / "preds"
TRUNC_FILE = ROOT / "results" / "eval_truncation.json"
OUT_FILE = ROOT / "results" / "bertscore_p6.json"

# Konfigurasi BERTScore disamakan dengan default eval.py supaya angkanya
# sebanding dengan metrik lain di naskah.
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BATCH_SIZE = 32

LABEL = {  # label berkas preds -> kunci ukuran di eval_truncation.json
    "qwen0.8b_baseline": ("0.8b", "baseline"),
    "qwen0.8b_finetuned": ("0.8b", "finetuned"),
    "qwen2b_baseline": ("2b", "baseline"),
    "qwen2b_finetuned": ("2b", "finetuned"),
    "qwen4b_baseline": ("4b", "baseline"),
    "qwen4b_finetuned": ("4b", "finetuned"),
}

TOLERANSI = 0.002  # selisih maksimum yang masih dianggap reproduksi


def muat_referensi() -> list[str]:
    """Referensi = isi pesan `assistant`, sama seperti split_prompt_reference()."""
    refs = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for baris in f:
            s = json.loads(baris)
            refs.append(next((m["content"] for m in s["messages"]
                              if m["role"] == "assistant"), ""))
    return refs


def muat_preds(label: str) -> list[str]:
    d = json.loads((PREDS_DIR / f"{label}.preds.json").read_text(encoding="utf-8"))
    # Prediksi disimpan PRA-clean_greeting; eval.py menerapkannya sebelum
    # menghitung metrik, jadi di sini harus diterapkan juga.
    return [clean_greeting(p) for p in d["preds"]]


def gerbang_keselarasan(refs: list[str]) -> tuple[bool, list[dict]]:
    """Reproduksi token-F1 & ROUGE-L dengan pemasangan urutan-berkas, lalu
    bandingkan dengan angka terverifikasi. Mengembalikan (lolos, laporan)."""
    acuan = json.loads(TRUNC_FILE.read_text(encoding="utf-8"))["per_ukuran"]
    laporan, lolos = [], True

    for label, (ukuran, kondisi) in LABEL.items():
        preds = muat_preds(label)
        if len(preds) != len(refs):
            laporan.append({"label": label, "status": "GAGAL",
                            "alasan": f"jumlah tidak cocok: {len(preds)} vs {len(refs)}"})
            lolos = False
            continue

        f1 = sum(token_f1(p, r) for p, r in zip(preds, refs)) / len(refs)
        rl = sum(rougeL(p, r) for p, r in zip(preds, refs)) / len(refs)
        harap = acuan[ukuran]["semua"][kondisi]
        d_f1, d_rl = abs(f1 - harap["f1"]), abs(rl - harap["rougeL"])
        ok = d_f1 <= TOLERANSI and d_rl <= TOLERANSI
        lolos &= ok

        laporan.append({
            "label": label, "status": "OK" if ok else "GAGAL",
            "token_f1": round(f1, 4), "token_f1_acuan": round(harap["f1"], 4),
            "rougeL": round(rl, 4), "rougeL_acuan": round(harap["rougeL"], 4),
            "selisih_f1": round(d_f1, 5), "selisih_rougeL": round(d_rl, 5),
        })
        tanda = "OK  " if ok else "GAGAL"
        print(f"  [{tanda}] {label:22s} F1 {f1:.4f} (acuan {harap['f1']:.4f}, Δ{d_f1:.5f})"
              f"   ROUGE-L {rl:.4f} (acuan {harap['rougeL']:.4f}, Δ{d_rl:.5f})")

    return lolos, laporan


def hitung_bertscore(refs: list[str]) -> tuple[dict, dict]:
    """Kembalikan (rerata per label, skor per-sampel per label). Skor per-sampel
    disimpan karena CI paired bootstrap butuh pasangan per butir, bukan rerata."""
    import torch
    from bert_score import score as _bs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[bertscore] model={BERTSCORE_MODEL} device={device}")

    rerata, per_sampel = {}, {}
    for label in LABEL:
        preds = muat_preds(label)
        _, _, F = _bs(preds, refs, model_type=BERTSCORE_MODEL,
                      batch_size=BATCH_SIZE, verbose=False, device=device)
        skor = F.tolist()
        per_sampel[label] = skor
        rerata[label] = round(sum(skor) / len(skor), 4)
        print(f"  {label:22s} BERTScore F1 = {rerata[label]:.4f}")
    return rerata, per_sampel


def bootstrap_berpasangan(per_sampel: dict, n_iter: int = 5000, seed: int = 42) -> dict:
    """CI 95% atas delta (fine-tuned - baseline), protokol sama dengan metrik
    Tingkat 1 lain: resample INDEKS yang sama untuk kedua kondisi (berpasangan),
    5.000 iterasi."""
    import random

    rng = random.Random(seed)
    hasil = {}
    for u in ["0.8b", "2b", "4b"]:
        a = per_sampel[f"qwen{u}_finetuned"]
        b = per_sampel[f"qwen{u}_baseline"]
        n = len(a)
        selisih = [x - y for x, y in zip(a, b)]
        titik = sum(selisih) / n

        sebaran = []
        for _ in range(n_iter):
            idx = [rng.randrange(n) for _ in range(n)]
            sebaran.append(sum(selisih[i] for i in idx) / n)
        sebaran.sort()
        lo = sebaran[int(0.025 * n_iter)]
        hi = sebaran[int(0.975 * n_iter)]

        hasil[u] = {"delta": round(titik, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "signifikan": lo > 0 or hi < 0}
        print(f"  {u:5s} Δ BERTScore = {titik:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]"
              f"  {'signifikan' if (lo > 0 or hi < 0) else 'TIDAK signifikan'}")
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periksa-saja", action="store_true",
                    help="jalankan gerbang keselarasan lalu berhenti")
    args = ap.parse_args()

    refs = muat_referensi()
    print(f"referensi: {len(refs)} baris dari {TEST_FILE.name}\n")
    print("=== GERBANG KESELARASAN (reproduksi token-F1 & ROUGE-L) ===")
    lolos, laporan = gerbang_keselarasan(refs)

    if not lolos:
        print("\nGERBANG GAGAL. preds TIDAK sejajar dengan urutan test.jsonl, atau "
              "protokolnya berbeda.\nBERTScore TIDAK dihitung — angkanya akan salah "
              "tanpa terlihat salah.\nPeriksa apakah eval.py:evaluate() mengacak "
              "sampel saat prediksi ini dibuat.")
        sys.exit(1)

    print("\nGerbang LOLOS — pemasangan urutan-berkas terbukti benar.")
    if args.periksa_saja:
        return

    bertscore, per_sampel = hitung_bertscore(refs)

    print("\n=== PAIRED BOOTSTRAP (5.000 iterasi, seed 42) ===")
    delta = bootstrap_berpasangan(per_sampel)

    keluaran = {
        "sumber": "dihitung ulang dari results/preds_p6/preds/ (prediksi tersimpan)",
        "test_file": str(TEST_FILE.relative_to(ROOT)).replace("\\", "/"),
        "n": len(refs),
        "bertscore_model": BERTSCORE_MODEL,
        "postclean": "clean_greeting diterapkan (sama seperti eval.py)",
        "gerbang_keselarasan": laporan,
        "bertscore_f1": bertscore,
        "delta_finetuned_minus_baseline": delta,
    }
    OUT_FILE.write_text(json.dumps(keluaran, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\ntersimpan: {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
