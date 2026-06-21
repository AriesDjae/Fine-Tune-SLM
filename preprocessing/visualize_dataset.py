"""
visualize_dataset.py  —  BAGIAN A (NOTE VISUALISASI): grafik identifikasi/
pembersihan dataset, dari DATA NYATA (Data/processed_final/*). Untuk skripsi.

Menghasilkan (PNG dpi=150 di results/figures/) + cetak angka kunci + penjelasan:
  A.1 fig_source_dist  — distribusi sumber (bar)
  A.2 fig_lang_prop    — proporsi bahasa ID vs EN (bar)
  A.3 fig_token_dist   — distribusi panjang token (histogram; perlu tokenizer)
  A.4 fig_noise_cat    — noise tersisa per kategori (bar)
  A.5 fig_funnel       — jumlah sampel per tahap preprocessing (bar)

Pakai:
    python preprocessing/visualize_dataset.py
    python preprocessing/visualize_dataset.py --split train --tokenizer unsloth/Qwen3.5-2B
    python preprocessing/visualize_dataset.py --no-token        # lewati A.3 (tanpa download tokenizer)
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # tanpa GUI -> aman di server/headless
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocessing"))
from preprocess_dataset import load_jsonl, is_id, msgs  # noqa: E402
from inspect_noise import scan_noise  # noqa: E402

FINAL = ROOT / "Data" / "processed_final"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

C = dict(green="#1D9E75", blue="#378ADD", orange="#BA7517", red="#E24B4A",
         purple="#534AB7", teal="#0F6E56")


def save(fig, name):
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def a1_source(samples, split):
    src = Counter(s.get("source", "unknown") for s in samples)
    names = [k for k, _ in src.most_common()]
    vals = [v for _, v in src.most_common()]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names[::-1], vals[::-1], color=C["green"])
    ax.set_xlabel("Jumlah sampel")
    ax.set_title(f"Distribusi sumber dataset ({split})")
    save(fig, f"fig_source_dist_{split}")
    top = src.most_common(1)[0]
    print(f"[A.1] {len(names)} sumber, total {sum(vals):,}. Terbesar: {top[0]} "
          f"(n={top[1]:,}, {top[1]/sum(vals)*100:.1f}%) -> perlu balancing antar domain.")


def a2_lang(samples, split):
    idn = sum(1 for s in samples if is_id(s))
    en = len(samples) - idn
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Indonesia", "English"], [idn, en], color=[C["blue"], C["orange"]])
    ax.set_ylabel("Jumlah sampel")
    ax.set_title(f"Proporsi bahasa ({split})")
    for i, v in enumerate([idn, en]):
        ax.text(i, v, f"{v:,}\n({v/len(samples)*100:.1f}%)", ha="center", va="bottom")
    save(fig, f"fig_lang_prop_{split}")
    print(f"[A.2] Indonesia {idn:,} ({idn/len(samples)*100:.1f}%) | English {en:,}. "
          f"Proporsi ID relevan untuk klaim deployment Puskesmas (RQ3).")


def a3_token(samples, split, tokenizer_id, limit=5000):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"[A.3] DILEWATI (gagal load tokenizer {tokenizer_id}: {repr(e)[:80]}). "
              f"Pakai --tokenizer <id lain> atau --no-token.")
        return
    lengths = []
    for s in samples[:limit]:
        try:
            text = tok.apply_chat_template(s["messages"], tokenize=False)
        except Exception:
            text = " ".join(m.get("content", "") for m in msgs(s))
        lengths.append(len(tok(text)["input_ids"]))
    p95 = np.percentile(lengths, 95)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, color=C["purple"])
    ax.axvline(p95, color=C["red"], linestyle="--", label=f"P95 = {p95:.0f}")
    ax.set_xlabel("Panjang token"); ax.set_ylabel("Frekuensi")
    ax.set_title(f"Distribusi panjang token ({split}, n={len(lengths)})")
    ax.legend()
    save(fig, f"fig_token_dist_{split}")
    print(f"[A.3] Token: median {np.median(lengths):.0f}, P95 {p95:.0f}, max {max(lengths)}. "
          f"P95 < 1024 -> justifikasi max_seq_length=1024 (truncation minimal, hemat memori).")


def a4_noise(samples, split):
    flagged = scan_noise(samples)
    cats = {k: len(v) for k, v in flagged.items() if v}
    total = sum(1 for s in samples for m in msgs(s) if m.get("role") == "assistant")
    if not cats:
        print("[A.4] Tidak ada kategori noise terdeteksi.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(cats.keys()), list(cats.values()), color=C["red"])
    ax.set_xlabel("Jumlah jawaban ter-flag")
    ax.set_title(f"Noise tersisa per kategori ({split})")
    save(fig, f"fig_noise_cat_{split}")
    top = max(cats.items(), key=lambda x: x[1])
    print(f"[A.4] Noise residual tertinggi: {top[0]} ({top[1]}, {top[1]/total*100:.2f}%). "
          f"Sisanya didokumentasikan sebagai limitasi BAB V.")


def a5_funnel():
    stats_path = ROOT / "results" / "preprocess_stats.json"
    if not stats_path.exists():
        print("[A.5] DILEWATI (results/preprocess_stats.json tak ada -> jalankan "
              "preprocess_dataset.py dulu).")
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    steps = [r["step"] for r in stats]
    counts = [r["train"] for r in stats]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(steps, counts, color=C["teal"])
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Jumlah sampel (train)")
    ax.set_title("Jumlah sampel per tahap preprocessing (funnel)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save(fig, "fig_funnel_train")
    print(f"[A.5] Funnel {counts[0]:,} -> {counts[-1]:,} sampel "
          f"({(1-counts[-1]/counts[0])*100:.1f}% dibuang). Tabel ini = tabel preprocessing BAB III.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--tokenizer", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--no-token", action="store_true", help="lewati A.3 (token dist)")
    args = ap.parse_args()

    samples = load_jsonl(FINAL / f"{args.split}_final.jsonl")
    print(f"== VISUALISASI DATASET ({args.split}_final.jsonl, n={len(samples):,}) ==")
    print(f"   PNG disimpan di: {FIGDIR}\n")
    a1_source(samples, args.split)
    a2_lang(samples, args.split)
    if not args.no_token:
        a3_token(samples, args.split, args.tokenizer)
    a4_noise(samples, args.split)
    a5_funnel()
    print("\nSelesai. Masukkan PNG di results/figures/ ke skripsi (BAB III).")


if __name__ == "__main__":
    main()
