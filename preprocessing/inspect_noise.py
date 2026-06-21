"""
inspect_noise.py  —  BAGIAN 1 (TUGAS 2): cell inspeksi noise ITERATIF.

Alat diagnostik untuk dijalankan BERULANG (CPU/lokal, 0 unit GPU):
    cleaning -> inspeksi -> temukan pola noise baru -> tambah filter -> ulangi.

Pakai:
    python preprocessing/inspect_noise.py                       # default: train_final
    python preprocessing/inspect_noise.py Data/processed_final/val_final.jsonl
    python preprocessing/inspect_noise.py <file> --simulate     # simulasi clean_greeting (Task 3: cek over-strip)

Alur kerja (iteratif):
  1. Jalankan -> lihat kategori noise mana yang masih tinggi.
  2. Pola baru yang muncul -> tambahkan ke chat_format._GREETING_PATTERNS atau
     filter quality di preprocess_dataset.py.
  3. Jalankan ulang preprocess_dataset.py, lalu script ini lagi.
  4. Ulangi sampai semua kategori < 0.5% (atau serendah yang AMAN tanpa over-strip).
  5. Catat persentase final tiap kategori -> dokumentasi limitasi BAB V.
"""

import json
import re
import sys
import random
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chat_format import clean_greeting  # noqa: E402

DEFAULT_FILE = ROOT / "Data" / "processed_final" / "train_final.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def assistant_texts(samples):
    out = []
    for s in samples:
        for m in s.get("messages", []):
            if m.get("role") == "assistant":
                out.append((s.get("source", "?"), m.get("content", "")))
    return out


# --- DETEKTOR NOISE: tandai sampel yang MASIH mencurigakan setelah cleaning ---
SUSPICIOUS_PATTERNS = {
    "sapaan_awal":  r"^(?:Hai|Halo|Alo|Hi|Hello|Hallo|Salam|Selamat|Dear|@\w+|Terima\s*kasih|Terimakasih|Thanks|Thank\s*you)\b",
    "mention":      r"@\w+",
    "url":          r"https?://|www\.",
    "html":         r"<[a-z]+>|&nbsp;|&amp;",
    "markdown_dump": r"#{2,}|\|\s*-{2,}",
    "placeholder":  r"\[deleted\]|\[removed\]|lorem ipsum|\bN/A\b|\bTBD\b",
    "kode_aneh":    r"\{[^}]*\}|\[\[[^\]]*\]\]",   # template residue
}


def scan_noise(samples):
    flagged = {k: [] for k in SUSPICIOUS_PATTERNS}
    flagged["terlalu_pendek"] = []
    for src, txt in assistant_texts(samples):
        if len(txt.strip()) < 20:
            flagged["terlalu_pendek"].append((src, txt[:80]))
        for name, pat in SUSPICIOUS_PATTERNS.items():
            if re.search(pat, txt, flags=re.IGNORECASE):
                flagged[name].append((src, txt[:100]))
    return flagged


def report(samples):
    flagged = scan_noise(samples)
    total = max(1, len(assistant_texts(samples)))
    print("=== RINGKASAN NOISE TERSISA ===")
    print(f"  (total assistant turns = {total})")
    clean = True
    for name, items in flagged.items():
        if items:
            clean = False
            print(f"  {name:16s}: {len(items):5d}  ({len(items)/total*100:.2f}%)")
    if clean:
        print("  (tidak ada kategori noise terdeteksi)")

    print("\n=== CONTOH SAMPEL MENCURIGAKAN (per kategori, maks 5) ===")
    for name, items in flagged.items():
        if items:
            print(f"\n--- {name} ({len(items)}) ---")
            for src, snip in random.sample(items, min(5, len(items))):
                print(f"  [{src}] {snip!r}")


def simulate_overstrip(samples, n=20):
    """Task 3: simulasikan clean_greeting pada data SEKARANG dan laporkan jawaban
    yang akan kehilangan >50 char -> verifikasi tidak ada konten medis valid hilang."""
    suspicious = []
    for src, txt in assistant_texts(samples):
        after = clean_greeting(txt)
        removed = len(txt) - len(after)
        if removed > 50:
            suspicious.append((removed, src, txt[:70], after[:70]))
    suspicious.sort(reverse=True)
    print(f"\n=== SIMULASI OVER-STRIP (clean_greeting buang >50 char, top {n}) ===")
    if not suspicious:
        print("  (tidak ada — aman, atau data sudah dibersihkan)")
        return
    for removed, src, before, after in suspicious[:n]:
        print(f"  -{removed}c [{src}] SBL: {before!r}")
        print(f"            SSD: {after!r}\n")


def main():
    args = [a for a in sys.argv[1:]]
    simulate = "--simulate" in args
    args = [a for a in args if not a.startswith("--")]
    path = Path(args[0]) if args else DEFAULT_FILE
    print(f"File: {path}")
    samples = load_jsonl(path)
    report(samples)
    if simulate:
        simulate_overstrip(samples)


if __name__ == "__main__":
    main()
