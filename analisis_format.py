"""
analisis_format.py — apakah Δ RQ1 sekadar efek format (markdown + kalimat pembuka)?

KENAPA ADA. Baseline menulis markdown (judul, cetak tebal, daftar bernomor) dan membuka
jawaban dengan basa-basi ("Saya mengerti kekhawatiran Anda...", "Sebagai asisten...");
fine-tuned tidak melakukan keduanya. Keberatan yang sah: "fine-tuning cuma mengajari
model membuang format, bukan menjawab lebih baik". Angka ablation ini (≤0,003) dulu
hanya tercatat di naskah tanpa kode maupun berkas hasil; skrip ini menggantikannya.

APA YANG DINORMALISASI (pada KEDUA kondisi, setelah clean_greeting seperti eval.py):
  1. markdown: penanda judul, cetak tebal/miring, butir daftar, nomor daftar, backtick,
     garis tabel/pemisah. Tokenisasi eval.py (`\\w+`) sudah mengabaikan simbol, jadi yang
     berpengaruh terutama NOMOR DAFTAR ("1.", "2.") yang ikut terhitung sebagai token.
  2. kalimat pembuka: kalimat di awal jawaban yang hanya berisi basa-basi/identitas
     asisten dibuang (maks. 3 kalimat, berhenti di kalimat pertama yang bukan basa-basi).

Pemakaian (BERTScore butuh GPU/env .venv-gpu; tanpa --bertscore hanya token-F1/ROUGE-L):

    python analisis_format.py [--bertscore]

Butuh `Data/processed_id/test.jsonl` dan `results/preds_p6/preds/*.preds.json`.
Menulis `results/eval_format.json`.
"""
import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chat_format import clean_greeting  # noqa: E402
from eval import rougeL, token_f1  # noqa: E402

TEST_FILE = ROOT / "Data" / "processed_id" / "test.jsonl"
PREDS_DIR = ROOT / "results" / "preds_p6" / "preds"
OUT_FILE = ROOT / "results" / "eval_format.json"
UKURAN = ["0.8b", "2b", "4b"]

_MD = [
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),          # judul
    (re.compile(r"^\s*[-*•+]\s+", re.M), ""),               # butir daftar
    (re.compile(r"^\s*\d{1,2}[.)]\s+", re.M), ""),          # nomor daftar
    (re.compile(r"(\*\*|__|\*|`)"), ""),                    # tebal/miring/kode
    (re.compile(r"^\s*([-*_=|]\s*){3,}$", re.M), ""),       # garis pemisah/tabel
    (re.compile(r"\|"), " "),                               # sel tabel
]
# Kalimat pembuka = basa-basi atau identitas asisten, tanpa isi medis.
_PEMBUKA = re.compile(
    r"^(saya\s+(sangat|mengerti|memahami|paham|akan|adalah|senang|menghargai|turut)|"
    r"sebagai\s+(asisten|ai|model)|adalah\s+asisten|sangat\s+senang|senang\s+sekali|"
    r"tentu|baik[,!.]|oke|ini\s+adalah\s+(pertanyaan|hal|kekhawatiran)|"
    r"terima\s*kasih|kasih\s+sudah|selamat\s+atas|berikut\s+(adalah|ini|penjelasan)|"
    r"mengerti\s+(bahwa|kekhawatiran)|pertanyaan\s+yang\s+(bagus|baik|sangat)|"
    r"wa.?alaikum|penting\s*:|sini\.|ibu\.|bapak\.)",
    re.I,
)
_KALIMAT = re.compile(r"[^.!?\n]*[.!?:]+\s*|[^\n]+\n*")


def buang_markdown(t: str) -> str:
    for pat, ganti in _MD:
        t = pat.sub(ganti, t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def buang_pembuka(t: str) -> str:
    sisa = t.lstrip()
    for _ in range(3):
        m = _KALIMAT.match(sisa)
        if not m or not _PEMBUKA.match(m.group(0).strip()):
            break
        sisa = sisa[m.end():].lstrip()
    return sisa if len(sisa) >= 15 else t


def normalisasi(t: str) -> str:
    return buang_pembuka(buang_markdown(t))


def muat_referensi() -> list[str]:
    with open(TEST_FILE, encoding="utf-8") as f:
        return [next((m["content"] for m in json.loads(b)["messages"]
                      if m["role"] == "assistant"), "") for b in f]


def muat_preds(label: str) -> list[str]:
    d = json.loads((PREDS_DIR / f"{label}.preds.json").read_text(encoding="utf-8"))
    return [clean_greeting(p) for p in d["preds"]]


def _skor(pasangan):
    p, r = pasangan
    return token_f1(p, r), rougeL(p, r)


def skor_per_sampel(preds, refs, pool):
    hasil = pool.map(_skor, list(zip(preds, refs)), chunksize=50)
    return np.array([h[0] for h in hasil]), np.array([h[1] for h in hasil])


def bootstrap(a, b, n_iter=5000, seed=42):
    """CI 95% paired bootstrap atas rerata (a - b), resample indeks yang sama."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_iter, len(d)))
    sebaran = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(sebaran, 2.5)), float(np.percentile(sebaran, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bertscore", action="store_true")
    args = ap.parse_args()

    refs = muat_referensi()
    skor = {}  # skor[ukuran][kondisi][versi][metrik] = array per sampel
    statistik = {}
    with Pool() as pool:
        for u in UKURAN:
            skor[u] = {}
            for k in ["baseline", "finetuned"]:
                asli = muat_preds(f"qwen{u}_{k}")
                norm = [normalisasi(p) for p in asli]
                statistik[f"{u}_{k}"] = {
                    "berubah": sum(a != b for a, b in zip(asli, norm)),
                    "pembuka_dibuang": sum(buang_markdown(a) != n for a, n in zip(asli, norm)),
                    "kata_rerata_asli": round(float(np.mean([len(a.split()) for a in asli])), 1),
                    "kata_rerata_norm": round(float(np.mean([len(n.split()) for n in norm])), 1),
                }
                skor[u][k] = {}
                for versi, teks in [("asli", asli), ("norm", norm)]:
                    f1, rl = skor_per_sampel(teks, refs, pool)
                    skor[u][k][versi] = {"token_f1": f1, "rougeL": rl, "_teks": teks}
                print(f"  {u:4s} {k:9s} F1 {skor[u][k]['asli']['token_f1'].mean():.4f} -> "
                      f"{skor[u][k]['norm']['token_f1'].mean():.4f}   ROUGE-L "
                      f"{skor[u][k]['asli']['rougeL'].mean():.4f} -> {skor[u][k]['norm']['rougeL'].mean():.4f}"
                      f"   (berubah {statistik[f'{u}_{k}']['berubah']})", flush=True)

    metrik = ["token_f1", "rougeL"]
    if args.bertscore:
        import torch
        from bert_score import score as _bs
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[bertscore] device={dev}", flush=True)
        for u in UKURAN:
            for k in ["baseline", "finetuned"]:
                for versi in ["asli", "norm"]:
                    _, _, F = _bs(skor[u][k][versi]["_teks"], refs, model_type="bert-base-multilingual-cased",
                                  batch_size=32, verbose=False, device=dev)
                    skor[u][k][versi]["bertscore"] = F.numpy()
                print(f"  {u} {k} BERTScore {skor[u][k]['asli']['bertscore'].mean():.4f} -> "
                      f"{skor[u][k]['norm']['bertscore'].mean():.4f}", flush=True)
        metrik.append("bertscore")

    out = {"n": len(refs), "postclean": "clean_greeting lalu normalisasi (markdown + kalimat pembuka)",
           "bootstrap": "paired, 5000x, seed 42", "statistik_normalisasi": statistik, "per_ukuran": {}}
    geser_maks = 0.0
    for u in UKURAN:
        out["per_ukuran"][u] = {}
        for m in metrik:
            b0, b1 = skor[u]["baseline"]["asli"][m], skor[u]["baseline"]["norm"][m]
            f0, f1 = skor[u]["finetuned"]["asli"][m], skor[u]["finetuned"]["norm"][m]
            d0, ci0 = bootstrap(f0, b0)
            d1, ci1 = bootstrap(f1, b1)
            geser = max(abs(b1.mean() - b0.mean()), abs(f1.mean() - f0.mean()))
            geser_maks = max(geser_maks, geser)
            out["per_ukuran"][u][m] = {
                "baseline": [round(float(b0.mean()), 4), round(float(b1.mean()), 4)],
                "finetuned": [round(float(f0.mean()), 4), round(float(f1.mean()), 4)],
                "delta_asli": round(d0, 4), "ci_asli": [round(x, 4) for x in ci0],
                "delta_norm": round(d1, 4), "ci_norm": [round(x, 4) for x in ci1],
                "signifikan_norm": ci1[0] > 0 or ci1[1] < 0,
                "geser_skor_maks": round(float(geser), 4),
                "geser_delta": round(d1 - d0, 4),
            }
            print(f"  {u:4s} {m:9s} Δ {d0:+.4f} -> {d1:+.4f}  CI {ci1[0]:+.4f}..{ci1[1]:+.4f}"
                  f"  geser skor maks {geser:.4f}")
    out["geser_skor_maks_semua"] = round(geser_maks, 4)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: o.item()),
                        encoding="utf-8")
    print(f"\nGeser skor maksimum (semua ukuran, kondisi, metrik): {geser_maks:.4f}\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
