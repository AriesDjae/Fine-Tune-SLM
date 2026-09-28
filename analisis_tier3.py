"""
analisis_tier3.py — analisis validasi dokter (Tier 3) terhadap penilai GPT-4o (Tier 2).

Membaca penilaian dokter dari dataset privat `AriesDjaenuri/p6-penilaian-dokter`
(`validasi/*.json`, ditulis Tugas 1 di Space), membuka tirai lewat
human_eval/validasi_indeks.json (id -> idx_test, ukuran, kunci A/B), lalu membandingkannya
dengan results/llm_judge/tier2_<ukuran>.jsonl pada pasangan yang SAMA.

Keluaran (results/tier3/ringkasan.json + cetak):
  1. Kesepakatan dokter<->GPT-4o per kolom rubrik (persentase setuju + Cohen's kappa):
     Aman, Menyesatkan (biner), Akurat (3 kelas), dan preferensi A/B/seri.
  2. Kesepakatan dokter<->dokter pada pasangan ganda (target 2), sebagai pembanding: bila
     kappa dokter<->GPT-4o mendekati kappa dokter<->dokter, GPT-4o setara seorang dokter lain.
  3. Delta Safe rate (fine-tuned - baseline) menurut DOKTER per ukuran + bootstrap 95%,
     berdampingan dengan GPT-4o pada butir yang sama: apakah dokter mengonfirmasi
     pembalikan tanda (0.8B naik, 2B/4B turun).

Penilaian ber-flag `uji` dibuang. Pada pasangan ganda, tiap penilaian dokter dihitung
sebagai satu pengamatan untuk (1) dan (3); (2) memakai dua penilai pada id yang sama.

    python analisis_tier3.py            # unduh dari dataset (butuh login HF) lalu analisis
    python analisis_tier3.py --lokal D  # baca validasi/*.json dari folder D
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF_DATASET = "AriesDjaenuri/p6-penilaian-dokter"
INDEKS = os.path.join("human_eval", "validasi_indeks.json")
KOLOM = ["aman", "akurat", "menyesatkan"]
UKURAN = ["0.8b", "2b", "4b"]
KELUAR = os.path.join("results", "tier3", "ringkasan.json")


def kappa(pasangan: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa tak berbobot untuk dua penilai pada label kategoris."""
    n = len(pasangan)
    if n == 0:
        return None
    po = sum(a == b for a, b in pasangan) / n
    ka, kb = Counter(a for a, _ in pasangan), Counter(b for _, b in pasangan)
    pe = sum(ka[k] * kb[k] for k in set(ka) | set(kb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def setuju(pasangan):
    return sum(a == b for a, b in pasangan) / len(pasangan) if pasangan else None


def muat_dokter(folder: str | None) -> list[dict]:
    if folder is None:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(HF_DATASET, repo_type="dataset",
                                   allow_patterns=["validasi/*.json"])
    berkas = (glob.glob(os.path.join(folder, "validasi", "*.json"))
              or glob.glob(os.path.join(folder, "*.json")))
    rek = []
    for b in berkas:
        with open(b, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("uji"):
            continue
        for n in d["nilai"]:
            rek.append({**n, "email": d["email"].lower(), "penilai": d["penilai"]})
    return rek


def muat_tier2(u: str) -> dict[int, dict]:
    with open(os.path.join("results", "llm_judge", f"tier2_{u}.jsonl"), encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, f) if r.get("nilai")}


def per_kondisi(nilai_ab: dict, kunci: dict) -> dict:
    """{'A': {...}, 'B': {...}} + kunci A/B -> {'baseline': {...}, 'finetuned': {...}}"""
    return {kunci["A"]: nilai_ab["A"], kunci["B"]: nilai_ab["B"]}


def bootstrap_delta(pasangan: list[tuple[bool, bool]], n_iter: int = 5000, seed: int = 42):
    rng = random.Random(seed)
    n = len(pasangan)
    titik = sum(f - b for b, f in pasangan) / n * 100
    sebaran = sorted(sum(f - b for b, f in (pasangan[rng.randrange(n)] for _ in range(n))) / n * 100
                     for _ in range(n_iter))
    return titik, sebaran[int(0.025 * n_iter)], sebaran[int(0.975 * n_iter)]


def cetak(judul: str, tabel: dict) -> None:
    print(f"\n{judul}")
    for k, v in tabel.items():
        kp = "—" if v["kappa"] is None else f"{v['kappa']:.2f}"
        print(f"  {k:12s} n={v['n']:4d}  setuju {v['setuju']:.1%}  κ={kp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lokal", default=None, help="folder berisi validasi/*.json")
    a = ap.parse_args()

    indeks = {p["id"]: p for p in json.load(open(INDEKS, encoding="utf-8"))["pasangan"]}
    tier2 = {u: muat_tier2(u) for u in UKURAN}
    dokter = [r for r in muat_dokter(a.lokal) if r["id"] in indeks]
    if not dokter:
        print("Belum ada penilaian dokter (non-uji).")
        return
    print(f"{len(dokter)} penilaian dari {len({r['email'] for r in dokter})} dokter, "
          f"{len({r['id'] for r in dokter})} pasangan berbeda")

    # 1. dokter <-> GPT-4o
    dg = defaultdict(list)
    for r in dokter:
        p = indeks[r["id"]]
        g = tier2[p["ukuran"]][p["idx_test"]]["nilai"]
        for sisi in ("A", "B"):
            for k in KOLOM:
                dg[k].append((r[sisi][k], g[sisi][k]))
        dg["lebih_baik"].append((r["lebih_baik"], g["lebih_baik"]))
    hasil = {"n_penilaian": len(dokter), "n_dokter": len({r["email"] for r in dokter}),
             "dokter_vs_gpt4o": {k: {"n": len(v), "setuju": setuju(v), "kappa": kappa(v)}
                                 for k, v in dg.items()}}
    cetak("Dokter ↔ GPT-4o (per jawaban; preferensi per pasangan)", hasil["dokter_vs_gpt4o"])

    # 2. dokter <-> dokter (pasangan ganda)
    per_id = defaultdict(list)
    for r in dokter:
        per_id[r["id"]].append(r)
    dd = defaultdict(list)
    for rs in per_id.values():
        if len(rs) < 2:
            continue
        r1, r2 = rs[0], rs[1]
        for sisi in ("A", "B"):
            for k in KOLOM:
                dd[k].append((r1[sisi][k], r2[sisi][k]))
        dd["lebih_baik"].append((r1["lebih_baik"], r2["lebih_baik"]))
    hasil["dokter_vs_dokter"] = {k: {"n": len(v), "setuju": setuju(v), "kappa": kappa(v)}
                                 for k, v in dd.items()}
    if dd:
        cetak("Dokter ↔ dokter (pasangan ganda)", hasil["dokter_vs_dokter"])

    # 3. Delta Safe menurut dokter per ukuran, dan GPT-4o pada butir yang sama
    hasil["delta_aman"] = {}
    print("\nΔ Safe rate (fine-tuned − baseline) pada pasangan yang dinilai dokter")
    for u in UKURAN:
        pd, pg = [], []
        for r in dokter:
            p = indeks[r["id"]]
            if p["ukuran"] != u:
                continue
            d = per_kondisi(r, p["kunci"])
            g = per_kondisi(tier2[u][p["idx_test"]]["nilai"], p["kunci"])
            pd.append((d["baseline"]["aman"] == "ya", d["finetuned"]["aman"] == "ya"))
            pg.append((g["baseline"]["aman"] == "ya", g["finetuned"]["aman"] == "ya"))
        if not pd:
            continue
        td, lo, hi = bootstrap_delta(pd)
        tg, glo, ghi = bootstrap_delta(pg)
        hasil["delta_aman"][u] = {"n": len(pd), "dokter": [td, lo, hi],
                                  "gpt4o_butir_sama": [tg, glo, ghi]}
        print(f"  {u:5s} n={len(pd):3d}  dokter {td:+6.1f} pp [{lo:+.1f}, {hi:+.1f}]"
              f"   GPT-4o {tg:+6.1f} pp [{glo:+.1f}, {ghi:+.1f}]")

    os.makedirs(os.path.dirname(KELUAR), exist_ok=True)
    with open(KELUAR, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=1)
    print(f"\ntersimpan: {KELUAR}")


if __name__ == "__main__":
    main()
