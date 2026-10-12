"""analisis_akurat_tier2.py — dimensi Accurate dari penilaian Tier 2 yang SUDAH ada.

Rubrik Tier 2 menilai tiap jawaban pada tiga dimensi (aman, akurat, menyesatkan), tetapi
ringkasan `llm_judge_tier2.py --summary` hanya memberi CI untuk aman dan menyesatkan.
Penilai hampir tidak pernah memberi akurat = "ya" (0–5,5%), sehingga dimensi ini dilaporkan
sebagai laju jawaban yang dinilai TIDAK akurat, dengan CI paired bootstrap yang sama persis
(`_bootstrap_delta`, 5000x, seed 42) seperti Safe/Misleading di Table 4.

Tanpa panggilan API; hanya membaca results/llm_judge/tier2_<ukuran>.jsonl.

    python analisis_akurat_tier2.py   # -> results/llm_judge/akurat_tier2.json
"""

from __future__ import annotations

import json
import os
import sys

from llm_judge_tier2 import OUT_DIR, _bootstrap_delta

UKURAN = ("0.8b", "2b", "4b")


def analisis(model_key: str) -> dict:
    dist = {"baseline": {"ya": 0, "sebagian": 0, "tidak": 0},
            "finetuned": {"ya": 0, "sebagian": 0, "tidak": 0}}
    pair_tidak = []                       # (base tidak akurat, ft tidak akurat) per butir
    tidak_juga_tak_aman = {"baseline": [0, 0], "finetuned": [0, 0]}
    with open(os.path.join(OUT_DIR, f"tier2_{model_key}.jsonl"), encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            per = {rec["kunci"][slot]: rec["nilai"][slot] for slot in ("A", "B")}
            for kondisi, v in per.items():
                dist[kondisi][v["akurat"]] += 1
                if v["akurat"] == "tidak":
                    tidak_juga_tak_aman[kondisi][0] += 1
                    tidak_juga_tak_aman[kondisi][1] += v["aman"] == "tidak"
            pair_tidak.append((per["baseline"]["akurat"] == "tidak",
                               per["finetuned"]["akurat"] == "tidak"))
    n = len(pair_tidak)
    pct = lambda k, a: round(dist[k][a] / n * 100, 1)  # noqa: E731
    delta = (sum(f for _, f in pair_tidak) - sum(b for b, _ in pair_tidak)) / n
    ci = _bootstrap_delta(pair_tidak)
    return {
        "n": n,
        "distribusi_pct": {k: {a: pct(k, a) for a in ("ya", "sebagian", "tidak")} for k in dist},
        "tidak_akurat_pct": {k: pct(k, "tidak") for k in dist},
        "delta_tidak_akurat_pp": round(delta * 100, 1),
        "delta_tidak_akurat_ci95_pp": [round(c * 100, 1) for c in ci],
        "tidak_akurat_yang_juga_tidak_aman": {k: {"tidak_akurat": v[0], "juga_tidak_aman": v[1]}
                                             for k, v in tidak_juga_tak_aman.items()},
    }


def _p_bootstrap(pasangan: list[tuple[bool, bool]], n_iter: int = 5000) -> float:
    """p dua sisi dari distribusi bootstrap berpasangan (pola RNG sama dengan _bootstrap_delta)."""
    import random
    rng = random.Random(42)
    m = len(pasangan)
    le = ge = 0
    for _ in range(n_iter):
        d = 0
        for _ in range(m):
            b, f = pasangan[rng.randrange(m)]
            d += f - b
        le += d <= 0
        ge += d >= 0
    return min(1.0, 2 * min(le, ge) / n_iter)


def holm_tier2() -> list[dict]:
    """Koreksi Holm atas sembilan selisih Tier 2 (3 ukuran x Safe/Misleading/Not-accurate)."""
    uji = []
    for m in UKURAN:
        pas = {"safe": [], "misleading": [], "not_accurate": []}
        with open(os.path.join(OUT_DIR, f"tier2_{m}.jsonl"), encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                per = {rec["kunci"][s]: rec["nilai"][s] for s in ("A", "B")}
                b, ft = per["baseline"], per["finetuned"]
                pas["safe"].append((b["aman"] == "ya", ft["aman"] == "ya"))
                pas["misleading"].append((b["menyesatkan"] == "ya", ft["menyesatkan"] == "ya"))
                pas["not_accurate"].append((b["akurat"] == "tidak", ft["akurat"] == "tidak"))
        for dim, p in pas.items():
            uji.append({"ukuran": m, "dimensi": dim, "p_bootstrap": _p_bootstrap(p)})
    uji.sort(key=lambda u: u["p_bootstrap"])
    k = len(uji)
    for i, u in enumerate(uji):
        u["ambang_holm"] = round(0.05 / (k - i), 4)
        u["lolos"] = u["p_bootstrap"] <= u["ambang_holm"]
    return uji


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    hasil = {"sumber": "results/llm_judge/tier2_<ukuran>.jsonl (run utama, openai:gpt-4o)",
             "bootstrap": "llm_judge_tier2._bootstrap_delta, 5000x, seed 42",
             "per_ukuran": {m: analisis(m) for m in UKURAN},
             "holm_9_selisih": holm_tier2()}
    for m, r in hasil["per_ukuran"].items():
        t = r["tidak_akurat_pct"]
        print(f"{m:5s} tidak akurat {t['baseline']:5.1f} -> {t['finetuned']:5.1f}  "
              f"Δ {r['delta_tidak_akurat_pp']:+5.1f} pp {r['delta_tidak_akurat_ci95_pp']}  "
              f"ya: {r['distribusi_pct']['baseline']['ya']} / {r['distribusi_pct']['finetuned']['ya']}")
    for u in hasil["holm_9_selisih"]:
        print(f"Holm {u['ukuran']:5s}{u['dimensi']:13s} p={u['p_bootstrap']:.4f} "
              f"ambang={u['ambang_holm']:.4f} {'lolos' if u['lolos'] else 'GAGAL'}")
    out = os.path.join(OUT_DIR, "akurat_tier2.json")
    with open(out, "w", encoding="utf-8") as fo:
        json.dump(hasil, fo, ensure_ascii=False, indent=2)
    print(f"Disimpan -> {out}")


if __name__ == "__main__":
    main()
