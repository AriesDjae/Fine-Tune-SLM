"""
plot_pipeline_v2.py — GRAFIK pipeline dataset Pivot 4 (strict_clean_v2), dari log JSON NYATA.

Membaca output tahap-tahap v2 dan menghasilkan PNG (dpi=150) di results/figures/:
  V.1 fig_v2_funnel        — jumlah sampel per aturan strict_clean_v2 (funnel bar)
  V.2 fig_v2_domain        — distribusi domain per split (grouped bar)
  V.3 fig_v2_answer_chars  — panjang jawaban (median/min/max) per split
  V.4 fig_v2_tokenlen      — percentile panjang token per model (p50/p90/p95/p99/max)
  V.5 fig_v2_verify        — sisa artefak per kategori dari verifier independen (target 0%)

Robust: tahap yang lognya belum ada akan di-skip dengan pesan. Tidak butuh GPU.
Pakai:  python preprocessing/plot_pipeline_v2.py
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "Data" / "processed_id"

C = dict(green="#1D9E75", blue="#378ADD", orange="#BA7517", red="#E24B4A",
         purple="#534AB7", teal="#0F6E56", grey="#8A8F98")
SPLIT_C = {"train": C["blue"], "val": C["orange"], "test": C["green"]}


def latest(pattern):
    files = sorted(glob.glob(str(RESULTS / pattern)))
    return Path(files[-1]) if files else None


def save(fig, name):
    p = FIGDIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


def plot_funnel():
    f = latest("strict_clean_v2_log_*.json")
    if not f:
        print("[V.1] skip — belum ada strict_clean_v2_log_*.json")
        return
    log = json.loads(f.read_text(encoding="utf-8"))
    rules = [r for r in log.get("rules", [])]
    if not rules:
        print("[V.1] skip — log tanpa rules")
        return
    labels = [r["rule"] for r in rules]
    ins = [r["in"] for r in rules]
    outs = [r["out"] for r in rules]
    sel = log.get("selection", {}).get("selected")
    if sel is not None:
        labels.append("selected"); ins.append(sel); outs.append(sel)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    ax.bar(x, ins, color=C["grey"], alpha=0.5, label="masuk (in)")
    ax.bar(x, outs, color=C["blue"], label="lolos (out)")
    for i, (a, b) in enumerate(zip(ins, outs)):
        if a > b:
            ax.annotate(f"-{a-b:,}", (i, b), ha="center", va="bottom",
                        fontsize=8, color=C["red"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("jumlah sampel")
    ax.set_title(f"V.1 Funnel strict_clean_v2  (sumber: {f.name})")
    ax.legend()
    save(fig, "fig_v2_funnel.png")


def _stats():
    p = OUT_DIR / "dataset_stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def plot_domain():
    st = _stats()
    if not st:
        print("[V.2] skip — belum ada dataset_stats.json")
        return
    splits = st.get("splits", {})
    domains = sorted({d for s in splits.values() for d in s.get("domain_distribution", {})})
    if not domains:
        print("[V.2] skip — tanpa domain")
        return
    x = np.arange(len(domains)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.1), 5))
    for i, sp in enumerate(("train", "val", "test")):
        dd = splits.get(sp, {}).get("domain_distribution", {})
        ax.bar(x + (i - 1) * w, [dd.get(d, 0) for d in domains], w,
               label=sp, color=SPLIT_C[sp])
    ax.set_xticks(x); ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("jumlah"); ax.set_title("V.2 Distribusi domain per split")
    ax.legend()
    save(fig, "fig_v2_domain.png")


def plot_answer_chars():
    st = _stats()
    if not st:
        print("[V.3] skip — belum ada dataset_stats.json")
        return
    splits = st.get("splits", {})
    names = [s for s in ("train", "val", "test") if s in splits]
    med = [splits[s]["answer_chars"].get("median", 0) for s in names]
    mn = [splits[s]["answer_chars"].get("min", 0) for s in names]
    mx = [splits[s]["answer_chars"].get("max", 0) for s in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, med, color=[SPLIT_C[s] for s in names], label="median")
    for i, s in enumerate(names):
        ax.annotate(f"min {mn[i]}\nmax {mx[i]}", (i, med[i]), ha="center",
                    va="bottom", fontsize=8, color=C["grey"])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("panjang jawaban (char)")
    ax.set_title("V.3 Panjang jawaban per split (median; min/max anotasi)")
    save(fig, "fig_v2_answer_chars.png")


def plot_tokenlen():
    f = latest("token_length_*.json")
    if not f:
        print("[V.4] skip — belum ada token_length_*.json")
        return
    rep = json.loads(f.read_text(encoding="utf-8"))
    models = {k: v for k, v in rep.get("models", {}).items() if v.get("status") == "ok"}
    if not models:
        print("[V.4] skip — tak ada tokenizer ok")
        return
    pcts = ["median", "p90", "p95", "p99", "max"]
    x = np.arange(len(pcts)); w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C["blue"], C["orange"], C["green"], C["purple"]]
    for i, (name, m) in enumerate(models.items()):
        ax.bar(x + i * w, [m.get(p, 0) for p in pcts], w, label=name, color=colors[i % 4])
    rec = rep.get("recommendation", {}).get("MAX_SEQ_LENGTH")
    if rec:
        ax.axhline(rec, color=C["red"], ls="--", lw=1.2, label=f"MAX_SEQ_LENGTH={rec}")
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(pcts)
    ax.set_ylabel("panjang token"); ax.set_title("V.4 Panjang token per model (chat template)")
    ax.legend(fontsize=8)
    save(fig, "fig_v2_tokenlen.png")


def plot_verify():
    f = latest("verify_v2_*.json")
    if not f:
        print("[V.5] skip — belum ada verify_v2_*.json")
        return
    rep = json.loads(f.read_text(encoding="utf-8"))
    splits = rep.get("splits", {})
    cats = [k for k in next(iter(splits.values())).keys()
            if not k.endswith("_pct") and k not in ("n",)]
    names = list(splits.keys())
    x = np.arange(len(cats)); w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(max(9, len(cats) * 0.9), 5))
    for i, sp in enumerate(names):
        ax.bar(x + i * w, [splits[sp].get(c, 0) for c in cats], w,
               label=sp, color=SPLIT_C.get(sp, C["grey"]))
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("sisa (count)")
    ax.set_title(f"V.5 Sisa artefak per kategori — verifier independen (target 0)  ({f.name})")
    ax.legend()
    save(fig, "fig_v2_verify.png")


def main():
    print("Membuat grafik pipeline v2 -> results/figures/")
    plot_funnel()
    plot_domain()
    plot_answer_chars()
    plot_tokenlen()
    plot_verify()
    print("selesai.")


if __name__ == "__main__":
    main()
