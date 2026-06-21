"""
plot_loss_curve.py - Kurva loss train/val Qwen3.5-0.8B (Pivot 5) untuk bahan bimbingan.
Sumber: results/qwen35_0_8b_trainer_state.json (5628 step / 3 epoch).
Output: results/figures/qwen35_0_8b_loss_curve.png
"""
import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "qwen35_0_8b_trainer_state.json")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

d = json.load(open(STATE, encoding="utf-8"))
lh = d["log_history"]
tr = [(e["step"], e["loss"]) for e in lh if "loss" in e]
ev = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]

best_step, best_eval = min(ev, key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot([s for s, _ in tr], [l for _, l in tr],
        color="#1f77b4", lw=1.0, alpha=0.85, label="Training loss")
ax.plot([s for s, _ in ev], [l for _, l in ev],
        color="#d62728", lw=1.8, marker="o", ms=3, label="Validation loss")
ax.scatter([best_step], [best_eval], s=90, facecolors="none",
           edgecolors="#2ca02c", linewidths=2, zorder=5,
           label=f"Best (step {best_step}, eval_loss={best_eval:.4f})")
ax.annotate(f"best = {best_eval:.4f}", (best_step, best_eval),
            textcoords="offset points", xytext=(-70, 14),
            color="#2ca02c", fontsize=9)

# garis batas epoch (3 epoch, 5628 step => 1876/epoch)
spe = d.get("global_step", 5628) / max(d.get("epoch", 3.0), 1e-9)
for i in range(1, 3):
    ax.axvline(i * spe, color="grey", ls="--", lw=0.7, alpha=0.5)
    ax.text(i * spe, ax.get_ylim()[1], f" epoch {i}", color="grey",
            fontsize=8, va="top")

ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title("Qwen3.5-0.8B QLoRA — Training vs Validation Loss (3 epoch)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()

out = os.path.join(OUTDIR, "qwen35_0_8b_loss_curve.png")
fig.savefig(out, dpi=150)
print("saved:", out)
print(f"train: {len(tr)} pts ({tr[0][1]:.4f} -> {tr[-1][1]:.4f})")
print(f"eval : {len(ev)} pts ({ev[0][1]:.4f} -> {ev[-1][1]:.4f}), best step {best_step}")
