"""Generate publication-grade benchmark and calibration figures from real evaluation outputs.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
BRAIN_DIR = Path(r"C:\Users\19193\.gemini\antigravity-ide\brain\fee4234f-9bb8-44d8-9516-eae7bb3b8b2d")

# 1. Load real prediction data
pred_df = pd.read_parquet(RESULTS / "EXP-003-fast" / "predictions.parquet")

# 2. Calibration computation
bins = np.linspace(0, 1, 11)
pred_df["bin"] = pd.cut(pred_df["p15"], bins=bins, include_lowest=True)
calib = pred_df.groupby("bin", observed=False).agg(
    pred=("p15", "mean"),
    true=("y15", "mean"),
    count=("p15", "count")
).reset_index()

# Set up matplotlib aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
fig.patch.set_facecolor("#ffffff")

# ----------------- PANEL 1: Model Progression -----------------
ax1 = axes[0]
models = ["B4v2 Markov\n(Baseline)", "EXP-001b\n(+ Rotations)", "EXP-002\n(+ Weather)", "EXP-003\n(+ Positioning)"]
brier = [0.1531, 0.1419, 0.1371, 0.1366]
crps = [13.29, 12.02, 11.43, 11.34]

x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, brier, width, label="Brier Score (prob error)", color="#4338ca", alpha=0.9, edgecolor="#312e81")
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, crps, width, label="CRPS (delay min error)", color="#0ea5e9", alpha=0.9, edgecolor="#0284c7")

ax1.set_ylabel("Brier Score (lower is better)", color="#4338ca", fontsize=11, fontweight="bold")
ax1_twin.set_ylabel("CRPS Error in Minutes (lower is better)", color="#0ea5e9", fontsize=11, fontweight="bold")
ax1.set_title("(A) Model Progression vs. Baseline\n(37,087 walk-forward flights)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.5)
ax1.set_ylim(0.12, 0.165)
ax1_twin.set_ylim(9.5, 14.5)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1_twin.grid(False)

# Annotate bars
for r, b in zip(rects1, brier):
    ax1.annotate(f"{b:.4f}", xy=(r.get_x() + r.get_width()/2, r.get_height()), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#312e81")
for r, c, skill in zip(rects2, crps, ["0.0%", "-9.6%", "-14.0%", "-14.7%"]):
    ax1_twin.annotate(f"{c:.2f}m\n({skill})", xy=(r.get_x() + r.get_width()/2, r.get_height()), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0369a1")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=8.5)

# ----------------- PANEL 2: Reliability / Calibration Diagram -----------------
ax2 = axes[1]
ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration (y = x)")
ax2.plot(calib["pred"], calib["true"], "s-", color="#6366f1", linewidth=2.5, markersize=8,
         markerfacecolor="#4f46e5", markeredgecolor="#ffffff", markeredgewidth=1.5, label="Final Model (EXP-003)")

# Fill deviation
ax2.fill_between(calib["pred"], calib["pred"], calib["true"], color="#6366f1", alpha=0.15)

ax2.set_xlabel("Predicted Probability of Delay ≥ 15m", fontsize=11, fontweight="bold")
ax2.set_ylabel("Observed Delay Fraction", fontsize=11, fontweight="bold")
ax2.set_title("(B) Reliability Curve: Empirical Calibration\n(ECE = 0.00786 | 80% Coverage = 80.12%)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle="--", alpha=0.5)

# Add sample counts annotation
ax2.text(0.05, 0.85, "Expected Calibration Error:\nECE = 0.00786\n80% Interval Coverage: 80.12%\nEvaluated on 37,087 flights",
         transform=ax2.transAxes, fontsize=9.5, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))
ax2.legend(loc="lower right", frameon=True, fontsize=9)

# ----------------- PANEL 3: Time-of-Day Physical Impact -----------------
ax3 = axes[2]
slices = ["Early Morning\n(05:00–08:59 AM)\nOvernight Position", "Evening\n(18:00–23:59 PM)\nHub Weather"]
base_crps = [8.56, 16.48]
final_crps = [7.57, 14.81]
deltas = ["-0.99 min\n(-11.6%)", "-1.67 min\n(-10.1%)"]

x_s = np.arange(len(slices))
w_s = 0.32

r_b = ax3.bar(x_s - w_s/2, base_crps, w_s, label="Markov Baseline", color="#94a3b8", alpha=0.9, edgecolor="#64748b")
r_f = ax3.bar(x_s + w_s/2, final_crps, w_s, label="Final Model (EXP-003)", color="#10b981", alpha=0.9, edgecolor="#059669")

ax3.set_ylabel("CRPS Error in Minutes", fontsize=11, fontweight="bold")
ax3.set_title("(C) Physical Feature Mechanism\nMorning Position vs. Evening Weather", fontsize=12, fontweight="bold", pad=12)
ax3.set_xticks(x_s)
ax3.set_xticklabels(slices, fontsize=9.5)
ax3.set_ylim(0, 19)
ax3.grid(axis="y", linestyle="--", alpha=0.5)

for r, val in zip(r_b, base_crps):
    ax3.annotate(f"{val:.2f}m", xy=(r.get_x() + r.get_width()/2, r.get_height()), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#475569")
for r, val, d in zip(r_f, final_crps, deltas):
    ax3.annotate(f"{val:.2f}m\n{d}", xy=(r.get_x() + r.get_width()/2, r.get_height()), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

ax3.legend(loc="upper left", frameon=True, fontsize=9)

plt.tight_layout()

# Save outputs
out_brain = BRAIN_DIR / "rdu_delay_benchmark_results.png"
out_docs = ROOT / "docs" / "rdu_delay_benchmark_results.png"

plt.savefig(out_brain, dpi=300, bbox_inches="tight")
plt.savefig(out_docs, dpi=300, bbox_inches="tight")
print(f"Saved figure to:\n1. {out_brain}\n2. {out_docs}")
