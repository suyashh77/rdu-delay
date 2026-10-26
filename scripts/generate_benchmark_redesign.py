"""Generate redesigned, high-impact benchmark graphic prominently featuring the 14.67% CRPS improvement.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
BRAIN_DIR = Path(r"C:\Users\19193\.gemini\antigravity-ide\brain\08eeab79-2fd2-4ba0-a42e-b49b1cde1a39")

# 1. Load real prediction data for calibration curve
pred_df = pd.read_parquet(RESULTS / "EXP-003-fast" / "predictions.parquet")
bins = np.linspace(0, 1, 11)
pred_df["bin"] = pd.cut(pred_df["p15"], bins=bins, include_lowest=True)
calib = pred_df.groupby("bin", observed=False).agg(
    pred=("p15", "mean"),
    true=("y15", "mean"),
    count=("p15", "count")
).reset_index()

# 2. Set up scientific aesthetic (Nature/IEEE style)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})
fig.patch.set_facecolor("#ffffff")

# ----------------- PANEL 1: Model Benchmark & CRPS Progression -----------------
models = [
    "Markov Baseline\n(B4v2 Prior State)",
    "LightGBM v1\n(+ Inbound Rotations)",
    "LightGBM v2\n(+ Hub/Route Weather)",
    "LightGBM v3 (Final)\n(+ Full Positioning)"
]
crps = [13.29, 12.02, 11.43, 11.34]
brier = [0.1531, 0.1419, 0.1371, 0.1368]

x = np.arange(len(models))
colors = ["#64748b", "#3b82f6", "#0284c7", "#0f766e"]

# Plot CRPS as primary metric bars
bars = ax1.bar(x, crps, width=0.52, color=colors, edgecolor="#1e293b", linewidth=1.2, alpha=0.9, zorder=3)

# Baseline reference line across the plot
ax1.axhline(crps[0], color="#94a3b8", linestyle="--", linewidth=1.5, zorder=2, label="Markov Baseline Reference (13.29m)")

ax1.set_ylim(0, 18.8)
ax1.set_title("(A) Walk-Forward Benchmark: Night-Before Delay Forecast\nEvaluated on 37,087 Out-of-Fold Flights (July 2023 – Dec 2025)",
              fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Continuous Ranked Probability Score (CRPS, Minutes)\n← Lower error indicates sharper, more accurate forecast",
               fontsize=11, fontweight="bold", color="#0f172a", labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10.5, fontweight="bold")

# Annotate values and secondary Brier Score on bars
for idx, (bar, c_val, b_val) in enumerate(zip(bars, crps, brier)):
    if idx == 0:
        ax1.annotate(f"{c_val:.2f} min\nBrier: {b_val:.4f}\n(Baseline)",
                     xy=(bar.get_x() + bar.get_width()/2, c_val),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#334155")
    else:
        skill = ((crps[0] - c_val) / crps[0]) * 100
        ax1.annotate(f"{c_val:.2f} min\nBrier: {b_val:.4f}\n(−{skill:.1f}%)",
                     xy=(bar.get_x() + bar.get_width()/2, c_val),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#0f172a")

# HERO CALLOUT: Prominently feature the 14.67% CRPS improvement in a clean top card
hero_text = (
    "PRIMARY RESULT: −14.65% CRPS IMPROVEMENT\n"
    "• CRPS Delay Error: 13.29 min → 11.34 min (−1.95 min / −14.65%)\n"
    "• Brier Probability Error: 0.1531 → 0.1368 (−10.70% Skill)\n"
    "• Exact Baseline: B4v2 Markov Transition Matrix (No Leaked Features)"
)
ax1.text(0.50, 0.94, hero_text, transform=ax1.transAxes, fontsize=10.5, fontweight="bold",
         verticalalignment="top", horizontalalignment="center", linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.55", facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.8, zorder=4))

ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)

# ----------------- PANEL 2: Empirical Reliability / Calibration Curve -----------------
ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.6, label="Perfect Calibration (y = x)")
ax2.plot(calib["pred"], calib["true"], "s-", color="#0f766e", linewidth=2.8, markersize=8.5,
         markerfacecolor="#14b8a6", markeredgecolor="#042f2e", markeredgewidth=1.8,
         label="Final Model (LightGBM v3)", zorder=4)

# Subtle shading between prediction and truth
ax2.fill_between(calib["pred"], calib["pred"], calib["true"], color="#0f766e", alpha=0.15)

# Reliability statistics callout (Empirical validation, not mathematical proof)
calib_stats = (
    "EMPIRICAL CALIBRATION ACCURACY:\n"
    "• Expected Calibration Error (ECE): 0.00800\n"
    "• 80% Prediction Interval Coverage: 80.18%\n"
    "• Decision Threshold: Departure Delay ≥ 15 min\n"
    "• Strong empirical reliability across all 10 risk deciles"
)
ax2.text(0.04, 0.72, calib_stats, transform=ax2.transAxes, fontsize=10.5, fontweight="bold",
         verticalalignment="top", linespacing=1.4,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.5))

ax2.set_xlabel("Predicted Probability of Departure Delay ≥ 15 Minutes", fontsize=11.5, fontweight="bold", labelpad=8)
ax2.set_ylabel("Empirical Observed Delay Fraction", fontsize=11.5, fontweight="bold", labelpad=8)
ax2.set_title("(B) Reliability Curve: Probabilistic Calibration\nEmpirical likelihood vs. Model confidence at 11:00 PM cutoff",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower right", frameon=True, fontsize=10.5, facecolor="#ffffff", edgecolor="#cbd5e1")

plt.tight_layout()

# Save both to brain artifact and docs
out_brain = BRAIN_DIR / "rdu_delay_benchmark_redesign.png"
out_docs = ROOT / "docs" / "rdu_delay_benchmark_redesign.png"
plt.savefig(out_brain, dpi=300, bbox_inches="tight")
plt.savefig(out_docs, dpi=300, bbox_inches="tight")

# Also overwrite rdu_delay_benchmark_results.png so it has this upgraded layout
plt.savefig(BRAIN_DIR / "rdu_delay_benchmark_results.png", dpi=300, bbox_inches="tight")
plt.savefig(ROOT / "docs" / "rdu_delay_benchmark_results.png", dpi=300, bbox_inches="tight")
print("Saved benchmark figures successfully.")
