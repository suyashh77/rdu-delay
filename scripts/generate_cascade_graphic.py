"""Generate mobile-optimized, publication-ready aircraft rotation cascade graphic.
Explicitly labels real historical BTS observations vs. night-before predictive signals.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BRAIN_DIR = Path(r"C:\Users\19193\.gemini\antigravity-ide\brain\fee4234f-9bb8-44d8-9516-eae7bb3b8b2d")
ROOT = Path(__file__).resolve().parents[1]

# Set up figure (16:9 aspect ratio optimized for LinkedIn and mobile viewing)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
fig.patch.set_facecolor("#ffffff")

# Real Aircraft Data from BTS Form 41 Database:
# Airframe N17245 (United Airlines Boeing 737-924ER, Flight Date 2023-01-01)
legs = [
    {
        "leg": "Leg 1: MSP → ORD\nUA 1483 (Morning)",
        "sched_dep": 6.25, "sched_arr": 7.82, "act_dep": 6.58, "act_arr": 8.03,
        "dep_d": 20, "arr_d": 13, "buffer": 0, "status": "Minor Delay (+13m)"
    },
    {
        "leg": "Leg 2: ORD → SLC\nUA 2049 (Midday)",
        "sched_dep": 8.92, "sched_arr": 11.62, "act_dep": 9.22, "act_arr": 12.05,
        "dep_d": 18, "arr_d": 26, "buffer": 66, "status": "Minor Delay (+26m)"
    },
    {
        "leg": "Leg 3: SLC → ORD\nUA 1754 (Afternoon)",
        "sched_dep": 12.62, "sched_arr": 16.77, "act_dep": 14.20, "act_arr": 19.10,
        "dep_d": 95, "arr_d": 140, "buffer": 60, "status": "Severe Delay (+140m)"
    },
    {
        "leg": "Leg 4: ORD → RDU\nUA 2433 (Target Flight)",
        "sched_dep": 18.30, "sched_arr": 21.28, "act_dep": 20.07, "act_arr": 22.95,
        "dep_d": 106, "arr_d": 100, "buffer": 92, "status": "Cascaded Delay (+100m)"
    },
]

y_pos = [3, 2, 1, 0]
h = 0.32

# 1. Plot Scheduled Windows and Actual Flights
for idx, (l, y) in enumerate(zip(legs, y_pos)):
    # Scheduled block (clean slate blue)
    ax.barh(y + h/2, l["sched_arr"] - l["sched_dep"], left=l["sched_dep"], height=h,
            color="#94a3b8", alpha=0.9, edgecolor="#334155", linewidth=1.2,
            label="Scheduled Flight Window (Timetable)" if idx == 0 else "")
    
    # Actual block (Color-coded by delay severity)
    color = "#dc2626" if l["arr_d"] >= 60 else ("#f59e0b" if l["arr_d"] >= 15 else "#10b981")
    edge = "#991b1b" if l["arr_d"] >= 60 else ("#b45309" if l["arr_d"] >= 15 else "#065f46")
    ax.barh(y - h/2, l["act_arr"] - l["act_dep"], left=l["act_dep"], height=h,
            color=color, alpha=0.92, edgecolor=edge, linewidth=1.2,
            label="Actual Flight Flown (Observed Delay)" if idx == 0 else "")

    # Buffer connections between consecutive legs
    if idx < len(legs) - 1:
        next_l = legs[idx + 1]
        next_y = y_pos[idx + 1]
        buf_start = l["sched_arr"]
        buf_end = next_l["sched_dep"]
        ax.plot([buf_start, buf_end], [y + h/2, next_y + h/2], color="#4f46e5", linestyle=":", linewidth=2.2)
        
        status_text = "Absorbed" if l["arr_d"] <= next_l["buffer"] else "EXHAUSTED"
        ax.annotate(f"Scheduled Turnaround: {next_l['buffer']} min\n({status_text})",
                    xy=((buf_start + buf_end)/2, (y + next_y)/2 + 0.12),
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="#1e1b4b" if status_text == "Absorbed" else "#991b1b",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="#e0e7ff" if status_text == "Absorbed" else "#fee2e2",
                              edgecolor="#818cf8" if status_text == "Absorbed" else "#f87171",
                              lw=1.2))

    # Exact delay text on actual bar
    ax.annotate(f"Dep: {l['dep_d']:+d}m | Arr: {l['arr_d']:+d}m",
                xy=(l["act_arr"], y - h/2), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", fontsize=11, fontweight="bold",
                color="#991b1b" if l["arr_d"] >= 60 else "#047857")

# 2. Strict Point-in-Time Cutoff Line at 11:00 PM ET (Day D-1)
ax.axvline(x=4.0, color="#0f172a", linestyle="--", linewidth=2.5, alpha=0.9, zorder=5)
ax.annotate("11:00 PM ET CUTOFF (Day D-1)\nForecast Generated Here\n(Zero Future Data Leaked)",
            xy=(4.0, 3.1), xytext=(18, -12), textcoords="offset points",
            fontsize=10.5, fontweight="heavy", color="#0f172a",
            arrowprops=dict(arrowstyle="->", color="#0f172a", lw=2),
            bbox=dict(boxstyle="square,pad=0.45", facecolor="#fef2f2", edgecolor="#ef4444", lw=1.5),
            zorder=6)

# 3. Model Signals Card (Large, High-Contrast, Explaining the Engineering)
signals_text = (
    "WHAT OUR LIGHTGBM MODEL EXTRACTS AT 11:00 PM BEFORE BED:\n"
    "• Overnight Tail Position: Airframe slept at MSP (no residual maintenance hold)\n"
    "• Rotation Depth: Target RDU flight is Leg 4 of 4 across a multi-hub network\n"
    "• Turnaround Slack: 66m (ORD) and 60m (SLC) buffers; vulnerable to compounded hold\n"
    "• Upstream Hub Weather: High-res convective forecast (TAF/HRRR) at ORD in late afternoon\n"
    "-> Result: Model accurately shifts probability mass to Severe Delay (>90m) into RDU"
)
ax.text(0.02, 0.04, signals_text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
        verticalalignment="bottom", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#4f46e5", lw=1.8),
        zorder=6)

# 4. Provenance Badge (Placed cleanly at bottom-right to avoid legend collision)
prov_text = (
    "DATA PROVENANCE: 100% Real BTS Flight Records\n"
    "United Airlines Tail N17245 (Boeing 737) on Jan 1, 2023\n"
    "Observed observations from study dataset (Form 41)"
)
ax.text(0.98, 0.04, prov_text, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
        horizontalalignment="right", verticalalignment="bottom", color="#334155",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2),
        zorder=6)

# Format axes
ax.set_ylim(-0.75, 3.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([l["leg"] for l in legs], fontsize=11.5, fontweight="bold", color="#0f172a")
ax.set_xlabel("Time of Day (Local Hours, 24-Hour Clock)", fontsize=12, fontweight="bold", labelpad=8)
ax.set_xlim(3.2, 24.8)
ax.set_xticks(range(4, 25, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(4, 25, 2)], fontsize=10.5, fontweight="bold")
ax.set_title("How Flight Delays Cascade Across an Aircraft Rotation Chain\nWhy Single-Airport Models Miss Cascading Delays Born Three Cities Away",
             fontsize=14, fontweight="bold", pad=16, color="#0f172a")

ax.legend(loc="upper right", frameon=True, fontsize=10.5, facecolor="#ffffff", edgecolor="#cbd5e1")
plt.tight_layout()

# Save outputs
out_brain = BRAIN_DIR / "rdu_delay_cascade_rotation.png"
out_docs = ROOT / "docs" / "rdu_delay_cascade_rotation.png"
plt.savefig(out_brain, dpi=300, bbox_inches="tight")
plt.savefig(out_docs, dpi=300, bbox_inches="tight")
print(f"Saved mobile-optimized cascade graphic to:\n1. {out_brain}\n2. {out_docs}")
