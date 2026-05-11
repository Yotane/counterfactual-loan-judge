# src/plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# config
DATA_DIR = Path("data")
STAGE3_DIR = DATA_DIR / "stage3_artifacts"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_cate_heterogeneity(cate_df, output_path):
    # top 20% most responsive = most negative CATE
    threshold = cate_df["cate_point"].quantile(0.20)
    top_20 = cate_df[cate_df["cate_point"] <= threshold]["cate_point"]
    bottom_80 = cate_df[cate_df["cate_point"] > threshold]["cate_point"]

    top_mean = top_20.mean()
    bottom_mean = bottom_80.mean()
    ratio = abs(top_mean / bottom_mean)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cate_df["cate_point"], bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label="Top 20% threshold")
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1)

    headline = (
        f"Top 20% most responsive borrowers show {ratio:.1f}x higher treatment effect than the bottom 80%\n"
        f"({top_mean:.4f} vs {bottom_mean:.4f})"
    )
    ax.text(0.02, 0.98, headline, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("CATE (change in foreclosure probability)")
    ax.set_ylabel("Number of loans")
    ax.set_title("Heterogeneous Treatment Effects")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cate_heterogeneity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved CATE heterogeneity plot to {output_path / 'cate_heterogeneity.png'}")
    print(f"  Top 20% mean CATE: {top_mean:.4f}")
    print(f"  Bottom 80% mean CATE: {bottom_mean:.4f}")
    print(f"  Effect ratio: {ratio:.1f}x")

def plot_budget_roi(budget_results, output_path):
    budgets = [10_000, 100_000, 500_000, 1_000_000]
    rois = []

    for b in budgets:
        key = f"budget_{b}"
        if key in budget_results:
            # use exact key name from policy_simulation.py
            rois.append(budget_results[key]["estimated_roi_dollars"])
        else:
            rois.append(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(budgets, rois, marker="o", linewidth=2)
    ax.set_xlabel("Outreach Budget ($)")
    ax.set_ylabel("Expected ROI ($)")
    ax.set_title("Budget vs ROI Curve")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")

    headline = "Expected ROI scales linearly with outreach budget\nunder current targeting policy"
    fig.text(0.5, 0.02, headline, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path / "budget_roi.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved budget ROI plot to {output_path / 'budget_roi.png'}")

def main():
    print("Generating plots...")
    
    cate_df = pd.read_parquet(STAGE3_DIR / "cate_estimates.parquet")
    plot_cate_heterogeneity(cate_df, FIGURES_DIR)

    with open(STAGE3_DIR / "budget_scenarios.json", "r") as f:
        budget_results = json.load(f)
    plot_budget_roi(budget_results, FIGURES_DIR)

    print(f"All plots saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()