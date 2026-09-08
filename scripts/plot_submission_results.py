"""Generate the frozen, publication-ready validation figures for the submission."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_utils import (
    BLUE,
    GREEN,
    GREY,
    LIGHT_GREY,
    ORANGE,
    apply_publication_style,
    save_figure,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs" / "submission"
FIGURES = ROOT / "submission" / "assets" / "figures"
SIXDIM_NAMES = ["Safe", "Lively", "Beautiful", "Wealthy", "Boring", "Depressing"]


def _sixdim_validation() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "urban_perception_sixdim" / "validation_report.csv")


def plot_sixdim_agreement() -> None:
    validation = _sixdim_validation().copy()
    validation["dimension"] = validation["target"].str.removeprefix("vlm_").str.title()
    validation = validation.set_index("dimension").loc[SIXDIM_NAMES].reset_index()
    y = np.arange(len(validation))

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    lower = validation["pearson_r"] - validation["pearson_ci_low"]
    upper = validation["pearson_ci_high"] - validation["pearson_r"]
    ax.errorbar(
        validation["pearson_r"],
        y - 0.11,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=2.5,
        label="Pearson r (95% CI)",
    )
    ax.scatter(
        validation["spearman_rho"],
        y + 0.11,
        color=ORANGE,
        marker="s",
        s=26,
        label="Spearman rho",
        zorder=3,
    )
    for yi, value in zip(y, validation["pearson_r"], strict=True):
        ax.text(value + 0.018, yi - 0.11, f"{value:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, validation["dimension"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("Agreement with normalized human ratings")
    ax.xaxis.grid(True, color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURES / "sixdim_agreement")


def plot_sixdim_calibration() -> None:
    indicators = pd.read_csv(OUTPUTS / "urban_perception_sixdim" / "indicators.csv")
    reference = pd.read_csv(ROOT / "data" / "shenzhen_sixdim" / "reference.csv")
    merged = indicators.merge(reference, on="capture_id", validate="one_to_one")
    validation = _sixdim_validation().set_index("target")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for ax, title in zip(axes.flat, SIXDIM_NAMES, strict=True):
        name = title.lower()
        x = merged[name].to_numpy(dtype=float)
        y = merged[f"vlm_{name}"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([0.0, 1.0])
        ax.scatter(x, y, s=9, alpha=0.28, color=BLUE, edgecolors="none")
        ax.plot(line_x, line_x, linestyle="--", color=GREY, linewidth=1.0, label="Identity")
        ax.plot(line_x, slope * line_x + intercept, color=ORANGE, linewidth=1.3, label="Linear fit")
        r = float(validation.loc[f"vlm_{name}", "pearson_r"])
        bias = float((y - x).mean())
        ax.text(
            0.04,
            0.94,
            f"r={r:.2f}\nbias={bias:+.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color=LIGHT_GREY, linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes[-1, :]:
        ax.set_xlabel("Human rating (0-1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("VLM score (0-1)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.14, hspace=0.28, wspace=0.18)
    save_figure(fig, FIGURES / "sixdim_calibration")

    summary_rows = []
    for title in SIXDIM_NAMES:
        name = title.lower()
        predicted = merged[f"vlm_{name}"].astype(float)
        human = merged[name].astype(float)
        summary_rows.append(
            {
                "dimension": name,
                "n": int(len(merged)),
                "predicted_mean": float(predicted.mean()),
                "human_mean": float(human.mean()),
                "predicted_sd": float(predicted.std()),
                "human_sd": float(human.std()),
                "predicted_min": float(predicted.min()),
                "predicted_max": float(predicted.max()),
                "predicted_unique_values": int(predicted.nunique()),
                "mae": float((predicted - human).abs().mean()),
                "mean_bias_predicted_minus_human": float((predicted - human).mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUTS / "urban_perception_sixdim" / "calibration_report.csv", index=False
    )


def plot_cross_task_agreement() -> None:
    prs = pd.read_csv(OUTPUTS / "restorative_quality_prs11" / "validation_report.csv")
    sixdim = _sixdim_validation()
    prs_labels = {
        "vlm_being_away": "Being-away",
        "vlm_coherence": "Coherence",
        "vlm_scope": "Scope",
        "vlm_fascination": "Fascination",
        "vlm_restorative_average": "PRS mean",
    }
    sixdim_labels = {f"vlm_{name.lower()}": name for name in SIXDIM_NAMES}
    rows = []
    for family, table, labels in (
        ("PRS-11 task", prs, prs_labels),
        ("Six-dimension task", sixdim, sixdim_labels),
    ):
        for record in table.to_dict("records"):
            rows.append(
                {
                    "family": family,
                    "dimension": labels[record["target"]],
                    "pearson_r": record["pearson_r"],
                    "ci_low": record["pearson_ci_low"],
                    "ci_high": record["pearson_ci_high"],
                }
            )
    data = pd.DataFrame(rows).sort_values(["family", "pearson_r"], ascending=[True, False])
    data["label"] = data["family"] + " | " + data["dimension"]
    y = np.arange(len(data))
    colors = [GREEN if family == "PRS-11 task" else BLUE for family in data["family"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    lower = data["pearson_r"] - data["ci_low"]
    upper = data["ci_high"] - data["pearson_r"]
    for yi, (_, row), color, low, high in zip(
        y, data.iterrows(), colors, lower, upper, strict=True
    ):
        ax.errorbar(
            row["pearson_r"],
            yi,
            xerr=[[low], [high]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
        )
    ax.set_yticks(y, data["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Pearson correlation with human reference (95% CI)")
    ax.xaxis.grid(True, color=LIGHT_GREY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, FIGURES / "cross_task_agreement")


def main() -> int:
    apply_publication_style()
    plot_sixdim_agreement()
    plot_sixdim_calibration()
    plot_cross_task_agreement()
    print(f"submission figures written to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
