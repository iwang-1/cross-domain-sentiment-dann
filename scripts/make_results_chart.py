"""Render the README results chart (light + dark) from the report's published numbers.

Numbers are the canonical results from report/final_report.pdf, Tables 1 and 2:
held-out-domain accuracy under the leave-one-domain-out (LODO) protocol, plus
Financial PhraseBank as an out-of-domain test set.

Usage: python scripts/make_results_chart.py  (writes docs/img/results_{light,dark}.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ["Yelp", "Amazon", "Twitter", "Reddit", "PhraseBank (OOD)"]

# report/final_report.pdf Table 1 (Raw DistilBERT, fine-tuned LODO model) and Table 2 (DANN)
RAW_DISTILBERT = [0.5000, 0.5105, 0.5015, 0.5128, 0.3221]
FINETUNED_LODO = [0.5200, 0.5070, 0.5020, 0.5040, 0.4644]
DANN = [0.7420, 0.7110, 0.6520, 0.6040, 0.6280]
# Off-the-shelf SST-2 DistilBERT reference (report Table 2) — a general sentiment
# model that still beats DANN on the review domains; shown as a per-domain marker
# so the chart tells the same honest story as the results table.
SST2_REFERENCE = [0.9370, 0.8910, 0.6490, 0.6400, 0.7050]

SERIES = [
    ("Raw DistilBERT (no fine-tuning)", RAW_DISTILBERT),
    ("Fine-tuned LODO baseline", FINETUNED_LODO),
    ("DANN (this project)", DANN),
]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "colors": ["#2a78d6", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "colors": ["#3987e5", "#199e70", "#c98500"],
    },
}


def render(mode: str, out: Path) -> None:
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    x = np.arange(len(DOMAINS))
    width = 0.26
    for i, (label, values) in enumerate(SERIES):
        bars = ax.bar(x + (i - 1) * (width + 0.02), values, width, label=label, color=t["colors"][i], zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=t["ink"])

    # Off-the-shelf SST-2 reference: a dashed marker spanning each domain group so
    # the caveat in the results table is visible in the chart itself.
    half = 1.5 * (width + 0.02)
    for xi, ref in zip(x, SST2_REFERENCE):
        ax.plot(
            [xi - half, xi + half], [ref, ref],
            color=t["muted"], linewidth=1.4, linestyle=(0, (3, 2)), zorder=4,
            label="Off-the-shelf SST-2 reference" if xi == 0 else None,
        )

    ax.axhline(0.5, color=t["muted"], linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax.text(len(DOMAINS) - 0.52, 0.512, "coin flip", fontsize=8, color=t["muted"], ha="right", va="bottom")

    ax.set_xticks(x, DOMAINS, fontsize=10, color=t["ink"])
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(colors=t["muted"], length=0)
    ax.set_ylabel("Held-out domain accuracy", fontsize=9, color=t["muted"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])

    ax.set_title(
        "Domain-adversarial training (DANN) lifts held-out-domain accuracy from ~random to 60–74%",
        fontsize=11, color=t["ink"], loc="left", pad=14,
    )
    leg = ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncols=1)
    for text in leg.get_texts():
        text.set_color(t["ink"])

    fig.tight_layout()
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    img_dir = Path(__file__).resolve().parent.parent / "docs" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        render(mode, img_dir / f"results_{mode}.png")
        print(f"wrote docs/img/results_{mode}.png")
