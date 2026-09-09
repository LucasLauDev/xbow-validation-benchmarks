"""
Comprehensive Exploratory Data Analysis (EDA) for xbow-val-benchmarks.
Balanced visual suite combining clean horizontal stacked/ranked bars, donut charts,
co-occurrence heatmaps, and boxplots with single-hue sequential gradient frequency bars.
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np

# ── Global Plot Style Configuration (Professional Academic Theme) ────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1e293b",
    "axes.labelsize": 10.5,
    "axes.labelweight": "semibold",
    "axes.titlecolor": "#0f172a",
    "axes.titlesize": 12.0,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#f1f5f9",
    "grid.linewidth": 0.75,
    "xtick.color": "#64748b",
    "xtick.labelsize": 9.5,
    "ytick.color": "#1e293b",
    "ytick.labelsize": 9.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "DejaVu Sans", "Arial"],
    "font.size": 9.5,
    "figure.dpi": 200,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#e2e8f0",
    "legend.fontsize": 9.0,
})

BASE_DIR  = Path(__file__).parent
BENCH_DIR = BASE_DIR / "benchmarks"
OUT_DIR   = BASE_DIR / "eda_figures"
OUT_DIR.mkdir(exist_ok=True)

# Cohesive Sequential Palette for Difficulty Levels
LEVEL_COLORS = {
    1: "#93c5fd",  # Soft Slate Blue (Easy)
    2: "#3b82f6",  # Medium Blue (Medium)
    3: "#1e3a8a",  # Deep Navy (Hard)
}
LEVEL_LABELS = {
    1: "Level 1 (Easy)",
    2: "Level 2 (Medium)",
    3: "Level 3 (Hard)"
}


def get_gradient_colors(counts, min_factor=0.35, max_factor=0.92):
    """Generate monochromatic shades: lighter = fewer, darker = more."""
    cmap = plt.get_cmap("Blues")
    min_c = min(counts) if counts else 0
    max_c = max(counts) if counts else 1
    if max_c == min_c:
        norm_vals = [0.7] * len(counts)
    else:
        norm_vals = [min_factor + (c - min_c) / (max_c - min_c) * (max_factor - min_factor) for c in counts]
    return [mcolors.to_hex(cmap(v)) for v in norm_vals]


def load_data():
    """Load benchmark.json from all challenge subdirectories."""
    records = []
    for bp in sorted(BENCH_DIR.iterdir()):
        bj = bp / "benchmark.json"
        if not bj.exists():
            continue
        try:
            with open(bj, encoding="utf-8") as f:
                d = json.load(f)
            records.append({
                "id": bp.name,
                "name": d.get("name", bp.name),
                "level": int(d.get("level", 1)),
                "tags": d.get("tags", []),
                "num_tags": len(d.get("tags", [])),
                "desc_length": len(d.get("description", "").split()),
                "win_condition": d.get("win_condition", "flag"),
            })
        except Exception as e:
            print(f"Warning: failed reading {bj}: {e}")
    return records


def save_figure(fig, name):
    """Save figure with clean padding and crisp resolution."""
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"  [+] Saved figure -> {p.name}")


# ── Figure 1: Donut Chart - Difficulty Level Proportion ──────────────────────
def fig01_level_donut(records):
    """Figure 1: Donut ring chart of benchmark difficulty level distribution."""
    levels = [1, 2, 3]
    total = len(records)
    counts = [sum(1 for r in records if r["level"] == l) for l in levels]
    colors = get_gradient_colors(counts, min_factor=0.40, max_factor=0.92)
    labels = [f"{LEVEL_LABELS[l]}\n({counts[i]}, {counts[i]/total*100:.1f}%)" for i, l in enumerate(levels)]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    wedges, texts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        startangle=120,
        pctdistance=0.75,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2.0),
        textprops=dict(color="#1e293b", fontsize=9.5, fontweight="medium")
    )

    ax.text(0, 0.06, f"{total}", ha="center", va="center", fontsize=20, fontweight="bold", color="#0f172a")
    ax.text(0, -0.12, "Benchmarks", ha="center", va="center", fontsize=9.5, color="#64748b", fontweight="medium")

    ax.set_title("XBOW: Difficulty Level Distribution (N=104)", pad=12)
    save_figure(fig, "01_difficulty_level_donut")


# ── Figure 2: Horizontal Stacked Bar - Top Tags by Difficulty Level ──────────
def fig02_top_tags_by_level_hbar(records):
    """Figure 2: Top 12 vulnerability tags stacked horizontally by difficulty level."""
    levels = [1, 2, 3]
    all_tags = [t for r in records for t in r["tags"]]
    top_tags = [t for t, _ in Counter(all_tags).most_common(12)]

    tag_level_counts = {
        l: [sum(1 for r in records if r["level"] == l and t in r["tags"]) for t in top_tags]
        for l in levels
    }
    totals = [sum(tag_level_counts[l][ti] for l in levels) for ti in range(len(top_tags))]

    order = sorted(range(len(top_tags)), key=lambda i: totals[i])
    display_labels = [top_tags[i].replace("_", " ").title() for i in order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y_pos = np.arange(len(display_labels))
    bar_height = 0.55
    left = np.zeros(len(display_labels))

    for l in levels:
        vals = np.array([tag_level_counts[l][i] for i in order])
        ax.barh(y_pos, vals, left=left, height=bar_height, color=LEVEL_COLORS[l], label=LEVEL_LABELS[l], edgecolor="white", linewidth=0.8)
        for yi, (v, lp) in enumerate(zip(vals, left)):
            if v >= 2:
                text_color = "white" if l in [2, 3] else "#0f172a"
                ax.text(lp + v / 2, yi, str(v), ha="center", va="center", color=text_color, fontweight="semibold", fontsize=8.5)
        left += vals

    for yi, tot in enumerate([totals[i] for i in order]):
        ax.text(tot + 0.3, yi, f"Total: {tot}", va="center", ha="left", color="#334155", fontweight="medium", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9.5)
    ax.set_xlim(0, max(totals) * 1.25)
    ax.set_xlabel("Number of Benchmarks", labelpad=7)
    ax.set_title("XBOW: Top 12 Vulnerability Tags Segmented by Difficulty Level", pad=10)
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.grid(axis="y", visible=False)

    save_figure(fig, "02_top_tags_by_level")


# ── Figure 3: Horizontal Gradient Bar - Complete Vulnerability Taxonomy ──────
def fig03_all_vulnerability_tags_hbar(records):
    """Figure 3: Complete ranking of all 26 vulnerability tags with lighter-to-darker gradient."""
    all_tags = [t for r in records for t in r["tags"]]
    tag_counts = Counter(all_tags)
    
    sorted_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t])
    counts = [tag_counts[t] for t in sorted_tags]
    labels = [t.replace("_", " ").title() for t in sorted_tags]

    # Monochromatic lighter-to-darker gradient
    colors = get_gradient_colors(counts)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=colors, height=0.6, edgecolor="none")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f" {count}",
            va="center", ha="left",
            fontsize=8.5, fontweight="semibold", color="#1e293b"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_xlim(0, max(counts) * 1.2)
    ax.set_xlabel("Occurrences Across Benchmark Suite", labelpad=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title("XBOW: Complete Vulnerability Taxonomy Frequency (26 Unique Tags)", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.grid(axis="y", visible=False)

    save_figure(fig, "03_all_vulnerability_tags_frequency")


# ── Figure 4: Co-occurrence Heatmap - Multi-Vulnerability Chaining ────────────
def fig04_tag_cooccurrence_heatmap(records):
    """Figure 4: Co-occurrence matrix showing pairs of vulnerabilities chained together."""
    all_tags = [t for r in records for t in r["tags"]]
    top_tags = [t for t, _ in Counter(all_tags).most_common(8)]
    labels = [t.replace("_", " ").title() for t in top_tags]

    matrix = np.zeros((len(top_tags), len(top_tags)), dtype=int)
    for r in records:
        ch_tags = set(r["tags"])
        for i, t1 in enumerate(top_tags):
            for j, t2 in enumerate(top_tags):
                if i != j and t1 in ch_tags and t2 in ch_tags:
                    matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=matrix.max() + 2)

    for i in range(len(top_tags)):
        for j in range(len(top_tags)):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > matrix.max() * 0.55 else "#0f172a"
                ax.text(j, i, f"{val}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9.5)
            else:
                ax.text(j, i, "-", ha="center", va="center", color="#cbd5e1", fontsize=9.0)

    ax.set_xticks(range(len(top_tags)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.0)
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(labels, fontsize=9.0)

    ax.set_title("XBOW: Vulnerability Co-occurrence Matrix (Chained Exploits)", pad=12)
    ax.grid(visible=False)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8.5, color="#64748b")
    cbar.outline.set_edgecolor("#cbd5e1")
    cbar.outline.set_linewidth(0.8)

    save_figure(fig, "04_tag_cooccurrence_heatmap")


# ── Figure 5: Boxplot & Strip Plot - Description Word Count by Level ─────────
def fig05_description_boxplot(records):
    """Figure 5: Boxplot with jittered overlay showing description word count by level."""
    levels = [1, 2, 3]
    labels = [LEVEL_LABELS[l] for l in levels]
    data = [[r["desc_length"] for r in records if r["level"] == l] for l in levels]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.45,
        medianprops=dict(color="#0f172a", linewidth=1.8),
        boxprops=dict(facecolor="#e0f2fe", edgecolor="#3b82f6", linewidth=1.2),
        whiskerprops=dict(color="#64748b", linewidth=1.0),
        capprops=dict(color="#64748b", linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor="#2563eb", markersize=4, alpha=0.6)
    )

    for i, points in enumerate(data):
        y = points
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, "o", color="#1e40af", alpha=0.55, markersize=5)

    ax.set_ylabel("Challenge Description Word Count", labelpad=8)
    ax.set_title("XBOW: Description Prompt Length Distribution by Difficulty Level (Boxplot)", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)

    save_figure(fig, "05_description_length_boxplot")


def main():
    print("=" * 60)
    print("Running Comprehensive EDA for XBOW Validation Benchmarks")
    print("=" * 60)
    records = load_data()
    print(f"Loaded {len(records)} benchmark challenges from {BENCH_DIR.name}/\n")
    
    fig01_level_donut(records)
    fig02_top_tags_by_level_hbar(records)
    fig03_all_vulnerability_tags_hbar(records)
    fig04_tag_cooccurrence_heatmap(records)
    fig05_description_boxplot(records)
    print(f"\nAll figures successfully generated in: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
