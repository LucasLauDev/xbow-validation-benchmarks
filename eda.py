"""
XBOW Validation Benchmarks — Exploratory Data Analysis
Reads benchmark.json + docker-compose.yml from every XBEN-*-24 folder
and produces a set of graphs saved to eda_figures/.
"""

import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8f9fb",
        "axes.edgecolor": "#d0d4de",
        "axes.labelcolor": "#2c3040",
        "axes.titlecolor": "#000000",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.grid": True,
        "grid.color": "#e2e5ed",
        "grid.linewidth": 0.6,
        "xtick.color": "#5a6075",
        "ytick.color": "#5a6075",
        "text.color": "#2c3040",
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d0d4de",
        "legend.labelcolor": "#2c3040",
        "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.size": 10,
        "savefig.facecolor": "#ffffff",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    }
)

ACCENT = ["#2563eb", "#ea580c", "#dc2626", "#16a34a", "#7c3aed", "#ca8a04", "#0891b2"]
LEVEL_COLORS = {"1": "#16a34a", "2": "#ea580c", "3": "#dc2626"}
LEVEL_LABELS = {"1": "Easy (L1)", "2": "Medium (L2)", "3": "Hard (L3)"}

OUTPUT_DIR = Path("eda_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_benchmarks(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted((root / "benchmarks").iterdir()):
        if not d.is_dir() or not d.name.startswith("XBEN-"):
            continue
        jf = d / "benchmark.json"
        cf = d / "docker-compose.yml"
        if not jf.exists():
            continue
        meta = json.loads(jf.read_text(encoding="utf-8"))
        tags = list(dict.fromkeys(meta.get("tags", [])))  # deduplicate, preserve order
        level = str(meta.get("level", "?"))
        # Docker service count
        n_services = 0
        if cf.exists():
            content = cf.read_text(encoding="utf-8")
            services = re.findall(r"^  (\w[\w-]+):\s*$", content, re.MULTILINE)
            n_services = len(services)
        rows.append(
            {
                "id": d.name,
                "name": meta.get("name", ""),
                "level": level,
                "tags": tags,
                "n_tags": len(tags),
                "n_services": n_services,
                "win_condition": meta.get("win_condition", ""),
            }
        )
    return pd.DataFrame(rows)


ROOT = Path(__file__).parent
df = load_benchmarks(ROOT)
print(f"Loaded {len(df)} benchmarks")

# ── Derived structures ────────────────────────────────────────────────────────

# Explode tags → one row per (benchmark, tag)
tag_rows = df.explode("tags").dropna(subset=["tags"])
tag_rows = tag_rows[tag_rows["tags"].str.strip() != ""]

# Global tag counts
tag_counts = tag_rows["tags"].value_counts()

# Per-level tag counts
level_tag = (
    tag_rows.groupby(["tags", "level"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["1", "2", "3"], fill_value=0)
)

# Sort by total
level_tag["total"] = level_tag.sum(axis=1)
level_tag = level_tag.sort_values("total", ascending=False)

# Service count distribution
service_dist = df["n_services"].value_counts().sort_index()

# Level distribution
level_dist = df["level"].value_counts().sort_index()

# Tags per benchmark distribution
tags_per_bm = df["n_tags"].value_counts().sort_index()

# Co-occurrence matrix (top 15 tags only)
TOP_N = 15
top_tags = list(tag_counts.head(TOP_N).index)

comat = pd.DataFrame(0, index=top_tags, columns=top_tags)
for tags in df["tags"]:
    tags_filtered = [t for t in tags if t in top_tags]
    for a, b in combinations(set(tags_filtered), 2):
        comat.loc[a, b] += 1
        comat.loc[b, a] += 1


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overview dashboard (2×2)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("XBOW Validation Benchmarks — Overview", fontsize=16, fontweight="semibold",
             color="#000000", y=0.98)

# 1a — Difficulty donut
ax = axes[0, 0]
counts = [level_dist.get(k, 0) for k in ["1", "2", "3"]]
labels = [f"{LEVEL_LABELS[k]}\n({counts[i]})" for i, k in enumerate(["1", "2", "3"])]
colors = [LEVEL_COLORS[k] for k in ["1", "2", "3"]]
wedges, texts, autotexts = ax.pie(
    counts,
    labels=labels,
    colors=colors,
    autopct="%1.0f%%",
    pctdistance=0.72,
    startangle=90,
    wedgeprops={"width": 0.52, "edgecolor": "#ffffff", "linewidth": 2},
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_color("#ffffff")
    t.set_fontweight("bold")
ax.set_title("Difficulty distribution")
ax.set_facecolor("#ffffff")

# 1b — Docker service count
ax = axes[0, 1]
ax.bar(
    service_dist.index.astype(str),
    service_dist.values,
    color=ACCENT[0],
    width=0.55,
    edgecolor="#0f1117",
    linewidth=1.2,
)
for i, (x, v) in enumerate(zip(service_dist.index, service_dist.values)):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=11, color="#1a1d27",
            fontweight="semibold")
ax.set_xlabel("Services in docker-compose.yml")
ax.set_ylabel("Benchmarks")
ax.set_title("Docker services per benchmark")
ax.set_xticks(range(len(service_dist)))
ax.set_xticklabels([f"{x} svc" for x in service_dist.index])
ax.set_ylim(0, service_dist.max() + 8)

# 1c — Tags per benchmark distribution
ax = axes[1, 0]
ax.bar(
    tags_per_bm.index.astype(str),
    tags_per_bm.values,
    color=ACCENT[4],
    width=0.55,
    edgecolor="#0f1117",
    linewidth=1.2,
)
for i, (x, v) in enumerate(zip(tags_per_bm.index, tags_per_bm.values)):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=11, color="#1a1d27",
            fontweight="semibold")
ax.set_xlabel("Tags assigned per benchmark")
ax.set_ylabel("Benchmarks")
ax.set_title("Tags per benchmark")
ax.set_xticks(range(len(tags_per_bm)))
ax.set_xticklabels([f"{x} tag{'s' if x != 1 else ''}" for x in tags_per_bm.index])
ax.set_ylim(0, tags_per_bm.max() + 8)

# 1d — Unique tags summary table
ax = axes[1, 1]
ax.axis("off")
summary_data = [
    ["Total benchmarks",    str(len(df))],
    ["Unique tags",         str(tag_counts.shape[0])],
    ["Easy (Level 1)",      str(level_dist.get("1", 0))],
    ["Medium (Level 2)",    str(level_dist.get("2", 0))],
    ["Hard (Level 3)",      str(level_dist.get("3", 0))],
    ["Single-container",    str((df["n_services"] == 1).sum())],
    ["Multi-container",     str((df["n_services"] > 1).sum())],
    ["Avg tags / benchmark",f"{df['n_tags'].mean():.2f}"],
    ["Max services",        str(df["n_services"].max())],
]
tbl = ax.table(
    cellText=summary_data,
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#f8f9fb" if r % 2 == 0 else "#ffffff")
    cell.set_edgecolor("#d0d4de")
    cell.set_text_props(color="#2c3040")
    if r == 0:
        cell.set_facecolor("#e8edf7")
        cell.set_text_props(color="#1a1d27", fontweight="bold")
    if c == 1:
        cell.set_text_props(ha="right", color="#2563eb", fontweight="semibold")
ax.set_title("Dataset summary", pad=16)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTPUT_DIR / "01_overview.png")
plt.close(fig)
print("Saved 01_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Tag frequency (all tags, horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle("Vulnerability Tag Frequency — All 104 Benchmarks", fontsize=14,
             fontweight="semibold", color="#000000")

sorted_tags = tag_counts.sort_values(ascending=True)
colors_bar = [
    LEVEL_COLORS["3"] if sorted_tags[t] <= 1 else
    ACCENT[0] if sorted_tags[t] >= 10 else
    "#7c3aed"
    for t in sorted_tags.index
]

bars = ax.barh(sorted_tags.index, sorted_tags.values, color=colors_bar,
               edgecolor="#0f1117", linewidth=0.8, height=0.7)
for bar, val in zip(bars, sorted_tags.values):
    ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9, color="#2c3040")

ax.set_xlabel("Number of benchmarks")
ax.set_xlim(0, sorted_tags.max() + 3)
ax.set_title("")

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=ACCENT[0],       label="High frequency (≥10)"),
    Patch(facecolor=ACCENT[4],       label="Mid frequency (2–9)"),
    Patch(facecolor=LEVEL_COLORS["3"], label="Rare / unique (1)"),
]
ax.legend(handles=legend_elems, loc="lower right", framealpha=0.7)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_tag_frequency.png")
plt.close(fig)
print("Saved 02_tag_frequency.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Top 12 tags stacked by difficulty
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("Top 12 Tags by Difficulty Level", fontsize=14, fontweight="semibold",
             color="#000000")

top12 = level_tag.head(12).copy()
y = np.arange(len(top12))
bar_h = 0.6

left = np.zeros(len(top12))
for lvl in ["1", "2", "3"]:
    vals = top12[lvl].values
    ax.barh(y, vals, left=left, height=bar_h, label=LEVEL_LABELS[lvl],
            color=LEVEL_COLORS[lvl], edgecolor="#0f1117", linewidth=0.8)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v >= 1:
            ax.text(l + v / 2, i, str(v), ha="center", va="center",
                    fontsize=8.5, color="#ffffff", fontweight="bold")
    left = left + vals

ax.set_yticks(y)
ax.set_yticklabels(top12.index, fontsize=10)
ax.set_xlabel("Benchmarks")
ax.legend(loc="lower right")
ax.set_xlim(0, top12["total"].max() + 2)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_tags_by_level.png")
plt.close(fig)
print("Saved 03_tags_by_level.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Tag co-occurrence heatmap (top 15 tags)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle(f"Tag Co-occurrence Heatmap (Top {TOP_N} Tags)", fontsize=14,
             fontweight="semibold", color="#000000")

mask = comat == 0
cmap = sns.color_palette("Blues_d", as_cmap=True)
sns.heatmap(
    comat,
    ax=ax,
    cmap="YlOrBr",
    mask=mask,
    linewidths=0.5,
    linecolor="#0f1117",
    annot=True,
    fmt="d",
            annot_kws={"size": 9, "color": "#1a1d27"},
    cbar_kws={"label": "Co-occurrences", "shrink": 0.7},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_tag_cooccurrence.png")
plt.close(fig)
print("Saved 04_tag_cooccurrence.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Tags per benchmark by difficulty (box + strip)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Tags per Benchmark by Difficulty", fontsize=14, fontweight="semibold",
             color="#000000")

level_order = ["1", "2", "3"]
df["level_label"] = df["level"].map(LEVEL_LABELS)
level_label_order = [LEVEL_LABELS[k] for k in level_order]

for i, lvl in enumerate(level_order):
    subset = df[df["level"] == lvl]["n_tags"]
    color = LEVEL_COLORS[lvl]
    # Box
    bp = ax.boxplot(
        subset.values,
        positions=[i],
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor=color + "44", edgecolor=color, linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        medianprops=dict(color="#ffffff", linewidth=2),
        flierprops=dict(marker="o", color=color, markersize=5, alpha=0.7),
    )
    # Strip
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subset))
    ax.scatter(np.full(len(subset), i) + jitter, subset.values,
               color=color, alpha=0.55, s=28, zorder=3, edgecolors="none")

ax.set_xticks(range(len(level_order)))
ax.set_xticklabels([LEVEL_LABELS[k] for k in level_order])
ax.set_ylabel("Number of tags")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "05_tags_per_bm_by_level.png")
plt.close(fig)
print("Saved 05_tags_per_bm_by_level.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Benchmark ID timeline coloured by level
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 4))
fig.suptitle("Benchmark IDs — Difficulty Across the Suite", fontsize=14,
             fontweight="semibold", color="#000000")

nums = df["id"].str.extract(r"XBEN-(\d+)-24").astype(int)[0]
df["num"] = nums

for lvl in ["1", "2", "3"]:
    sub = df[df["level"] == lvl]
    ax.scatter(sub["num"], np.ones(len(sub)), c=LEVEL_COLORS[lvl],
               s=90, zorder=3, label=LEVEL_LABELS[lvl], alpha=0.9, edgecolors="#0f1117",
               linewidths=0.6)

ax.set_yticks([])
ax.set_xlabel("Benchmark number (001–104)")
ax.set_xlim(0, 106)
ax.legend(loc="upper right")
ax.set_ylim(0.5, 1.5)
ax.set_title("")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "06_id_timeline.png")
plt.close(fig)
print("Saved 06_id_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Docker service breakdown by tag category
# ══════════════════════════════════════════════════════════════════════════════

# Average service count per primary tag (first tag in list)
df["primary_tag"] = df["tags"].apply(lambda t: t[0] if t else None)
svc_by_tag = (
    df.dropna(subset=["primary_tag"])
    .groupby("primary_tag")["n_services"]
    .agg(["mean", "count"])
    .query("count >= 2")
    .sort_values("mean", ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Avg Docker Services by Primary Tag (min 2 benchmarks)", fontsize=13,
             fontweight="semibold", color="#000000")

bars = ax.barh(
    svc_by_tag.index,
    svc_by_tag["mean"],
    color=ACCENT[1],
    edgecolor="#0f1117",
    linewidth=0.8,
    height=0.65,
)
for bar, (tag, row) in zip(bars, svc_by_tag.iterrows()):
    ax.text(row["mean"] + 0.03, bar.get_y() + bar.get_height() / 2,
            f'{row["mean"]:.2f}  (n={int(row["count"])})',
            va="center", fontsize=8.5, color="#2c3040")

ax.axvline(1.0, color="#2563eb", linewidth=1.2, linestyle="--", alpha=0.6,
           label="Single-container baseline")
ax.set_xlabel("Average service count")
ax.set_xlim(0, svc_by_tag["mean"].max() + 0.6)
ax.legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "07_services_by_tag.png")
plt.close(fig)
print("Saved 07_services_by_tag.png")


# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nAll figures saved to ./{OUTPUT_DIR}/")
print("Files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
