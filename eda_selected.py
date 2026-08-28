"""
XBOW Validation Benchmarks — EDA on Selected Categories
Reads from `selected-benchmarks/benchmarks/` (produced by prepare_selected.py)
and produces EDA figures saved to `selected_eda_figures/`.

Run prepare_selected.py first.
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
from matplotlib.patches import Patch

# ── Style (matches eda.py) ─────────────────────────────────────────────────────
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

# ── Category → canonical tag(s) ───────────────────────────────────────────────
SELECTED_CATEGORIES = {
    "XSS":               ["xss"],
    "Privilege Escalation": ["privilege_escalation"],
    "SSTI":              ["ssti"],
    "SQLi":              ["sqli", "blind_sqli"],
    "Path Traversal":    ["path_traversal"],
    "SSRF":              ["ssrf"],
    "XXE":               ["xxe"],
    "IDOR":              ["idor"],
    "RCE":               ["command_injection"],
    "Information Disclosure": ["information_disclosure"]
}

TAG_TO_CATEGORY: dict[str, str] = {
    tag: cat for cat, tags in SELECTED_CATEGORIES.items() for tag in tags
}

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "selected-benchmarks"
OUTPUT_DIR = ROOT / "selected_eda_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_benchmarks(src: Path) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(
            f"'{src}' not found.\n"
            "Run  python prepare_selected.py  first."
        )
    rows = []
    for d in sorted(src.iterdir()):
        if not d.is_dir() or not d.name.startswith("XBEN-"):
            continue
        jf = d / "benchmark.json"
        cf = d / "docker-compose.yml"
        if not jf.exists():
            continue
        meta = json.loads(jf.read_text(encoding="utf-8"))
        tags = list(dict.fromkeys(meta.get("tags", [])))
        level = str(meta.get("level", "?"))
        n_services = 0
        if cf.exists():
            content = cf.read_text(encoding="utf-8")
            services = re.findall(r"^  (\w[\w-]+):\s*$", content, re.MULTILINE)
            n_services = len(services)
        # Derive primary category from the first matched selected tag
        primary_cat = next(
            (TAG_TO_CATEGORY[t] for t in tags if t in TAG_TO_CATEGORY), "Other"
        )
        rows.append(
            {
                "id": d.name,
                "name": meta.get("name", ""),
                "level": level,
                "tags": tags,
                "n_tags": len(tags),
                "n_services": n_services,
                "primary_cat": primary_cat,
                "win_condition": meta.get("win_condition", ""),
            }
        )
    return pd.DataFrame(rows)


df = load_benchmarks(SRC_DIR)
print(f"Loaded {len(df)} selected benchmarks")

# ── Derived structures ─────────────────────────────────────────────────────────

tag_rows = df.explode("tags").dropna(subset=["tags"])
tag_rows = tag_rows[tag_rows["tags"].str.strip() != ""]
tag_counts = tag_rows["tags"].value_counts()

# Map each tag row to its category label (selected tags get their category, others → "Other")
tag_rows = tag_rows.copy()
tag_rows["category"] = tag_rows["tags"].map(TAG_TO_CATEGORY).fillna("Other")

# Per-level tag counts (selected tags only)
selected_tag_rows = tag_rows[tag_rows["category"] != "Other"]
level_tag = (
    selected_tag_rows.groupby(["category", "level"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["1", "2", "3"], fill_value=0)
)
level_tag["total"] = level_tag.sum(axis=1)
level_tag = level_tag.sort_values("total", ascending=False)

# Category counts (how many benchmarks per category, counted per benchmark not per tag)
cat_counts: Counter = Counter()
for _, row in df.iterrows():
    matched = {TAG_TO_CATEGORY[t] for t in row["tags"] if t in TAG_TO_CATEGORY}
    for cat in matched:
        cat_counts[cat] += 1

service_dist = df["n_services"].value_counts().sort_index()
level_dist   = df["level"].value_counts().sort_index()
tags_per_bm  = df["n_tags"].value_counts().sort_index()

# Co-occurrence among selected category tags only
sel_tags_all = [t for tags in SELECTED_CATEGORIES.values() for t in tags]
comat = pd.DataFrame(0, index=list(SELECTED_CATEGORIES.keys()),
                     columns=list(SELECTED_CATEGORIES.keys()))
for tags in df["tags"]:
    cats_present = list({TAG_TO_CATEGORY[t] for t in tags if t in TAG_TO_CATEGORY})
    for a, b in combinations(cats_present, 2):
        comat.loc[a, b] += 1
        comat.loc[b, a] += 1


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Overview dashboard (2×2)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Selected Vulnerability Categories — Overview",
    fontsize=16, fontweight="semibold", color="#000000", y=0.98,
)

# 1a — Difficulty donut
ax = axes[0, 0]
counts_lvl = [level_dist.get(k, 0) for k in ["1", "2", "3"]]
labels_lvl = [f"{LEVEL_LABELS[k]}\n({counts_lvl[i]})" for i, k in enumerate(["1", "2", "3"])]
colors_lvl = [LEVEL_COLORS[k] for k in ["1", "2", "3"]]
wedges, texts, autotexts = ax.pie(
    counts_lvl,
    labels=labels_lvl,
    colors=colors_lvl,
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

# 1b — Category benchmark counts (horizontal bar)
ax = axes[0, 1]
sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1])
cat_names = [c for c, _ in sorted_cats]
cat_vals  = [v for _, v in sorted_cats]
bar_colors = [
    ACCENT[i % len(ACCENT)] for i in range(len(cat_names))
]
bars = ax.barh(cat_names, cat_vals, color=bar_colors,
               edgecolor="#0f1117", linewidth=0.8, height=0.65)
for bar, val in zip(bars, cat_vals):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=10, color="#2c3040", fontweight="semibold")
ax.set_xlabel("Benchmarks")
ax.set_xlim(0, max(cat_vals) + 3)
ax.set_title("Benchmarks per category")

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
    ax.text(i, v + 0.4, str(v), ha="center", va="bottom", fontsize=11,
            color="#1a1d27", fontweight="semibold")
ax.set_xlabel("Tags assigned per benchmark")
ax.set_ylabel("Benchmarks")
ax.set_title("Tags per benchmark")
ax.set_xticks(range(len(tags_per_bm)))
ax.set_xticklabels([f"{x} tag{'s' if x != 1 else ''}" for x in tags_per_bm.index])
ax.set_ylim(0, tags_per_bm.max() + 6)

# 1d — Summary table
ax = axes[1, 1]
ax.axis("off")
summary_data = [
    ["Total selected",       str(len(df))],
    ["Unique categories",    str(len(cat_counts))],
    ["Easy (Level 1)",       str(level_dist.get("1", 0))],
    ["Medium (Level 2)",     str(level_dist.get("2", 0))],
    ["Hard (Level 3)",       str(level_dist.get("3", 0))],
    ["Single-container",     str((df["n_services"] == 1).sum())],
    ["Multi-container",      str((df["n_services"] > 1).sum())],
    ["Avg tags / benchmark", f"{df['n_tags'].mean():.2f}"],
    ["Max services",         str(df["n_services"].max())],
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
# Figure 2 — Category frequency bar (horizontal)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    f"Selected Category Frequency — {len(df)} Benchmarks",
    fontsize=14, fontweight="semibold", color="#000000",
)

sorted_full = sorted(cat_counts.items(), key=lambda x: x[1], reverse=False)
names_ = [c for c, _ in sorted_full]
vals_  = [v for _, v in sorted_full]
palette = [ACCENT[i % len(ACCENT)] for i in range(len(names_))]

bars = ax.barh(names_, vals_, color=palette, edgecolor="#0f1117", linewidth=0.8, height=0.6)
for bar, val in zip(bars, vals_):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=11, color="#2c3040", fontweight="semibold")

ax.set_xlabel("Number of benchmarks")
ax.set_xlim(0, max(vals_) + 4)
ax.set_title("")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_category_frequency.png")
plt.close(fig)
print("Saved 02_category_frequency.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Category counts stacked by difficulty level
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("Categories by Difficulty Level", fontsize=14,
             fontweight="semibold", color="#000000")

top_lvl = level_tag.copy()
y = np.arange(len(top_lvl))
bar_h = 0.6

left = np.zeros(len(top_lvl))
for lvl in ["1", "2", "3"]:
    vals = top_lvl[lvl].values
    ax.barh(y, vals, left=left, height=bar_h, label=LEVEL_LABELS[lvl],
            color=LEVEL_COLORS[lvl], edgecolor="#0f1117", linewidth=0.8)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v >= 1:
            ax.text(l + v / 2, i, str(v), ha="center", va="center",
                    fontsize=9, color="#ffffff", fontweight="bold")
    left = left + vals

ax.set_yticks(y)
ax.set_yticklabels(top_lvl.index, fontsize=11)
ax.set_xlabel("Benchmarks")
ax.legend(loc="lower right")
ax.set_xlim(0, top_lvl["total"].max() + 3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_categories_by_level.png")
plt.close(fig)
print("Saved 03_categories_by_level.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Category co-occurrence heatmap
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle("Category Co-occurrence Heatmap (Selected Benchmarks)", fontsize=14,
             fontweight="semibold", color="#000000")

mask = comat == 0
sns.heatmap(
    comat,
    ax=ax,
    cmap="YlOrBr",
    mask=mask,
    linewidths=0.5,
    linecolor="#0f1117",
    annot=True,
    fmt="d",
    annot_kws={"size": 10, "color": "#1a1d27"},
    cbar_kws={"label": "Co-occurrences", "shrink": 0.7},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_category_cooccurrence.png")
plt.close(fig)
print("Saved 04_category_cooccurrence.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Tags per benchmark by difficulty (box + strip)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Tags per Benchmark by Difficulty", fontsize=14,
             fontweight="semibold", color="#000000")

level_order = ["1", "2", "3"]
rng = np.random.default_rng(42)
for i, lvl in enumerate(level_order):
    subset = df[df["level"] == lvl]["n_tags"]
    color = LEVEL_COLORS[lvl]
    ax.boxplot(
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
    jitter = rng.uniform(-0.12, 0.12, len(subset))
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
# Figure 6 — Benchmark ID timeline coloured by category
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Benchmark IDs — Selected Categories Across the Suite", fontsize=14,
             fontweight="semibold", color="#000000")

cat_list = list(SELECTED_CATEGORIES.keys())
cat_palette = {cat: ACCENT[i % len(ACCENT)] for i, cat in enumerate(cat_list)}

nums = df["id"].str.extract(r"XBEN-(\d+)-24").astype(int)[0]
df["num"] = nums

y_pos = {cat: i for i, cat in enumerate(cat_list)}
for _, row in df.iterrows():
    matched_cats = [TAG_TO_CATEGORY[t] for t in row["tags"] if t in TAG_TO_CATEGORY]
    matched_cats = list(dict.fromkeys(matched_cats))  # deduplicate, preserve order
    for cat in matched_cats:
        ax.scatter(
            row["num"], y_pos[cat],
            c=cat_palette[cat], s=80, zorder=3, alpha=0.85,
            edgecolors="#0f1117", linewidths=0.5,
        )

ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(list(y_pos.keys()), fontsize=10)
ax.set_xlabel("Benchmark number")
ax.set_xlim(0, 106)
ax.set_title("")
ax.grid(axis="x", alpha=0.4)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "06_id_timeline_by_category.png")
plt.close(fig)
print("Saved 06_id_timeline_by_category.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Docker service distribution per category
# ══════════════════════════════════════════════════════════════════════════════

svc_by_cat = (
    df.groupby("primary_cat")["n_services"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Avg Docker Services by Vulnerability Category", fontsize=13,
             fontweight="semibold", color="#000000")

bars = ax.barh(
    svc_by_cat.index,
    svc_by_cat["mean"],
    color=ACCENT[1],
    edgecolor="#0f1117",
    linewidth=0.8,
    height=0.6,
)
for bar, (tag, row) in zip(bars, svc_by_cat.iterrows()):
    ax.text(row["mean"] + 0.03, bar.get_y() + bar.get_height() / 2,
            f'{row["mean"]:.2f}  (n={int(row["count"])})',
            va="center", fontsize=9, color="#2c3040")

ax.axvline(1.0, color="#2563eb", linewidth=1.2, linestyle="--", alpha=0.6,
           label="Single-container baseline")
ax.set_xlabel("Average service count")
ax.set_xlim(0, svc_by_cat["mean"].max() + 0.8)
ax.legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "07_services_by_category.png")
plt.close(fig)
print("Saved 07_services_by_category.png")


# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nAll figures saved to ./{OUTPUT_DIR}/")
print("Files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
