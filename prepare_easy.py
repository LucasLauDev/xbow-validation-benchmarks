"""
XBOW Validation Benchmarks — Data Cleaning / Selection
Copies benchmark folders whose tags match the selected vulnerability
categories into a `selected-benchmarks/benchmarks/` directory.

Selected categories → canonical tag(s):
  Cross-Site Scripting (XSS)              → xss
  Privilege Escalation                    → privilege_escalation
  Server-Side Template Injection (SSTI)   → ssti
  SQL Injection (SQLi)                    → sqli, blind_sqli
  Path Traversal (Directory Traversal)    → path_traversal
  Server-Side Request Forgery (SSRF)      → ssrf
  XML External Entity Injection (XXE)     → xxe
  IDOR (Insecure Direct Object Reference) → idor
  Remote Code Execution (RCE)             → command_injection
"""

import json
import shutil
from pathlib import Path

# ── Category → tag mapping ────────────────────────────────────────────────────

SELECTED_CATEGORIES = {
    "Cross-Site Scripting (XSS)":               ["xss"],
    "Privilege Escalation":                     ["privilege_escalation"],
    "Server-Side Template Injection (SSTI)":    ["ssti"],
    "SQL Injection (SQLi)":                     ["sqli", "blind_sqli"],
    "Path Traversal (Directory Traversal)":     ["path_traversal"],
    "Server-Side Request Forgery (SSRF)":       ["ssrf"],
    "XML External Entity Injection (XXE)":      ["xxe"],
    "IDOR (Insecure Direct Object Reference)":  ["idor"],
    "Remote Code Execution (RCE)":              ["command_injection"],
}

SELECTED_TAGS: set[str] = {tag for tags in SELECTED_CATEGORIES.values() for tag in tags}

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "benchmarks"
DST_DIR = ROOT / "selected-easy-benchmarks" 

# ── Selection logic ───────────────────────────────────────────────────────────

def get_tags(benchmark_dir: Path) -> list[str]:
    jf = benchmark_dir / "benchmark.json"
    if not jf.exists():
        return []
    meta = json.loads(jf.read_text(encoding="utf-8"))
    return meta.get("tags", [])


def matches(tags: list[str]) -> bool:
    return bool(set(tags) & SELECTED_TAGS)


def primary_category(tags: list[str]) -> str:
    """Return the first matching category name for reporting."""
    for cat, cat_tags in SELECTED_CATEGORIES.items():
        if set(tags) & set(cat_tags):
            return cat
    return "Unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if DST_DIR.exists():
        print(f"Removing existing {DST_DIR} …")
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True)

    candidates = sorted(
        [d for d in SRC_DIR.iterdir() if d.is_dir() and d.name.startswith("XBEN-")]
    )
    print(f"Scanning {len(candidates)} benchmarks …\n")

    selected, skipped = [], []
    cat_counts: dict[str, int] = {c: 0 for c in SELECTED_CATEGORIES}

    # Find the single easiest benchmark for each category
    bm_info = []
    for bm_dir in candidates:
        tags = get_tags(bm_dir)
        level = 999
        jf = bm_dir / "benchmark.json"
        if jf.exists():
            try:
                meta = json.loads(jf.read_text(encoding="utf-8"))
                level = int(meta.get("level", 1))
            except Exception:
                pass
        bm_info.append((bm_dir, tags, level))

    selected_names = set()
    for cat, cat_tags in SELECTED_CATEGORIES.items():
        matching_bms = []
        for bm_dir, tags, level in bm_info:
            if set(tags) & set(cat_tags):
                matching_bms.append((level, bm_dir.name, bm_dir))
        if matching_bms:
            matching_bms.sort(key=lambda x: (x[0], x[1]))
            selected_names.add(matching_bms[0][1])

    for bm_dir in candidates:
        if bm_dir.name in selected_names:
            tags = get_tags(bm_dir)
            dst = DST_DIR / bm_dir.name
            shutil.copytree(bm_dir, dst)
            selected.append(bm_dir.name)
            for cat, cat_tags in SELECTED_CATEGORIES.items():
                if set(tags) & set(cat_tags):
                    cat_counts[cat] += 1
        else:
            skipped.append(bm_dir.name)

    # for bm_dir in candidates:
    #     tags = get_tags(bm_dir)
    #     if not matches(tags):
    #         skipped.append(bm_dir.name)
    #         continue
    # 
    #     # Copy the entire benchmark folder
    #     dst = DST_DIR / bm_dir.name
    #     shutil.copytree(bm_dir, dst)
    #     selected.append(bm_dir.name)
    # 
    #     # Tally per category (a benchmark may match multiple)
    #     for cat, cat_tags in SELECTED_CATEGORIES.items():
    #         if set(tags) & set(cat_tags):
    #             cat_counts[cat] += 1

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Selected : {len(selected):>3}  benchmarks")
    print(f"  Skipped  : {len(skipped):>3}  benchmarks")
    print(f"{'─'*60}")
    print("\nBreakdown by category:")
    for cat, count in cat_counts.items():
        bar = "█" * count
        print(f"  {count:>2}  {bar:<25}  {cat}")

    print(f"\nSelected benchmark IDs:")
    for name in selected:
        tags = get_tags(DST_DIR / name)
        print(f"  {name}  →  {', '.join(tags)}")

    print(f"\nOutput directory: {DST_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
