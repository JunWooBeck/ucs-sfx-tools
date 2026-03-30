"""UCS Dataset Builder — Combine, Filter, and Split.

A reusable, config-driven tool for building custom UCS datasets from
multiple classified sources. Supports:
  - Combining multiple UCS-classified CSVs
  - Category-level filtering (include or exclude mode)
  - Subcategory-level exceptions within excluded categories
  - Stratified splitting via UCS_Split (subprocess)
  - Automatic build report generation

Usage:
    python combine_and_split.py --config config_envsound.json
    python combine_and_split.py --config config_envsound.json --dry-run
    python combine_and_split.py --config config_animals.json

Config examples:
    Exclude mode: {"mode": "exclude", "exclude_categories": ["MUSICAL"]}
    Include mode: {"mode": "include", "include_categories": ["ANIMALS", "BIRDS"]}
    No filter:    {} or {"mode": "none"}
"""

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
UCS_SPLIT_PATH = SCRIPT_DIR / "ucs_split.py"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def resolve_path(base_dir, path_str):
    """Resolve a potentially relative path against base_dir."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_sources(config, base_dir):
    """Load all source CSVs and add source_dataset column."""
    cat_col = config["column_names"]["category"]
    fname_col = config["column_names"]["filename"]

    all_dfs = []
    for source in config["sources"]:
        csv_path = resolve_path(base_dir, source["csv_path"])
        if not csv_path.exists():
            logging.error(f"Source CSV not found: {csv_path}")
            return None

        df = pd.read_csv(csv_path)

        if fname_col not in df.columns or cat_col not in df.columns:
            logging.error(f"Missing required columns in {csv_path}")
            return None

        df["source_dataset"] = source["name"]
        logging.info(f"  Loaded {source['name']}: {len(df):,} rows")
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def apply_filter(df, filter_config, cat_col, subcat_col):
    """Apply category/subcategory filtering based on config."""
    if not filter_config or filter_config.get("mode") == "none":
        logging.info("  Filter: none (keeping all rows)")
        return df

    mode = filter_config.get("mode", "exclude")
    original_count = len(df)

    if mode == "include":
        include_cats = set(filter_config.get("include_categories", []))
        if not include_cats:
            logging.info("  Filter: include mode but no categories specified, keeping all")
            return df
        mask = df[cat_col].isin(include_cats)
        filtered = df[mask].copy()
        logging.info(f"  Filter (include): kept {len(include_cats)} categories")

    elif mode == "exclude":
        exclude_cats = set(filter_config.get("exclude_categories", []))
        except_rules = filter_config.get("exclude_categories_except", {})

        # Start with all rows
        mask = ~df[cat_col].isin(exclude_cats)

        # Handle categories that are excluded but have subcategory exceptions
        for cat_name, rule in except_rules.items():
            keep_subcats = set(rule.get("keep_subcategories", []))
            if keep_subcats and subcat_col and subcat_col in df.columns:
                exception_mask = (
                    (df[cat_col] == cat_name) &
                    (df[subcat_col].isin(keep_subcats))
                )
                mask = mask | exception_mask
                n_kept = exception_mask.sum()
                logging.info(f"  Exception: kept {n_kept} rows from {cat_name} "
                             f"(subcategories: {keep_subcats})")

            # Also mark this category for full exclusion (minus exceptions)
            if cat_name not in exclude_cats:
                full_exclude_mask = (
                    (df[cat_col] == cat_name) &
                    (~df[subcat_col].isin(keep_subcats) if subcat_col and subcat_col in df.columns else True)
                )
                mask = mask & ~full_exclude_mask

        filtered = df[mask].copy()
        excluded = list(exclude_cats) + [
            f"{cat} (except {rule.get('keep_subcategories', [])})"
            for cat, rule in except_rules.items()
            if cat not in exclude_cats
        ]
        logging.info(f"  Filter (exclude): {excluded}")

    else:
        logging.warning(f"  Unknown filter mode '{mode}', keeping all rows")
        return df

    removed = original_count - len(filtered)
    logging.info(f"  Removed: {removed:,} rows ({removed/original_count*100:.1f}%)")
    logging.info(f"  Remaining: {len(filtered):,} rows")
    return filtered


def check_filename_collisions(df, fname_col):
    """Check for cross-dataset filename collisions."""
    dup_mask = df[fname_col].duplicated(keep=False)
    if not dup_mask.any():
        logging.info("  Filename collision check: PASS (no duplicates)")
        return True

    dup_df = df[dup_mask]
    cross_dataset = dup_df.groupby(fname_col)["source_dataset"].nunique()
    collisions = cross_dataset[cross_dataset > 1]

    if len(collisions) > 0:
        logging.error(f"  FATAL: {len(collisions)} cross-dataset filename collisions!")
        for fname in list(collisions.index)[:10]:
            sources = dup_df[dup_df[fname_col] == fname]["source_dataset"].tolist()
            logging.error(f"    {fname}: {sources}")
        return False

    # Within-dataset duplicates (warn but continue)
    n_within = dup_mask.sum()
    logging.warning(f"  Within-dataset duplicate filenames: {n_within} "
                    f"(removing duplicates, keeping first)")
    return True


def run_split(combined_csv_path, output_dir, ratios, seed):
    """Call UCS_Split/ucs_split.py via subprocess."""
    if not UCS_SPLIT_PATH.exists():
        logging.error(f"UCS_Split not found: {UCS_SPLIT_PATH}")
        return False

    cmd = [
        sys.executable,
        str(UCS_SPLIT_PATH),
        "--csv", str(combined_csv_path),
        "--output_dir", str(output_dir),
        "--train", str(ratios["train"]),
        "--val", str(ratios["val"]),
        "--test", str(ratios["test"]),
        "--seed", str(seed),
    ]

    logging.info(f"  Running: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logging.info(f"  [UCS_Split] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            logging.warning(f"  [UCS_Split] {line}")

    if result.returncode != 0:
        logging.error(f"  UCS_Split failed with return code {result.returncode}")
        return False

    return True


def generate_report(config, combined_df, base_dir, filter_config,
                    cat_col, subcat_col, source_stats, output_dir, report_dir):
    """Generate a build report markdown file."""
    dataset_name = config["dataset_name"]
    report_path = resolve_path(base_dir, report_dir) / f"{dataset_name}_build_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {dataset_name} Dataset Build Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Description**: {config.get('description', 'N/A')}",
        "",
        "---",
        "",
        "## Sources",
        "",
        "| Source | Total | After Filter |",
        "|:-------|------:|-------------:|",
    ]

    for name, stats in source_stats.items():
        lines.append(f"| {name} | {stats['total']:,} | {stats['filtered']:,} |")

    total_before = sum(s["total"] for s in source_stats.values())
    total_after = sum(s["filtered"] for s in source_stats.values())
    lines.append(f"| **Total** | **{total_before:,}** | **{total_after:,}** |")
    lines.append("")

    # Filter rules
    lines.append("## Filter Rules")
    lines.append("")
    mode = filter_config.get("mode", "none") if filter_config else "none"
    if mode == "exclude":
        exc = filter_config.get("exclude_categories", [])
        lines.append(f"- **Mode**: exclude")
        lines.append(f"- **Excluded categories**: {', '.join(exc)}")
        for cat, rule in filter_config.get("exclude_categories_except", {}).items():
            keep = rule.get("keep_subcategories", [])
            lines.append(f"- **Exception**: {cat} — kept subcategories: {', '.join(keep)}")
    elif mode == "include":
        inc = filter_config.get("include_categories", [])
        lines.append(f"- **Mode**: include")
        lines.append(f"- **Included categories**: {', '.join(inc)}")
    else:
        lines.append("- **Mode**: none (no filtering)")
    lines.append("")

    # Category distribution
    cat_counts = combined_df[cat_col].value_counts()
    lines.append(f"## Category Distribution ({len(cat_counts)} categories)")
    lines.append("")
    lines.append("| Category | Count | Ratio |")
    lines.append("|:---------|------:|------:|")
    for cat, count in cat_counts.items():
        pct = count / len(combined_df) * 100
        lines.append(f"| {cat} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # Split info
    split_cfg = config.get("split", {})
    if split_cfg.get("enabled"):
        ratios = split_cfg["ratios"]
        out = resolve_path(base_dir, output_dir)
        base_stem = f"{dataset_name}_combined"

        lines.append("## Split Results")
        lines.append("")
        lines.append(f"- **Ratios**: train={ratios['train']}, val={ratios['val']}, test={ratios['test']}")
        lines.append(f"- **Seed**: {split_cfg.get('seed', 42)}")
        lines.append("")

        for split_name in ["train", "val", "test"]:
            split_path = out / f"{base_stem}_{split_name}.csv"
            if split_path.exists():
                n = len(pd.read_csv(split_path))
                pct = n / len(combined_df) * 100
                lines.append(f"- **{split_name}**: {n:,} ({pct:.1f}%)")

    lines.append("")

    report = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"  Report saved: {report_path}")
    return report_path


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="UCS Dataset Builder — Combine, Filter, and Split")
    parser.add_argument("--config", required=True,
                        help="Config JSON file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    cat_col = config["column_names"]["category"]
    subcat_col = config["column_names"].get("subcategory")
    fname_col = config["column_names"]["filename"]
    filter_config = config.get("filter", {})
    output_dir = config["output"]["output_dir"]
    report_dir = config["output"]["report_dir"]

    logging.info("=" * 60)
    logging.info(f"  UCS Dataset Builder: {dataset_name}")
    logging.info(f"  Config: {config_path.name}")
    logging.info("=" * 60)

    # Step 1: Load sources
    logging.info("")
    logging.info("Step 1: Load source CSVs")
    logging.info("-" * 40)
    combined = load_sources(config, base_dir)
    if combined is None:
        return 1
    logging.info(f"  Combined: {len(combined):,} rows, "
                 f"{combined[cat_col].nunique()} categories")

    # Track per-source stats before filtering
    source_totals = combined.groupby("source_dataset").size().to_dict()

    # Step 2: Filter
    logging.info("")
    logging.info("Step 2: Apply filter")
    logging.info("-" * 40)
    filtered = apply_filter(combined, filter_config, cat_col, subcat_col)

    # Track per-source stats after filtering
    source_filtered = filtered.groupby("source_dataset").size().to_dict()
    source_stats = {}
    for name in source_totals:
        source_stats[name] = {
            "total": source_totals[name],
            "filtered": source_filtered.get(name, 0),
        }

    # Step 3: Check filename collisions
    logging.info("")
    logging.info("Step 3: Validate filenames")
    logging.info("-" * 40)
    if not check_filename_collisions(filtered, fname_col):
        return 1

    # Remove within-dataset duplicates if any
    before_dedup = len(filtered)
    filtered = filtered.drop_duplicates(subset=[fname_col], keep="first")
    if len(filtered) < before_dedup:
        logging.info(f"  Removed {before_dedup - len(filtered)} within-dataset duplicates")

    # Step 4: Save combined CSV
    logging.info("")
    logging.info("Step 4: Save combined CSV")
    logging.info("-" * 40)
    out_dir = resolve_path(base_dir, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / f"{dataset_name}_combined.csv"

    # Save without source_dataset for UCS_Split compatibility
    output_cols = [fname_col, cat_col]
    if subcat_col and subcat_col in filtered.columns:
        output_cols.append(subcat_col)
    kw_col = config["column_names"].get("keywords")
    if kw_col and kw_col in filtered.columns:
        output_cols.append(kw_col)
    output_cols.append("source_dataset")

    if args.dry_run:
        logging.info(f"  [DRY-RUN] Would save {len(filtered):,} rows to {combined_csv}")
    else:
        filtered[output_cols].to_csv(combined_csv, index=False)
        logging.info(f"  Saved: {combined_csv} ({len(filtered):,} rows)")

    # Step 5: Split
    split_config = config.get("split", {})
    if split_config.get("enabled") and not args.dry_run:
        logging.info("")
        logging.info("Step 5: Stratified split")
        logging.info("-" * 40)

        # Save a version without source_dataset for UCS_Split
        split_input = out_dir / f"{dataset_name}_combined.csv"
        split_cols = [c for c in output_cols if c != "source_dataset"]
        # Overwrite combined CSV with split-compatible version (no source_dataset)
        # then restore after split
        split_input_tmp = out_dir / f"_tmp_split_input.csv"
        filtered[split_cols].to_csv(split_input_tmp, index=False)

        ratios = split_config["ratios"]
        seed = split_config.get("seed", 42)
        success = run_split(split_input_tmp, out_dir, ratios, seed)

        # Clean up temp file and rename outputs
        split_input_tmp.unlink(missing_ok=True)
        # Rename _tmp_split_input_*.csv to {dataset_name}_combined_*.csv
        for split_name in ["train", "val", "test"]:
            tmp_path = out_dir / f"_tmp_split_input_{split_name}.csv"
            final_path = out_dir / f"{dataset_name}_combined_{split_name}.csv"
            if tmp_path.exists():
                tmp_path.rename(final_path)
        tmp_summary = out_dir / "_tmp_split_input_split_summary.json"
        final_summary = out_dir / f"{dataset_name}_combined_split_summary.json"
        if tmp_summary.exists():
            tmp_summary.rename(final_summary)

        if not success:
            logging.error("  Split failed!")
            return 1
    elif args.dry_run:
        logging.info("")
        logging.info("Step 5: [DRY-RUN] Would run stratified split")

    # Step 6: Generate report
    logging.info("")
    logging.info("Step 6: Generate report")
    logging.info("-" * 40)
    if not args.dry_run:
        generate_report(
            config, filtered, base_dir, filter_config,
            cat_col, subcat_col, source_stats, output_dir, report_dir,
        )

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"  {dataset_name} dataset build complete!")
    logging.info(f"  Total samples: {len(filtered):,}")
    logging.info(f"  Categories: {filtered[cat_col].nunique()}")
    if subcat_col and subcat_col in filtered.columns:
        n_sub = filtered[subcat_col].dropna().replace("", pd.NA).dropna().nunique()
        logging.info(f"  SubCategories: {n_sub}")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
