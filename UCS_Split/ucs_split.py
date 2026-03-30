"""
UCS Stratified Dataset Splitter

Splits any CSV dataset into train/val/test sets using
Category+SubCategory stratified sampling.

Input CSV must have at least: filename, Category columns.
SubCategory column is optional but improves stratification quality.

Usage:
    # Single CSV:
    python ucs_split.py --csv data.csv

    # Multiple CSVs (merged then split):
    python ucs_split.py --csv train.csv eval.csv

    # Custom config:
    python ucs_split.py --csv data.csv --config my_config.json

    # Override ratios from command line:
    python ucs_split.py --csv data.csv --train 0.8 --val 0.1 --test 0.1

    # Two-way split (train/test only, no val):
    python ucs_split.py --csv data.csv --train 0.85 --val 0 --test 0.15
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_strat_key(row, cat_col, subcat_col):
    """Build stratification key from Category + SubCategory."""
    cat = row[cat_col]
    sub = row.get(subcat_col) if subcat_col else None
    if pd.notna(sub) and sub != "":
        return f"{cat}||{sub}"
    return f"{cat}||_NONE_"


def stratified_split(df, train_ratio, val_ratio, test_ratio,
                     cat_col, subcat_col, seed, min_group):
    """
    Two-pass stratified split by Category||SubCategory.

    Pass 1: Separate train+val from test.
    Pass 2: Separate train from val.
    Small groups (< min_group samples) are forced into train set.
    """
    df = df.copy()
    df["_strat_key"] = df.apply(
        lambda row: make_strat_key(row, cat_col, subcat_col), axis=1
    )

    key_counts = df["_strat_key"].value_counts()
    small_keys = key_counts[key_counts < min_group].index
    small_df = df[df["_strat_key"].isin(small_keys)]
    splittable_df = df[~df["_strat_key"].isin(small_keys)]

    logging.info(f"  Small groups (< {min_group}): {len(small_keys)} keys, "
                 f"{len(small_df)} samples -> forced to train")

    if len(splittable_df) == 0:
        logging.warning("  All groups are too small for stratification. "
                        "Falling back to random split.")
        splittable_df = df.copy()
        small_df = pd.DataFrame(columns=df.columns)

    # Handle two-way split (no val)
    skip_val = val_ratio == 0

    # Pass 1: train+val / test
    if test_ratio > 0:
        trainval_df, test_df = train_test_split(
            splittable_df, test_size=test_ratio,
            random_state=seed, stratify=splittable_df["_strat_key"],
        )
    else:
        trainval_df = splittable_df
        test_df = pd.DataFrame(columns=df.columns)

    # Pass 2: train / val
    if skip_val:
        train_df = pd.concat([trainval_df, small_df], ignore_index=True)
        val_df = pd.DataFrame(columns=df.columns)
    else:
        tv_counts = trainval_df["_strat_key"].value_counts()
        small_tv_keys = tv_counts[tv_counts < 2].index
        small_tv_df = trainval_df[trainval_df["_strat_key"].isin(small_tv_keys)]
        splittable_tv = trainval_df[~trainval_df["_strat_key"].isin(small_tv_keys)]

        val_adj = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(
            splittable_tv, test_size=val_adj,
            random_state=seed, stratify=splittable_tv["_strat_key"],
        )
        train_df = pd.concat([train_df, small_df, small_tv_df], ignore_index=True)

    # Drop internal column
    for split_df in [train_df, val_df, test_df]:
        if "_strat_key" in split_df.columns:
            split_df.drop(columns=["_strat_key"], inplace=True)

    return train_df, val_df, test_df


def validate_split(train_df, val_df, test_df, cat_col, subcat_col):
    """Print validation checks for the split quality."""
    train_cats = set(train_df[cat_col].unique())

    for name, split_df in [("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            continue

        split_cats = set(split_df[cat_col].unique())
        split_only = split_cats - train_cats
        if split_only:
            logging.warning(f"  {name}-only categories (not in train): {split_only}")
        else:
            logging.info(f"  OK: No {name}-only categories")

        if subcat_col and subcat_col in train_df.columns:
            train_subs = set(train_df[subcat_col].dropna().unique()) - {""}
            split_subs = set(split_df[subcat_col].dropna().unique()) - {""}
            split_only_subs = split_subs - train_subs
            if split_only_subs:
                logging.warning(f"  {name}-only subcategories: {split_only_subs}")
            else:
                logging.info(f"  OK: No {name}-only subcategories")

    # Correlation between train and test category distributions
    if len(test_df) > 0:
        common = sorted(train_cats & set(test_df[cat_col].unique()))
        if len(common) >= 3:
            td = train_df[cat_col].value_counts(normalize=True)
            ed = test_df[cat_col].value_counts(normalize=True)
            corr, _ = pearsonr(
                [td.get(c, 0) for c in common],
                [ed.get(c, 0) for c in common],
            )
            logging.info(f"  Category distribution correlation (train<->test): {corr:.6f}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="UCS Stratified Dataset Splitter"
    )
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Input CSV file(s). Multiple files are merged before splitting.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Config JSON path (default: config.json)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--train", type=float, default=None,
                        help="Train ratio (overrides config)")
    parser.add_argument("--val", type=float, default=None,
                        help="Val ratio (overrides config, use 0 for no val set)")
    parser.add_argument("--test", type=float, default=None,
                        help="Test ratio (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    col_names = config["column_names"]
    cat_col = col_names["category"]
    subcat_col = col_names.get("subcategory")
    fname_col = col_names["filename"]

    # Resolve parameters (CLI overrides config)
    train_ratio = args.train if args.train is not None else config["split_ratios"]["train"]
    val_ratio = args.val if args.val is not None else config["split_ratios"]["val"]
    test_ratio = args.test if args.test is not None else config["split_ratios"]["test"]
    seed = args.seed if args.seed is not None else config["random_seed"]
    min_group = config.get("min_group_for_stratify", 5)

    output_dir = Path(args.output_dir if args.output_dir else config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        logging.error(f"Split ratios must sum to 1.0 (got {total:.2f})")
        return 1

    # Step 1: Load & merge CSVs
    logging.info("=" * 60)
    logging.info("Step 1: Load CSV(s)")
    logging.info("=" * 60)
    dfs = []
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        logging.info(f"  {csv_path}: {len(df):,} rows")

        # Validate required columns
        missing = []
        if fname_col not in df.columns:
            missing.append(fname_col)
        if cat_col not in df.columns:
            missing.append(cat_col)
        if missing:
            logging.error(f"  Missing required columns: {missing}")
            logging.error(f"  Available columns: {list(df.columns)}")
            return 1

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logging.info(f"  Total: {len(combined):,} samples")
    logging.info(f"  Categories: {combined[cat_col].nunique()}")
    if subcat_col and subcat_col in combined.columns:
        n_sub = combined[subcat_col].dropna().replace("", pd.NA).dropna().nunique()
        logging.info(f"  SubCategories: {n_sub}")

    # Step 2: Stratified split
    logging.info("")
    logging.info("=" * 60)
    logging.info("Step 2: Stratified Split")
    logging.info(f"  Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logging.info(f"  Seed: {seed}")
    logging.info("=" * 60)

    train_df, val_df, test_df = stratified_split(
        combined, train_ratio, val_ratio, test_ratio,
        cat_col, subcat_col, seed, min_group,
    )

    logging.info(f"  Train: {len(train_df):,} ({len(train_df)/len(combined)*100:.1f}%)")
    if len(val_df) > 0:
        logging.info(f"  Val:   {len(val_df):,} ({len(val_df)/len(combined)*100:.1f}%)")
    logging.info(f"  Test:  {len(test_df):,} ({len(test_df)/len(combined)*100:.1f}%)")

    # Step 3: Validation
    logging.info("")
    logging.info("=" * 60)
    logging.info("Step 3: Validate Split")
    logging.info("=" * 60)
    validate_split(train_df, val_df, test_df, cat_col, subcat_col)

    # Step 4: Save
    logging.info("")
    logging.info("=" * 60)
    logging.info("Step 4: Save")
    logging.info("=" * 60)

    # Derive output filenames from first input CSV
    base_name = Path(args.csv[0]).stem
    if len(args.csv) > 1:
        base_name = "merged"

    splits_to_save = [("train", train_df)]
    if len(val_df) > 0:
        splits_to_save.append(("val", val_df))
    if len(test_df) > 0:
        splits_to_save.append(("test", test_df))

    for split_name, split_df in splits_to_save:
        out_path = output_dir / f"{base_name}_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        logging.info(f"  Saved: {out_path} ({len(split_df):,} rows)")

    # Save summary
    summary = {
        "source_csvs": args.csv,
        "total_samples": len(combined),
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "actual_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "random_seed": seed,
        "stratification": f"{cat_col}||{subcat_col}" if subcat_col else cat_col,
        "num_categories": combined[cat_col].nunique(),
    }
    summary_path = output_dir / f"{base_name}_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"  Summary: {summary_path}")

    logging.info("")
    logging.info("DONE!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
