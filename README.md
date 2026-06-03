# UCS-SFX-Tools

Tag-to-UCS conversion framework for sound effects datasets, with stratified splitting and multi-source merging.

## Repository Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── UCS_Convert/
│   ├── UCS_Convert.py          # Tag-to-UCS conversion pipeline
│   ├── config.json             # Dataset paths + 411 manual mappings
│   └── UCS_v8.2.1_Full_List.csv  # UCS taxonomy structure
└── UCS_Split/
    ├── ucs_split.py            # Stratified train/val/test splitter
    ├── combine_and_split.py    # Multi-source dataset builder
    ├── config.json             # Split configuration (ratios, seed)
    └── config_envsound.json    # EnvSound-UCS build configuration
```

## Modules

### UCS_Convert

Rule-based tag-to-UCS conversion pipeline with a four-stage cascade:
1. **Pre-defined Mapping** -- curated tag-to-UCS lookup (411 entries)
2. **SubCategory Match** -- direct match against UCS subcategory names
3. **Category Match** -- direct match against UCS category names
4. **Synonym Match** -- reverse lookup against 9,972 UCS synonyms

Per-file conflict resolution: specificity filtering, majority vote, positional priority.

```bash
cd UCS_Convert/
# Edit config.json: replace <YOUR_PATH> placeholders with local paths
python UCS_Convert.py
python UCS_Convert.py --dry-run      # preview without writing
python UCS_Convert.py --validate     # validate existing outputs
```

**Input**: CSV with `fname` and `labels` (comma-separated tags) columns.

**Output** (per dataset):
- `*_UCS_classified.csv` -- mapped files (`filename, Category, SubCategory, Keywords`)
- `*_UCS_unclassified.csv` -- unmapped files
- `*_unclassified_summary.csv` -- tag frequency of unmapped tags
- `*_ambiguity_review.csv` -- files with competing category assignments

### UCS_Split

Stratified train/val/test splitting using `Category||SubCategory` composite key.

```bash
cd UCS_Split/
python ucs_split.py --csv classified.csv
python ucs_split.py --csv data1.csv data2.csv   # merge then split
python ucs_split.py --csv data.csv --train 0.8 --val 0.1 --test 0.1
```

### combine_and_split.py

Config-driven multi-source dataset builder with category filtering.

```bash
cd UCS_Split/
python combine_and_split.py --config config_envsound.json
python combine_and_split.py --config config_envsound.json --dry-run
```

> **Note on split reproducibility:** Pre-computed splits in each dataset repository are the canonical references. Running `ucs_split.py` with the same seed may produce different splits depending on `scikit-learn` version.

## Setup

```bash
pip install -r requirements.txt
```

## UCS Version

All tools use **UCS v8.2.1** (included as `UCS_Convert/UCS_v8.2.1_Full_List.csv`).

## License

CC-BY-SA 4.0
