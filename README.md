# UCS SFX Tools

A modular framework for converting sound effects dataset annotations to the [Universal Category System (UCS)](https://universalcategorysystem.com) and producing stratified dataset splits.

## Modules

### UCS_Convert

Rule-based tag-to-UCS conversion pipeline with a four-stage cascade:
1. **Pre-defined Mapping** — curated tag-to-UCS lookup
2. **SubCategory Match** — direct match against UCS subcategory names
3. **Category Match** — direct match against UCS category names
4. **Synonym Match** — reverse lookup against 9,972 UCS synonyms

Per-file conflict resolution applies specificity filtering, majority vote, and positional priority.

```bash
cd UCS_Convert
# Edit config.json: replace <YOUR_PATH> placeholders with your local paths
python UCS_Convert.py
python UCS_Convert.py --dry-run      # preview without writing
python UCS_Convert.py --validate     # validate existing outputs
```

### UCS_Split

**`ucs_split.py`** — Stratified train/val/test splitting (stratification key: `Category||SubCategory`).

```bash
cd UCS_Split
python ucs_split.py --csv classified.csv
python ucs_split.py --csv data1.csv data2.csv   # merge then split
python ucs_split.py --csv data.csv --train 0.8 --val 0.1 --test 0.1
```

**`combine_and_split.py`** — Config-driven multi-source dataset builder (combines, filters, splits).

```bash
python combine_and_split.py --config config_envsound.json
python combine_and_split.py --config config_envsound.json --dry-run
```

## Setup

```bash
pip install -r requirements.txt
```

## UCS Version

All tools use **UCS v8.2.1** (included as `UCS_v8.2.1_Full_List.csv`).

## License

CC-BY-SA 4.0
