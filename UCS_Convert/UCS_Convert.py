import pandas as pd
import json
import os
import sys
import collections
import logging
from functools import lru_cache

def setup_logging(level_name):
    """Set logging level."""
    level = logging.getLevelName(level_name.upper())
    if not isinstance(level, int):
        level = logging.INFO
        logging.warning(f"Wrong log_level '{level_name}'. Set default to INFO.")

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ucs_convert.log', encoding='utf-8')
        ]
    )
    logging.debug("Logging set to DEBUG.")


def load_config(config_path='config.json'):
    """Load configuration from a JSON file."""
    logging.info(f"Loading configuration from '{config_path}'...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        logging.error(f"Error: Cannot find configuration file '{config_path}'.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error: '{config_path}' is not a valid JSON file: {e}")
        return None


@lru_cache(maxsize=8)
def create_file_map(directory, audio_extension):
    """Create file mapping index with caching."""
    logging.info(f"Creating index from '{directory}' (extension: {audio_extension})...")
    file_map = {}
    audio_extension_lower = audio_extension.lower()

    if not os.path.isdir(directory):
        logging.warning(f"Error: Audio directory '{directory}' not found, skipping.")
        return file_map

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(audio_extension_lower):
                # File ID (exclude extension) for looking up in CSV
                file_id = os.path.splitext(file)[0]
                # Store filename
                file_map[file_id] = file

    logging.info(f"{len(file_map)} audio files mapped.")
    return file_map


def load_ucs_lookups(config):
    """Create UCS lookup tables for category mapping."""
    ucs_structure_path = config['ucs_paths']['structure_csv']

    # 1. Load UCS structure
    try:
        # Skip UCS CSV top 2 lines
        df_ucs = pd.read_csv(ucs_structure_path, skiprows=2)
        logging.info(f"Loaded UCS structure from '{ucs_structure_path}'")
    except FileNotFoundError:
        logging.error(f"Error: Cannot find '{ucs_structure_path}'.")
        return None, None, None, None
    except Exception as e:
        logging.error(f"Error: Loading '{ucs_structure_path}': {e}")
        return None, None, None, None

    # Get column names from config
    cat_col = config['column_names']['ucs_category_column']
    subcat_col = config['column_names']['ucs_subcategory_column']

    if cat_col not in df_ucs.columns or subcat_col not in df_ucs.columns:
        logging.error(f"Error: UCS structure file is missing required columns ('{cat_col}', '{subcat_col}').")
        logging.error(f"Available columns: {list(df_ucs.columns)}")
        return None, None, None, None

    # 1.1: 'SubCategory' -> 'Category' mapping (e.g: "walk" -> "FOOTSTEPS")
    # Skip empty sub-categories using .dropna()
    subcat_to_cat_map = {
        str(row[subcat_col]).lower(): str(row[cat_col]).upper()
        for _, row in df_ucs.dropna(subset=[subcat_col]).iterrows()
        if pd.notna(row[subcat_col])
    }

    # 1.2: Valid 'Category' set (normalized to uppercase)
    valid_categories = {cat.upper() for cat in df_ucs[cat_col].unique() if pd.notna(cat)}

    logging.info(f"Loaded {len(subcat_to_cat_map)} subcategories and {len(valid_categories)} categories from '{ucs_structure_path}'.")

    # 2. Load manual mappings (priority)
    manual_map = {}
    mapping_object = config.get('manual_mapping', {})

    if not mapping_object:
        logging.warning("Warning: 'manual_mapping' is empty or missing in config.json. Only auto-matching will be performed.")
    else:
        # 2.1: 'tag' -> {Category, SubCategory} mapping
        for tag, mapping in mapping_object.items():
            if not isinstance(mapping, dict):
                logging.warning(f"Warning: Manual mapping for '{tag}' is not a valid object. Skipping.")
                continue

            # Category and SubCategory -> UPPERCASE
            category_val = mapping.get('Category')
            subcategory_val = mapping.get('SubCategory')

            if category_val and isinstance(category_val, str):
                category_val = category_val.upper()

            if subcategory_val and isinstance(subcategory_val, str):
                subcategory_val = subcategory_val.upper()

            manual_map[tag.lower()] = {
                'Category': category_val,
                'SubCategory': subcategory_val
            }
        logging.info(f"Loaded {len(manual_map)} manual mappings from config.")

    # 3. Build synonym reverse lookup: synonym -> {Category: [SubCategories]}
    synonym_map = {}
    syn_col = 'Synonyms - Comma Separated'
    if syn_col in df_ucs.columns:
        for _, row in df_ucs.dropna(subset=[syn_col]).iterrows():
            cat_val = str(row[cat_col]).upper().strip()
            sub_val = str(row[subcat_col]).upper().strip() if pd.notna(row[subcat_col]) else None
            if not sub_val:
                continue
            for syn in str(row[syn_col]).split(','):
                syn_key = syn.strip().lower()
                if not syn_key:
                    continue
                if syn_key not in synonym_map:
                    synonym_map[syn_key] = {}
                if cat_val not in synonym_map[syn_key]:
                    synonym_map[syn_key][cat_val] = []
                if sub_val not in synonym_map[syn_key][cat_val]:
                    synonym_map[syn_key][cat_val].append(sub_val)
        logging.info(f"Loaded {len(synonym_map)} synonym entries from UCS structure.")
    else:
        logging.warning("Warning: 'Synonyms - Comma Separated' column not found. Synonym matching disabled.")

    return subcat_to_cat_map, valid_categories, manual_map, synonym_map


def classify_tag(tag, subcat_to_cat_map, valid_categories, manual_map, default_category, synonym_map):
    """
    Classify a tag using manual and automatic mapping.
    Returns UPPERCASE Category and SubCategory.
    """
    if not tag or pd.isna(tag):
        return default_category, None

    tag_lower = str(tag).lower().strip()
    if not tag_lower:
        return default_category, None

    tag_upper = tag_lower.upper()
    tag_normalized = tag_lower.replace('_', ' ')

    # 1. Check Manual Mapping (highest priority)
    if tag_lower in manual_map:
        mapping = manual_map[tag_lower]
        return mapping['Category'], mapping['SubCategory']
    if tag_normalized in manual_map:
        mapping = manual_map[tag_normalized]
        return mapping['Category'], mapping['SubCategory']

    # 2. Auto Mapping - SubCategory first -> derive Category from it
    if tag_lower in subcat_to_cat_map:
        category = subcat_to_cat_map[tag_lower]
        return category, tag_upper

    # 3. Auto Mapping - Category match -> reverse-find SubCategory via synonyms
    if tag_upper in valid_categories:
        if tag_normalized in synonym_map and tag_upper in synonym_map[tag_normalized]:
            best_sub = synonym_map[tag_normalized][tag_upper][0]
            return tag_upper, best_sub
        return tag_upper, None

    # 4. Synonym lookup - normalize tag and search UCS synonyms
    if tag_normalized in synonym_map:
        cat_matches = synonym_map[tag_normalized]
        best_cat = max(cat_matches, key=lambda c: len(cat_matches[c]))
        best_sub = cat_matches[best_cat][0]
        return best_cat, best_sub

    # 5. Mapping failed - use default
    return default_category, None


def process_csv_batch(df_chunk, file_map, config, lookups):
    """
    Process a chunk of DataFrame rows with advanced priority logic:
    1. Collect all candidates
    2. Prefer SubCategories
    3. Add to review list if ambiguous
    4. Frequency
    5. Right side tag wins
    """
    subcat_to_cat_map, valid_categories, manual_map, synonym_map = lookups

    tags_col = config['column_names']['fsd50k_tags_column']
    fname_col = config['column_names']['fsd50k_fname_column']
    default_cat = config['defaults']['default_category']
    tag_separator = config.get('tag_separator', ',')

    classified_rows = []
    unclassified_rows = []
    unclassified_tags = []
    review_rows = []  # List for ambiguous files

    for _, row in df_chunk.iterrows():
        tags_value = row.get(tags_col, '')
        if pd.isna(tags_value):
            original_tags_str = ''
        else:
            original_tags_str = str(tags_value)

        fname_value = row.get(fname_col, '')
        if pd.isna(fname_value) or fname_value == '':
            continue
        fname_numeric = str(fname_value)

        if not fname_numeric:
            continue

        # Retrieve filename
        filename_only = file_map.get(fname_numeric)
        if not filename_only:
            continue

        tags_list = [t.strip() for t in original_tags_str.split(tag_separator) if t.strip()]

        # Collect Candidates
        candidates = []
        for idx, tag in enumerate(tags_list):
            cat, sub = classify_tag(tag, subcat_to_cat_map, valid_categories, manual_map, default_cat, synonym_map)

            if cat != default_cat:
                candidates.append({
                    'tag': tag,
                    'cat': cat,
                    'sub': sub,
                    'idx': idx # Save index for Rule 3
                })

        # Decision
        final_cat = default_cat
        final_sub = None
        is_classified = False
        conflict_detected = False

        if not candidates:
            # Case: No valid mapping found
            unclassified_rows.append({
                config['column_names']['fsd50k_fname_column']: filename_only,
                config['column_names']['ucs_category_column']: default_cat,
                config['column_names']['ucs_subcategory_column']: None,
                config['column_names']['ucs_keywords_column']: original_tags_str
            })
            unclassified_tags.extend(tags_list)
            continue

        # Specificity Filter
        # If any candidate has a SubCategory, keep only those with SubCategories.
        has_subcategory = any(c['sub'] is not None for c in candidates)
        if has_subcategory:
            filtered_candidates = [c for c in candidates if c['sub'] is not None]
        else:
            filtered_candidates = candidates

        # Conflict Detection
        # Check if we have different Category conflict
        unique_categories = set(c['cat'] for c in filtered_candidates)
        if len(unique_categories) > 1:
            conflict_detected = True

        # Majority Vote
        cat_counts = collections.Counter(c['cat'] for c in filtered_candidates)
        max_freq = max(cat_counts.values())
        # Get all categories related
        top_categories = [cat for cat, count in cat_counts.items() if count == max_freq]

        # Filter candidates to only those belonging to the winning category/categories
        best_candidates = [c for c in filtered_candidates if c['cat'] in top_categories]

        # Rightmost tag wins
        best_candidates.sort(key=lambda x: x['idx'], reverse=True)
        winner = best_candidates[0]

        final_cat = winner['cat']
        final_sub = winner['sub']
        is_classified = True

        # Construct Output Rows
        row_data = {
            config['column_names']['fsd50k_fname_column']: filename_only,
            config['column_names']['ucs_category_column']: final_cat,
            config['column_names']['ucs_subcategory_column']: final_sub,
            config['column_names']['ucs_keywords_column']: original_tags_str
        }

        classified_rows.append(row_data)

        # Add to Review List if there was conflict
        if conflict_detected:
            candidate_summary = "; ".join([f"{c['tag']}->{c['cat']}/{c['sub']}" for c in candidates])
            review_row_data = row_data.copy()
            review_row_data['Candidate_Mappings'] = candidate_summary
            review_rows.append(review_row_data)

    return classified_rows, unclassified_rows, unclassified_tags, review_rows


def process_csv(input_csv, classified_output_csv, unclassified_output_csv,
                       unclassified_summary_csv, file_map, config, lookups, is_dry_run=False):
    """Convert original CSV to UCS format and divide into classified/unclassified/review."""

    try:
        chunk_size = 1000
        df_fsd_iter = pd.read_csv(input_csv, chunksize=chunk_size)
    except FileNotFoundError:
        logging.error(f"Error: Cannot find original CSV '{input_csv}'.")
        return
    except Exception as e:
        logging.error(f"Error reading CSV '{input_csv}': {e}")
        return

    logging.info(f"Processing '{input_csv}' in chunks (size: {chunk_size})...")

    all_classified = []
    all_unclassified = []
    all_unclassified_tags = []
    all_review_rows = []
    processed = 0
    last_logged = 0

    for chunk in df_fsd_iter:
        classified, unclassified, unclassified_tags, review_rows = process_csv_batch(
            chunk, file_map, config, lookups
        )
        all_classified.extend(classified)
        all_unclassified.extend(unclassified)
        all_unclassified_tags.extend(unclassified_tags)
        all_review_rows.extend(review_rows)

        processed += len(chunk)
        if processed - last_logged >= 5000:
            logging.info(f"  Processed {processed} samples...")
            last_logged = processed

    # Save results
    internal_cat_col = config['column_names']['ucs_category_column']
    internal_subcat_col = config['column_names']['ucs_subcategory_column']
    internal_keywords_col = config['column_names']['ucs_keywords_column']
    fname_col = config['column_names']['fsd50k_fname_column']

    output_cols_map = config['column_names']['output_columns']

    # 1. Save classified files
    if all_classified:
        df_classified = pd.DataFrame(all_classified)
        df_classified[internal_subcat_col] = df_classified[internal_subcat_col].fillna('')

        df_classified = df_classified.rename(columns={
            fname_col: output_cols_map['filename'],
            internal_cat_col: output_cols_map['category'],
            internal_subcat_col: output_cols_map['subcategory'],
            internal_keywords_col: output_cols_map['tags']
        })

        df_classified = df_classified[list(output_cols_map.values())]

        if is_dry_run:
            logging.info(f"[Dry-Run] Would save {len(df_classified)} classified items to '{classified_output_csv}'.")
        else:
            df_classified.to_csv(classified_output_csv, index=False)
            logging.info(f"Classification successful: Saved {len(df_classified)} items to '{classified_output_csv}'.")
    else:
        logging.info("No files classified successfully.")

    # 2. Save unclassified files
    if all_unclassified:
        df_unclassified = pd.DataFrame(all_unclassified)
        df_unclassified[internal_subcat_col] = df_unclassified[internal_subcat_col].fillna('')

        df_unclassified = df_unclassified.rename(columns={
            fname_col: output_cols_map['filename'],
            internal_cat_col: output_cols_map['category'],
            internal_subcat_col: output_cols_map['subcategory'],
            internal_keywords_col: output_cols_map['tags']
        })

        df_unclassified = df_unclassified[list(output_cols_map.values())]

        if is_dry_run:
            logging.info(f"[Dry-Run] Would save {len(df_unclassified)} unclassified items to '{unclassified_output_csv}'.")
        else:
            df_unclassified.to_csv(unclassified_output_csv, index=False)
            logging.info(f"Classification failed: Saved {len(df_unclassified)} items to '{unclassified_output_csv}'.")
    else:
        logging.info("No files failed classification. (All files classified successfully)")

    # 3. Save unclassified summary
    if all_unclassified_tags:
        tag_counter = collections.Counter(all_unclassified_tags)
        df_summary = pd.DataFrame(tag_counter.items(), columns=['tag', 'count'])
        df_summary = df_summary.sort_values(by='count', ascending=False)

        if is_dry_run:
            logging.info(f"[Dry-Run] Would save {len(df_summary)} tag items to '{unclassified_summary_csv}'.")
        else:
            df_summary.to_csv(unclassified_summary_csv, index=False)
            logging.info(f"Unclassified summary: Saved {len(df_summary)} tag items to '{unclassified_summary_csv}'.")
    else:
        logging.info("Unclassified summary: No tags failed classification.")

    # 4. Save Ambiguity Review list
    if all_review_rows:
        # Generate review filename automatically based on classified output filename
        base, ext = os.path.splitext(classified_output_csv)
        review_csv_path = f"{base}_ambiguity_review{ext}"

        df_review = pd.DataFrame(all_review_rows)
        df_review[internal_subcat_col] = df_review[internal_subcat_col].fillna('')

        # Rename columns to match output format but keep Candidate_Mappings
        rename_map = {
            fname_col: output_cols_map['filename'],
            internal_cat_col: output_cols_map['category'],
            internal_subcat_col: output_cols_map['subcategory'],
            internal_keywords_col: output_cols_map['tags']
        }
        df_review = df_review.rename(columns=rename_map)

        # Order columns: Standard outputs + Candidate Mappings
        cols = list(output_cols_map.values()) + ['Candidate_Mappings']
        existing_cols = [c for c in cols if c in df_review.columns]
        df_review = df_review[existing_cols]

        if is_dry_run:
            logging.info(f"[Dry-Run] Would save {len(df_review)} ambiguous items to '{review_csv_path}'.")
        else:
            df_review.to_csv(review_csv_path, index=False)
            logging.info(f"Ambiguity Review: Saved {len(df_review)} items to '{review_csv_path}'.")
    else:
        logging.info("Ambiguity Review: No conflicting categories found.")


def run_validation(config):
    """
    Validate already generated classified CSV files.
    """
    logging.info("--- Validation mode ---")

    if 'dataset_sets' not in config:
        logging.error("Error: 'dataset_sets' list is not defined in 'config.json'.")
        return

    output_cols_map = config['column_names']['output_columns']
    filename_col = output_cols_map['filename']
    category_col = output_cols_map['category']
    default_cat = config['defaults']['default_category']

    total_errors = 0

    for dataset in config['dataset_sets']:
        set_name = dataset.get('set_name', dataset.get('input_csv'))
        classified_csv = dataset.get('output_classified_csv')
        audio_dir = dataset.get('audio_dir')

        logging.info(f"\n[{set_name}] Validation set start: '{classified_csv}'")

        if not os.path.exists(classified_csv):
            logging.warning(f"Error: Cannot find '{classified_csv}'. Validation skipped.")
            total_errors += 1
            continue

        try:
            df_classified = pd.read_csv(classified_csv)
        except Exception as e:
            logging.error(f"Error: Failed to read '{classified_csv}' file: {e}")
            total_errors += 1
            continue

        set_errors = 0

        # 1. Validate file paths (Reconstruct path just for checking)
        logging.info("  Validating files exist...")
        missing_files = 0
        if filename_col not in df_classified.columns:
            logging.error(f"  Error: '{filename_col}' column not found.")
            set_errors += 1
        else:
            for fname in df_classified[filename_col]:
                # Reconstruct full path to check existence
                full_path = os.path.join(audio_dir, fname)
                if not os.path.isfile(full_path):
                    logging.debug(f"    -> File missing: {full_path}")
                    missing_files += 1
            if missing_files > 0:
                logging.warning(f"  Warning: {missing_files} files could not be found in '{audio_dir}'.")
                set_errors += missing_files
            else:
                logging.info("  File existence validation passed.")

        # 2. Validate Category classification
        logging.info("  Validating Category classification...")
        if category_col not in df_classified.columns:
            logging.error(f"  Error: '{category_col}' column not found.")
            set_errors += 1
        else:
            bad_rows = df_classified[
                df_classified[category_col].isnull() |
                (df_classified[category_col] == default_cat)
            ]
            if not bad_rows.empty:
                logging.warning(f"  Warning: {len(bad_rows)} rows are unclassified or classified as '{default_cat}'.")
                logging.debug(f"  First 5 problematic rows:\n{bad_rows.head()}")
                set_errors += len(bad_rows)
            else:
                logging.info("  Category classification validation passed (all rows are validly classified).")

        if set_errors == 0:
            logging.info(f"[{set_name}] Set validation complete: No errors.")
        else:
            logging.warning(f"[{set_name}] Set validation complete: {set_errors} errors/warnings found.")

        total_errors += set_errors

    if total_errors == 0:
        logging.info("\n--- All sets validation complete: No errors ---")
    else:
        logging.warning(f"\n--- All sets validation complete: Found {total_errors} total errors/warnings ---")


def main():
    # Check command line arguments
    is_dry_run = '--dry-run' in sys.argv
    is_validate = '--validate' in sys.argv

    # Setup logging first
    setup_logging("INFO")

    # Load configuration
    config = load_config()
    if not config:
        logging.error("Failed to load configuration. Exiting.")
        return 1

    # Update logging level from config
    log_level_from_config = config.get("log_level", "INFO").upper()
    level = logging.getLevelName(log_level_from_config)
    if isinstance(level, int):
        logging.getLogger().setLevel(level)
        logging.info(f"Log level set to {log_level_from_config} from config.")
    else:
        logging.warning(f"Invalid log_level '{log_level_from_config}' in config. Using INFO.")

    # Run validation mode if requested
    if is_validate:
        run_validation(config)
        return 0

    if is_dry_run:
        logging.info("--- Dry-Run mode ---")
        logging.info("Will not write anything to the filesystem.")

    # Main conversion logic
    # 1. Load UCS lookup maps once at script start
    lookups = load_ucs_lookups(config)
    if lookups[0] is None:
        logging.error("Error: Failed to load UCS lookup maps. Aborting script.")
        return 1

    # 2. Process all datasets defined in configuration
    if 'dataset_sets' not in config:
        logging.error("Error: 'dataset_sets' list is not defined in 'config.json'.")
        return 1

    successful_sets = 0
    failed_sets = 0

    for dataset in config['dataset_sets']:
        set_name = dataset.get('set_name', dataset.get('input_csv'))
        logging.info(f"\n--- [{set_name}] Processing start ---")

        # Get all required paths
        input_csv = dataset.get('input_csv')
        audio_dir = dataset.get('audio_dir')
        classified_csv = dataset.get('output_classified_csv')
        unclassified_csv = dataset.get('output_unclassified_csv')
        unclassified_summary_csv = dataset.get('output_unclassified_summary')

        # Validate all paths are present
        if not all([input_csv, audio_dir, classified_csv, unclassified_csv, unclassified_summary_csv]):
            logging.warning(f"Warning: Settings for '{set_name}' set are incomplete (all 5 paths required). Skipping this set.")
            failed_sets += 1
            continue

        # 3. Create audio file map for this set (with caching)
        file_map = create_file_map(
            audio_dir,
            config['audio_paths']['audio_extension']
        )

        if not file_map:
            logging.warning(f"Warning: No audio files found in '{audio_dir}'. Skipping this set.")
            failed_sets += 1
            continue

        # 4. Process the CSV for this dataset
        try:
            process_csv(
                input_csv,
                classified_csv,
                unclassified_csv,
                unclassified_summary_csv,
                file_map,
                config,
                lookups,
                is_dry_run
            )
            successful_sets += 1
            logging.info(f"--- [{set_name}] Processing complete ---")
        except Exception as e:
            logging.error(f"Error processing {set_name}: {e}")
            failed_sets += 1

    # Final summary
    logging.info(f"\n=== All tasks completed ===")
    logging.info(f"Successful: {successful_sets} sets")
    if failed_sets > 0:
        logging.warning(f"Failed: {failed_sets} sets")

    return 0 if failed_sets == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
