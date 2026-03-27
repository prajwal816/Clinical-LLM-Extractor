"""Validate output JSON files for schema and taxonomy compliance."""

import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_output_file(filepath: Path, valid_pairs: set, valid_statuses: set) -> list[str]:
    """Validate a single output JSON file. Returns list of errors."""
    errors = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    
    # Check top-level structure
    if "patient_id" not in data:
        errors.append("Missing 'patient_id' field")
    if "conditions" not in data:
        errors.append("Missing 'conditions' field")
        return errors
    
    if not isinstance(data["conditions"], list):
        errors.append("'conditions' must be a list")
        return errors
    
    for i, cond in enumerate(data["conditions"]):
        prefix = f"Condition {i} ({cond.get('condition_name', '?')})"
        
        # Required fields
        for field in ["condition_name", "category", "subcategory", "status", "evidence"]:
            if field not in cond:
                errors.append(f"{prefix}: missing '{field}'")
        
        # onset can be null but must be present
        if "onset" not in cond:
            errors.append(f"{prefix}: missing 'onset'")
        
        # Taxonomy validation
        cat = cond.get("category", "")
        sub = cond.get("subcategory", "")
        if (cat, sub) not in valid_pairs:
            errors.append(f"{prefix}: invalid taxonomy pair ({cat}, {sub})")
        
        # Status validation
        status = cond.get("status", "")
        if status not in valid_statuses:
            errors.append(f"{prefix}: invalid status '{status}'")
        
        # Evidence validation
        evidence = cond.get("evidence", [])
        if not isinstance(evidence, list):
            errors.append(f"{prefix}: 'evidence' must be a list")
        elif len(evidence) == 0:
            errors.append(f"{prefix}: evidence list is empty")
        else:
            for j, ev in enumerate(evidence):
                if "note_id" not in ev:
                    errors.append(f"{prefix}: evidence[{j}] missing 'note_id'")
                if "line_no" not in ev:
                    errors.append(f"{prefix}: evidence[{j}] missing 'line_no'")
                elif not isinstance(ev["line_no"], int):
                    errors.append(f"{prefix}: evidence[{j}] 'line_no' must be int")
                if "span" not in ev:
                    errors.append(f"{prefix}: evidence[{j}] missing 'span'")
    
    return errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate extraction output files")
    parser.add_argument("--output-dir", required=True, help="Directory with output JSON files")
    parser.add_argument("--taxonomy", default="taxonomy.json", help="Path to taxonomy.json")
    args = parser.parse_args()
    
    # Load taxonomy
    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    valid_pairs = set()
    for cat_key, cat_val in taxonomy.get("condition_categories", {}).items():
        for sub_key in cat_val.get("subcategories", {}):
            valid_pairs.add((cat_key, sub_key))
    
    valid_statuses = set(taxonomy.get("status_values", {}).keys())
    
    # Validate all files
    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("patient_*.json"))
    
    if not files:
        logger.error(f"No patient files found in {output_dir}")
        sys.exit(1)
    
    total_errors = 0
    total_conditions = 0
    
    for filepath in files:
        errors = validate_output_file(filepath, valid_pairs, valid_statuses)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_conds = len(data.get("conditions", []))
        total_conditions += n_conds
        
        if errors:
            logger.error(f"\n{filepath.name}: {len(errors)} errors, {n_conds} conditions")
            for err in errors:
                logger.error(f"  - {err}")
            total_errors += len(errors)
        else:
            logger.info(f"✓ {filepath.name}: {n_conds} conditions — VALID")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Validated {len(files)} files, {total_conditions} total conditions")
    
    if total_errors > 0:
        logger.error(f"{total_errors} total errors found")
        sys.exit(1)
    else:
        logger.info("All files are valid! ✓")


if __name__ == "__main__":
    main()
