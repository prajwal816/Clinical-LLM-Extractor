"""Output formatter — schema validation and JSON writing."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Valid date patterns
DATE_FULL = re.compile(r'^\d{1,2}\s+\w+\s+\d{4}$')          # "16 March 2026"
DATE_MONTH_YEAR = re.compile(r'^\w+\s+\d{4}$')               # "March 2014"
DATE_YEAR = re.compile(r'^\d{4}$')                            # "2014"


def normalize_date(onset: str | None) -> str | None:
    """Normalize onset date to expected format, or return null."""
    if onset is None or onset == "" or onset == "null" or onset == "unknown":
        return None
    
    onset = onset.strip()
    
    # Already valid formats
    if DATE_FULL.match(onset) or DATE_MONTH_YEAR.match(onset) or DATE_YEAR.match(onset):
        return onset
    
    # Try to convert common patterns
    # MM/YYYY -> Month YYYY
    m = re.match(r'^(\d{1,2})/(\d{4})$', onset)
    if m:
        month_num = int(m.group(1))
        year = m.group(2)
        month_names = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        if 1 <= month_num <= 12:
            return f"{month_names[month_num]} {year}"
    
    # MM/DD/YYYY -> DD Month YYYY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', onset)
    if m:
        month_num = int(m.group(1))
        day = int(m.group(2))
        year = m.group(3)
        month_names = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        if 1 <= month_num <= 12:
            return f"{day} {month_names[month_num]} {year}"
    
    # DD.MM.YYYY -> DD Month YYYY
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', onset)
    if m:
        day = int(m.group(1))
        month_num = int(m.group(2))
        year = m.group(3)
        month_names = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        if 1 <= month_num <= 12:
            return f"{day} {month_names[month_num]} {year}"
    
    # YYYY-MM-DD -> DD Month YYYY
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', onset)
    if m:
        year = m.group(1)
        month_num = int(m.group(2))
        day = int(m.group(3))
        month_names = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        if 1 <= month_num <= 12:
            return f"{day} {month_names[month_num]} {year}"
    
    # YYYY-MM -> Month YYYY
    m = re.match(r'^(\d{4})-(\d{1,2})$', onset)
    if m:
        year = m.group(1)
        month_num = int(m.group(2))
        month_names = ["", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        if 1 <= month_num <= 12:
            return f"{month_names[month_num]} {year}"
    
    # "Month DD, YYYY" -> "DD Month YYYY"
    m = re.match(r'^(\w+)\s+(\d{1,2}),?\s+(\d{4})$', onset)
    if m:
        month = m.group(1)
        day = int(m.group(2))
        year = m.group(3)
        return f"{day} {month} {year}"
    
    # If it looks reasonable, return as-is
    logger.warning(f"Could not normalize date: '{onset}', returning as-is")
    return onset


def validate_condition(condition: dict) -> bool:
    """Check that a condition has all required fields."""
    required = ["condition_name", "category", "subcategory", "status", "evidence"]
    for field in required:
        if field not in condition:
            logger.warning(f"Missing field '{field}' in condition: {condition.get('condition_name', '?')}")
            return False
    
    # Validate evidence entries
    if not isinstance(condition.get("evidence"), list):
        return False
    
    for ev in condition["evidence"]:
        if not all(k in ev for k in ("note_id", "line_no", "span")):
            logger.warning(f"Invalid evidence entry in '{condition['condition_name']}': {ev}")
            return False
        # Ensure line_no is an integer
        if not isinstance(ev["line_no"], int):
            try:
                ev["line_no"] = int(ev["line_no"])
            except (ValueError, TypeError):
                return False
    
    return True


def format_patient_output(raw_result: dict, patient_id: str) -> dict:
    """
    Clean and format the raw extraction result into final schema.
    """
    conditions = raw_result.get("conditions", [])
    
    formatted_conditions = []
    for cond in conditions:
        if not validate_condition(cond):
            logger.warning(f"Skipping invalid condition: {cond.get('condition_name', '?')}")
            continue
        
        # Normalize onset date
        cond["onset"] = normalize_date(cond.get("onset"))
        
        # Ensure evidence line_no are integers
        for ev in cond.get("evidence", []):
            if not isinstance(ev["line_no"], int):
                ev["line_no"] = int(ev["line_no"])
        
        # Clean up condition_name
        name = cond.get("condition_name", "").strip()
        if name:
            cond["condition_name"] = name
            formatted_conditions.append(cond)
    
    return {
        "patient_id": patient_id,
        "conditions": formatted_conditions,
    }


def write_patient_output(output: dict, output_dir: str | Path, patient_id: str):
    """Write the patient output JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / f"{patient_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote {filepath} ({len(output.get('conditions', []))} conditions)")
    return filepath
