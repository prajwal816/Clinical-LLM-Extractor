"""Clinical LLM Extractor — Main pipeline entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

import config
from document_loader import load_patient_notes
from extractor import extract_conditions_for_patient
from llm_client import LLMClient
from output_formatter import format_patient_output, write_patient_output
from taxonomy_mapper import validate_and_fix_conditions


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_train_dir(data_dir: Path) -> Path | None:
    """
    Attempt to locate the train directory for few-shot examples.
    Looks for a sibling 'train' directory or a 'train' subdirectory.
    """
    # If data_dir itself is the train dir
    if (data_dir / "labels").exists():
        return data_dir
    
    # Sibling directory
    sibling = data_dir.parent / "train"
    if sibling.exists() and (sibling / "labels").exists():
        return sibling
    
    # Parent's train directory
    parent_train = data_dir.parent.parent / "train"
    if parent_train.exists() and (parent_train / "labels").exists():
        return parent_train
    
    return None


def process_patient(
    llm: LLMClient,
    data_dir: Path,
    patient_id: str,
    taxonomy: dict,
    train_dir: Path | None,
    output_dir: Path,
) -> dict | None:
    """Process a single patient through the full pipeline."""
    logger = logging.getLogger(__name__)
    
    # Step 1: Load notes
    notes = load_patient_notes(data_dir, patient_id)
    if not notes:
        logger.error(f"No notes found for {patient_id}")
        return None
    
    # Step 2: Extract conditions via LLM
    raw_result = extract_conditions_for_patient(
        llm=llm,
        notes=notes,
        patient_id=patient_id,
        taxonomy=taxonomy,
        train_dir=train_dir,
    )
    
    # Step 3: Validate taxonomy mapping
    conditions = raw_result.get("conditions", [])
    conditions = validate_and_fix_conditions(conditions, taxonomy)
    raw_result["conditions"] = conditions
    
    # Step 4: Format output
    output = format_patient_output(raw_result, patient_id)
    
    # Step 5: Write output
    write_patient_output(output, output_dir, patient_id)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Clinical LLM Extractor — Extract structured conditions from patient notes"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory containing patient_XX folders"
    )
    parser.add_argument(
        "--patient-list",
        required=True,
        help="Path to JSON file containing list of patient IDs to process"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory for result JSON files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load patient list
    patient_list_path = Path(args.patient_list)
    if not patient_list_path.exists():
        logger.error(f"Patient list not found: {patient_list_path}")
        sys.exit(1)
    
    with open(patient_list_path, "r") as f:
        patient_ids = json.load(f)
    
    if not isinstance(patient_ids, list):
        logger.error("Patient list must be a JSON array of patient IDs")
        sys.exit(1)
    
    logger.info(f"Processing {len(patient_ids)} patients")
    
    # Load taxonomy
    data_dir = Path(args.data_dir)
    taxonomy = config.load_taxonomy()
    
    # Override temperature if specified
    config.LLM_TEMPERATURE = args.temperature
    
    # Find training directory for few-shot examples
    train_dir = find_train_dir(data_dir)
    if train_dir:
        logger.info(f"Using few-shot examples from: {train_dir}")
    else:
        logger.warning("No training directory found — proceeding without few-shot examples")
    
    # Initialize LLM client
    llm = LLMClient()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each patient
    results = {}
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        try:
            result = process_patient(
                llm=llm,
                data_dir=data_dir,
                patient_id=patient_id,
                taxonomy=taxonomy,
                train_dir=train_dir,
                output_dir=output_dir,
            )
            if result:
                results[patient_id] = result
                n_conds = len(result.get("conditions", []))
                logger.info(f"[OK] {patient_id}: {n_conds} conditions extracted")
            else:
                logger.error(f"[FAIL] {patient_id}: extraction failed")
        except Exception as e:
            logger.error(f"[FAIL] {patient_id}: {e}", exc_info=True)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processed {len(results)}/{len(patient_ids)} patients successfully")
    logger.info(llm.get_usage_summary())
    logger.info(f"Output written to: {output_dir}")
    
    # Return non-zero exit code if any patients failed
    if len(results) < len(patient_ids):
        sys.exit(1)


if __name__ == "__main__":
    main()
