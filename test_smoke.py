"""Quick smoke test for the pipeline modules."""
import json
from pathlib import Path
from document_loader import load_patient_notes
from config import load_taxonomy, get_valid_category_subcategory_pairs
from taxonomy_mapper import validate_and_fix_conditions
from few_shot_builder import get_few_shot_messages
from output_formatter import normalize_date

# Test document loading
notes = load_patient_notes("train", "patient_06")
print(f"Loaded {len(notes)} notes for patient_06")
for n in notes:
    print(f"  {n.note_id}: {n.num_lines} lines, date={n.encounter_date}")

# Test taxonomy loading
taxonomy = load_taxonomy()
pairs = get_valid_category_subcategory_pairs(taxonomy)
print(f"\n{len(pairs)} valid category-subcategory pairs")

# Test with training labels
with open("train/labels/patient_06.json") as f:
    label = json.load(f)
conditions = label["conditions"]
fixed = validate_and_fix_conditions(conditions, taxonomy)
print(f"\nValidation: {len(conditions)} input -> {len(fixed)} output (all should pass)")

# Test few-shot builder
msgs = get_few_shot_messages(Path("train"))
print(f"\nFew-shot: {len(msgs)} messages built")
print(f"  User msg: {len(msgs[0]['content'])} chars")
print(f"  Assistant msg: {len(msgs[1]['content'])} chars")

# Test date normalization
test_dates = [
    "March 2014", "2014", "16 March 2026", None, "05/2014",
    "02/22/2018", "03.04.2017", "2021-09-28", "unknown", ""
]
print("\nDate normalization:")
for d in test_dates:
    print(f"  '{d}' -> '{normalize_date(d)}'")

print("\n✓ All smoke tests passed!")
