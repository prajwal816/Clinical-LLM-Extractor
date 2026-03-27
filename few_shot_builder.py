"""Few-shot example builder from training labels."""

import json
import logging
from pathlib import Path

from document_loader import Note, format_note_for_prompt

logger = logging.getLogger(__name__)


def _load_training_label(labels_dir: Path, patient_id: str) -> dict | None:
    """Load a training label JSON for a patient."""
    label_path = labels_dir / f"{patient_id}.json"
    if not label_path.exists():
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_few_shot_example(
    notes: list[Note],
    label: dict,
    max_conditions: int = 5,
) -> tuple[str, str]:
    """
    Build a (user_message, assistant_message) pair for few-shot prompting.
    
    Uses a subset of conditions to keep the example concise.
    
    Returns:
        Tuple of (formatted notes text, formatted JSON output)
    """
    # Format a condensed version of the notes (first 60 lines each, max 3 notes)
    note_texts = []
    for note in notes[:3]:
        lines_to_show = min(60, note.num_lines)
        numbered = []
        for i in range(1, lines_to_show + 1):
            numbered.append(f"[Line {i}] {note.lines[i]}")
        header = f"=== {note.note_id} ==="
        if note.encounter_date:
            header += f" (encounter date: {note.encounter_date})"
        note_texts.append(header + "\n" + "\n".join(numbered) + "\n[... remaining lines omitted ...]")
    
    user_msg = "\n\n".join(note_texts)
    
    # Build a condensed output with limited conditions
    conditions = label.get("conditions", [])[:max_conditions]
    # For each condition, limit evidence to 3 entries
    condensed = []
    for cond in conditions:
        c = dict(cond)
        c["evidence"] = c.get("evidence", [])[:3]
        condensed.append(c)
    
    output = {
        "patient_id": label["patient_id"],
        "conditions": condensed,
    }
    
    assistant_msg = json.dumps(output, indent=2, ensure_ascii=False)
    return user_msg, assistant_msg


def get_few_shot_messages(
    train_dir: Path,
    example_patient_id: str = "patient_06",
) -> list[dict[str, str]]:
    """
    Build few-shot example messages from a training patient.
    
    Returns list of message dicts suitable for inserting into chat messages.
    """
    from document_loader import load_patient_notes
    
    labels_dir = train_dir / "labels"
    label = _load_training_label(labels_dir, example_patient_id)
    if label is None:
        logger.warning(f"No training label found for {example_patient_id}")
        return []
    
    notes = load_patient_notes(train_dir, example_patient_id)
    if not notes:
        logger.warning(f"No notes found for {example_patient_id}")
        return []
    
    user_msg, assistant_msg = build_few_shot_example(notes, label)
    
    return [
        {"role": "user", "content": f"Extract all conditions from these clinical notes:\n\n{user_msg}"},
        {"role": "assistant", "content": assistant_msg},
    ]
