"""Few-shot example builder from training labels — uses multiple patients for diversity."""

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


def _build_condensed_example(
    notes: list[Note],
    label: dict,
    max_notes: int = 2,
    max_lines_per_note: int = 50,
    max_conditions: int = 6,
) -> tuple[str, str]:
    """
    Build a condensed (user_message, assistant_message) pair for few-shot prompting.
    Shows a diverse subset of conditions with varying statuses (active, resolved, suspected).
    """
    # Format condensed notes
    note_texts = []
    for note in notes[:max_notes]:
        lines_to_show = min(max_lines_per_note, note.num_lines)
        numbered = []
        for i in range(1, lines_to_show + 1):
            numbered.append(f"[Line {i}] {note.lines[i]}")
        header = f"=== {note.note_id} ==="
        if note.encounter_date:
            header += f" (encounter date: {note.encounter_date})"
        note_texts.append(header + "\n" + "\n".join(numbered))
        if lines_to_show < note.num_lines:
            note_texts.append(f"[... {note.num_lines - lines_to_show} more lines ...]")

    user_msg = "\n\n".join(note_texts)

    # Select diverse conditions — try to include all 3 statuses
    all_conditions = label.get("conditions", [])
    selected = []
    by_status = {"active": [], "resolved": [], "suspected": []}
    for c in all_conditions:
        by_status.get(c.get("status", "active"), []).append(c)

    # Pick from each status
    for status in ["active", "resolved", "suspected"]:
        items = by_status.get(status, [])
        take = min(len(items), max(1, max_conditions // 3))
        selected.extend(items[:take])

    # Fill remaining slots
    remaining = max_conditions - len(selected)
    if remaining > 0:
        used_names = {c["condition_name"] for c in selected}
        for c in all_conditions:
            if c["condition_name"] not in used_names and remaining > 0:
                selected.append(c)
                remaining -= 1

    # Limit evidence per condition to keep example compact
    condensed = []
    for cond in selected[:max_conditions]:
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
    example_patient_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build few-shot example messages from training patients.
    Uses patient_06 (cancer + comorbidities) as the primary example.

    Returns list of message dicts suitable for inserting into chat messages.
    """
    from document_loader import load_patient_notes

    if example_patient_ids is None:
        example_patient_ids = ["patient_06"]

    labels_dir = train_dir / "labels"
    messages = []

    for pid in example_patient_ids:
        label = _load_training_label(labels_dir, pid)
        if label is None:
            logger.warning(f"No training label found for {pid}")
            continue

        notes = load_patient_notes(train_dir, pid)
        if not notes:
            logger.warning(f"No notes found for {pid}")
            continue

        user_msg, assistant_msg = _build_condensed_example(notes, label)

        messages.extend([
            {
                "role": "user",
                "content": (
                    f"Extract ALL conditions for {pid} from these clinical notes. "
                    f"Be exhaustive.\n\n{user_msg}"
                ),
            },
            {"role": "assistant", "content": assistant_msg},
        ])

    return messages
