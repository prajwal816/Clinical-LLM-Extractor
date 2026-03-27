"""Document loader for clinical markdown notes."""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Note:
    """Represents a single clinical note."""
    note_id: str                    # e.g. "text_0"
    filepath: Path
    raw_text: str
    lines: list[str]                # 1-indexed content (index 0 = "")
    encounter_date: str | None = None   # extracted date from note header

    @property
    def num_lines(self) -> int:
        return len(self.lines) - 1  # exclude placeholder index 0


def _extract_encounter_date(text: str) -> str | None:
    """
    Try to extract the encounter/admission date from the note header.
    Looks for patterns like:
      - 'from MM/DD/YYYY to MM/DD/YYYY'
      - 'from DD.MM.YYYY to DD.MM.YYYY'
      - dates near 'admitted', 'treated', 'clinic'
    Returns the earliest date found as a string, or None.
    """
    # Pattern: MM/DD/YYYY or MM/DD/YY
    dates_slash = re.findall(
        r'(\d{1,2}/\d{1,2}/\d{2,4})', text[:1500]
    )
    # Pattern: DD.MM.YYYY
    dates_dot = re.findall(
        r'(\d{1,2}\.\d{1,2}\.\d{4})', text[:1500]
    )

    all_dates = []
    for d in dates_slash:
        all_dates.append(d)
    for d in dates_dot:
        all_dates.append(d)

    if all_dates:
        return all_dates[0]
    return None


def load_patient_notes(data_dir: str | Path, patient_id: str) -> list[Note]:
    """
    Load all markdown notes for a patient, sorted chronologically by filename.
    
    Args:
        data_dir: Path to data directory (containing patient_XX folders)
        patient_id: e.g. "patient_06"
    
    Returns:
        List of Note objects, ordered by text_N filename
    """
    patient_dir = Path(data_dir) / patient_id
    if not patient_dir.exists():
        logger.warning(f"Patient directory not found: {patient_dir}")
        return []

    note_files = sorted(patient_dir.glob("text_*.md"),
                        key=lambda p: int(re.search(r'text_(\d+)', p.stem).group(1)))

    notes = []
    for filepath in note_files:
        note_id = filepath.stem  # e.g. "text_0"
        raw_text = filepath.read_text(encoding="utf-8")

        # Build 1-indexed line list: index 0 is empty placeholder
        raw_lines = raw_text.split("\n")
        lines = [""] + raw_lines  # lines[1] = first line of file

        encounter_date = _extract_encounter_date(raw_text)

        note = Note(
            note_id=note_id,
            filepath=filepath,
            raw_text=raw_text,
            lines=lines,
            encounter_date=encounter_date,
        )
        notes.append(note)
        logger.debug(f"Loaded {note_id}: {note.num_lines} lines, date={encounter_date}")

    logger.info(f"Loaded {len(notes)} notes for {patient_id}")
    return notes


def format_note_for_prompt(note: Note) -> str:
    """Format a note with line numbers for inclusion in an LLM prompt."""
    numbered_lines = []
    for i in range(1, len(note.lines)):
        numbered_lines.append(f"[Line {i}] {note.lines[i]}")
    header = f"=== {note.note_id} ==="
    if note.encounter_date:
        header += f" (encounter date: {note.encounter_date})"
    return header + "\n" + "\n".join(numbered_lines)
