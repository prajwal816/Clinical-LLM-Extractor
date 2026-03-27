"""Document loader for clinical markdown notes with robust date extraction."""

import re
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MONTH_MAP = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
    "1": "January", "2": "February", "3": "March", "4": "April",
    "5": "May", "6": "June", "7": "July", "8": "August",
    "9": "September",
}


@dataclass
class Note:
    """Represents a single clinical note."""
    note_id: str                        # e.g. "text_0"
    filepath: Path
    raw_text: str
    lines: list[str]                    # 1-indexed content (index 0 = "")
    encounter_date: str | None = None   # normalized "Month YYYY" or "DD Month YYYY"

    @property
    def num_lines(self) -> int:
        return len(self.lines) - 1


def _parse_date_to_month_year(date_str: str) -> str | None:
    """Convert various date formats to 'Month YYYY' or 'DD Month YYYY'."""
    month_names = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # MM/DD/YYYY or M/D/YYYY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', date_str)
    if m:
        month = int(m.group(1))
        year = m.group(3)
        if len(year) == 2:
            year = "20" + year if int(year) < 50 else "19" + year
        if 1 <= month <= 12:
            return f"{month_names[month]} {year}"

    # DD.MM.YYYY
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', date_str)
    if m:
        month = int(m.group(2))
        year = m.group(3)
        if 1 <= month <= 12:
            return f"{month_names[month]} {year}"

    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_str)
    if m:
        month = int(m.group(2))
        year = m.group(1)
        if 1 <= month <= 12:
            return f"{month_names[month]} {year}"

    return None


def _extract_encounter_date(text: str) -> str | None:
    """
    Extract the encounter/admission date from the note header.
    
    Looks for patterns in the first ~2000 chars:
      - "from MM/DD/YYYY to MM/DD/YYYY" -> first date
      - "from DD.MM.YYYY to DD.MM.YYYY"
      - "admitted ... MM/DD/YYYY"
      - "treated ... from ... to ..."
      - "on MM/DD/YYYY"
      - Standalone date references near the top
    """
    header = text[:2000]

    # Pattern 1: "from DATE to DATE" (most common in these notes)
    m = re.search(
        r'from\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+to\s+(\d{1,2}/\d{1,2}/\d{2,4})',
        header, re.IGNORECASE
    )
    if m:
        result = _parse_date_to_month_year(m.group(1))
        if result:
            return result

    # Pattern 2: "from DD.MM.YYYY to DD.MM.YYYY"
    m = re.search(
        r'from\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+to\s+(\d{1,2}\.\d{1,2}\.\d{4})',
        header, re.IGNORECASE
    )
    if m:
        result = _parse_date_to_month_year(m.group(1))
        if result:
            return result

    # Pattern 3: "admitted ... on DATE" or "treated ... on DATE"
    m = re.search(
        r'(?:admitted|treated|seen|visit|presentation)\s+.*?(\d{1,2}/\d{1,2}/\d{2,4})',
        header, re.IGNORECASE
    )
    if m:
        result = _parse_date_to_month_year(m.group(1))
        if result:
            return result

    # Pattern 4: Any MM/DD/YYYY near top
    dates_slash = re.findall(r'(\d{1,2}/\d{1,2}/\d{2,4})', header)
    for d in dates_slash:
        result = _parse_date_to_month_year(d)
        if result:
            return result

    # Pattern 5: Any DD.MM.YYYY
    dates_dot = re.findall(r'(\d{1,2}\.\d{1,2}\.\d{4})', header)
    for d in dates_dot:
        result = _parse_date_to_month_year(d)
        if result:
            return result

    # Pattern 6: Month name patterns "January 2021", "in March 2020"
    m = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        header
    )
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Pattern 7: "3 April 2017" or "03.04.2017"
    m = re.search(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', header)
    if m:
        return f"{m.group(2)} {m.group(3)}"

    return None


def load_patient_notes(data_dir: str | Path, patient_id: str) -> list[Note]:
    """
    Load all markdown notes for a patient, sorted chronologically by filename.
    """
    patient_dir = Path(data_dir) / patient_id
    if not patient_dir.exists():
        logger.warning(f"Patient directory not found: {patient_dir}")
        return []

    note_files = sorted(
        patient_dir.glob("text_*.md"),
        key=lambda p: int(re.search(r'text_(\d+)', p.stem).group(1))
    )

    notes = []
    for filepath in note_files:
        note_id = filepath.stem
        raw_text = filepath.read_text(encoding="utf-8")

        # Build 1-indexed line list
        raw_lines = raw_text.split("\n")
        lines = [""] + raw_lines

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
