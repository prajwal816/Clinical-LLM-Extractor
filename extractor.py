"""Core condition extraction pipeline using LLM."""

import json
import logging
from pathlib import Path

from config import build_taxonomy_prompt_section, load_taxonomy
from document_loader import Note, format_note_for_prompt
from few_shot_builder import get_few_shot_messages
from llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical NLP expert. Your task is to extract ALL medical conditions from clinical notes for a single patient and produce a structured JSON output.

## Task
Analyze the provided clinical notes (with line numbers) and extract every medical condition, diagnosis, and clinically significant finding. For each condition, provide:
- condition_name: Specific, descriptive name
- category: Must be a valid category from the taxonomy
- subcategory: Must be a valid subcategory from the taxonomy  
- status: "active", "resolved", or "suspected" based on the LATEST note where the condition appears
- onset: Earliest date the condition is documented (see date rules below)
- evidence: ALL supporting text spans across ALL notes with note_id, line_no, and exact span text

## CRITICAL RULES

### Condition Identification
1. Extract ALL conditions: diagnoses, findings, lab abnormalities, procedures (as resolved conditions)
2. One entry per DISTINCT condition. If same condition affects multiple anatomical sites, create SEPARATE entries (e.g., brain metastasis and liver metastasis)
3. ONLY extract conditions that fit the taxonomy categories
4. Look in ALL sections: Diagnoses, Other Diagnoses, Medical History, exam findings, lab results, imaging, narrative text

### Status Rules
- **active**: Confirmed and currently present — newly diagnosed, chronic, worsening, recurrent, appears in "Diagnoses" section, has current treatment
- **resolved**: No longer present — "status post", "history of", appears only in "Medical History", "in remission", "no evidence of disease"
- **suspected**: Not confirmed — "suspected", "suspicion of", "suggestive of", "rule out", "possible"
- Status reflects the LATEST note where the condition appears. If "suspected" in text_0 becomes "active" in text_3, report "active"

### Date/Onset Rules
- Priority 1 — Stated date: If notes give specific date (e.g., "first diagnosis 03/2021", "cholecystectomy in 2015"), use that
- Priority 2 — Note date: If no date stated, use encounter date of earliest note where condition appears
- Priority 3 — Relative dates: Convert using note context (e.g., "since mid-December" in January 2017 note → "December 2016")
- Do NOT infer onset across conditions (fatigue onset ≠ thyroid disorder onset)
- Formats: "16 March 2026", "March 2014", "2014", or null if unknown

### Evidence Rules
- Include evidence from EVERY note where the condition is mentioned
- Each evidence entry needs: note_id (e.g., "text_0"), line_no (integer), span (exact text from that line)
- Include mentions in admission notes, discharge notes, follow-ups, lists, narrative text
- The span must be the actual text from the note at that line number

### Disambiguation
- Heart failure: categorize by underlying cause, not as cardiovascular.structural (unless cause unknown or primary myocardial)
- Diabetic complications: categorize under metabolic_endocrine.diabetes, NOT the affected organ's category

{taxonomy_section}

## Output Format
Return ONLY valid JSON with this structure:
```json
{{
  "patient_id": "patient_XX",
  "conditions": [
    {{
      "condition_name": "...",
      "category": "...",
      "subcategory": "...",
      "status": "active|resolved|suspected",
      "onset": "...",
      "evidence": [
        {{"note_id": "text_0", "line_no": 12, "span": "exact text from note"}}
      ]
    }}
  ]
}}
```

Be thorough and comprehensive. Extract ALL conditions — missing conditions is worse than having too many.
"""


def _build_system_message(taxonomy: dict) -> dict[str, str]:
    """Build the system message with taxonomy embedded."""
    taxonomy_section = build_taxonomy_prompt_section(taxonomy)
    content = SYSTEM_PROMPT.replace("{taxonomy_section}", taxonomy_section)
    return {"role": "system", "content": content}


def _build_patient_user_message(
    notes: list[Note],
    patient_id: str,
) -> str:
    """Build the user message containing all patient notes."""
    parts = [f"Extract ALL conditions for {patient_id} from these clinical notes:\n"]
    for note in notes:
        parts.append(format_note_for_prompt(note))
        parts.append("")  # blank line separator
    return "\n".join(parts)


def extract_conditions_for_patient(
    llm: LLMClient,
    notes: list[Note],
    patient_id: str,
    taxonomy: dict,
    train_dir: Path | None = None,
) -> dict:
    """
    Run the full extraction pipeline for a single patient.
    
    Strategy: Single LLM call with all notes + few-shot example.
    For patients with many/long notes, we batch into chunks.
    
    Returns:
        Dict with patient_id and conditions list.
    """
    # Build messages
    system_msg = _build_system_message(taxonomy)
    
    # Few-shot examples from training
    few_shot_msgs = []
    if train_dir and train_dir.exists():
        few_shot_msgs = get_few_shot_messages(train_dir)
    
    # Build user message with all notes
    user_content = _build_patient_user_message(notes, patient_id)
    
    # Check if we need to chunk (rough estimate: 4 chars per token, limit ~100k tokens)
    total_chars = len(user_content)
    for msg in few_shot_msgs:
        total_chars += len(msg.get("content", ""))
    total_chars += len(system_msg["content"])
    
    if total_chars > 350000:  # ~87k tokens — chunk the notes
        logger.info(f"Large patient ({total_chars} chars) — using chunked extraction")
        return _extract_chunked(llm, notes, patient_id, taxonomy, system_msg, few_shot_msgs)
    
    # Single call for normal-sized patients
    messages = [system_msg] + few_shot_msgs + [
        {"role": "user", "content": user_content}
    ]
    
    logger.info(f"Extracting conditions for {patient_id} ({len(notes)} notes, ~{total_chars//4} tokens)")
    
    try:
        result = llm.chat_json(messages, max_tokens=16000)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for {patient_id}: {e}")
        # Retry with explicit JSON instruction
        messages.append({"role": "user", "content": "Please return ONLY valid JSON. No other text."})
        result = llm.chat_json(messages, max_tokens=16000)
    
    # Ensure patient_id is correct
    if isinstance(result, dict):
        result["patient_id"] = patient_id
    
    return result


def _extract_chunked(
    llm: LLMClient,
    notes: list[Note],
    patient_id: str,
    taxonomy: dict,
    system_msg: dict,
    few_shot_msgs: list[dict],
) -> dict:
    """
    For patients with very long notes, extract in phases:
    Phase 1: Extract from each note (or small group) separately
    Phase 2: Consolidate all extracted conditions
    """
    # Phase 1: Per-note extraction
    all_conditions = []
    
    # Group notes into chunks of 2-3
    chunk_size = 2
    for i in range(0, len(notes), chunk_size):
        chunk = notes[i:i + chunk_size]
        user_content = _build_patient_user_message(chunk, patient_id)
        
        messages = [system_msg] + few_shot_msgs + [
            {"role": "user", "content": user_content}
        ]
        
        try:
            result = llm.chat_json(messages, max_tokens=16000)
            conditions = result.get("conditions", []) if isinstance(result, dict) else []
            all_conditions.extend(conditions)
        except Exception as e:
            logger.error(f"Chunk extraction failed for {patient_id} notes {i}-{i+chunk_size}: {e}")
    
    # Phase 2: Consolidation
    if not all_conditions:
        return {"patient_id": patient_id, "conditions": []}
    
    consolidation_prompt = f"""You previously extracted conditions from individual notes for {patient_id}.
Now consolidate them into a single unified list:

1. MERGE duplicate conditions (same condition mentioned across notes) into one entry
2. For merged conditions, set status to the one from the LATEST note  
3. For merged conditions, set onset to the EARLIEST date
4. For merged conditions, combine ALL evidence entries
5. Keep the most specific condition_name
6. Ensure all category/subcategory pairs are valid per the taxonomy

Previously extracted conditions:
```json
{json.dumps(all_conditions, indent=2, ensure_ascii=False)}
```

Return the consolidated JSON with the same schema:
```json
{{"patient_id": "{patient_id}", "conditions": [...]}}
```"""
    
    messages = [system_msg, {"role": "user", "content": consolidation_prompt}]
    
    try:
        result = llm.chat_json(messages, max_tokens=16000)
        if isinstance(result, dict):
            result["patient_id"] = patient_id
        return result
    except Exception as e:
        logger.error(f"Consolidation failed for {patient_id}: {e}")
        return {"patient_id": patient_id, "conditions": all_conditions}
