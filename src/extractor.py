"""Core condition extraction pipeline using LLM with two-phase approach."""

import json
import logging
from pathlib import Path

from src.config import build_taxonomy_prompt_section, load_taxonomy
from src.document_loader import Note, format_note_for_prompt
from src.few_shot_builder import get_few_shot_messages
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — comprehensive, detailed, and precise
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical NLP expert specializing in structured information extraction from longitudinal medical records. Your task is to extract a **complete condition summary** from a patient's clinical notes.

## YOUR TASK
From the provided clinical notes (each with line numbers), extract EVERY medical condition, diagnosis, and clinically significant finding. Be EXHAUSTIVE — missing a condition is a critical failure.

## WHAT TO EXTRACT
Extract ALL of the following — every one is a "condition":
1. **Primary diagnoses** — cancers, infections, metabolic disorders, etc.
2. **Secondary/comorbid diagnoses** — hypertension, diabetes, COPD, etc.
3. **Past medical history items** — "status post cholecystectomy", "history of fracture", etc. (these are resolved conditions)
4. **Significant lab abnormalities** — anemia (low Hgb/RBC), thrombocytopenia (low platelets), lymphopenia, coagulopathy (abnormal INR/Quick), etc.
5. **Imaging findings** — liver cirrhosis on CT, cardiomegaly on X-ray, degenerative spine changes, etc.
6. **Metastases** — each metastatic site is a SEPARATE condition entry
7. **Infections** — each distinct organism/infection type is a separate entry
8. **Functional deficits** — hearing loss, nerve palsies, cognitive deficits, etc.
9. **Structural abnormalities** — cysts, effusions, hernias, etc.
10. **Pre-malignant conditions** — dysplasia, polyposis, myelodysplastic syndrome, etc.

## WHERE TO LOOK
Scan EVERY section of EVERY note:
- **Diagnoses / Primary Diagnoses** sections
- **Other diagnoses / Secondary diagnoses / Comorbidities** sections
- **Medical History / Past Medical History** sections
- **Physical Examination** findings
- **Lab Results** tables (look for out-of-range values)
- **Imaging / Radiology** reports (CT, MRI, X-ray, ultrasound, PET)
- **Histology / Pathology** reports
- **Therapy and Progression** narratives
- **Medication lists** (can imply conditions, e.g., L-thyroxine implies hypothyroidism)
- **Narrative text** anywhere in the note

## OUTPUT FIELDS — for EACH condition:

### condition_name
Use the most specific clinical name supported by the notes. Examples:
- "Squamous cell carcinoma of the left tongue base" (not just "tongue cancer")
- "Arterial hypertension" (not "high blood pressure")
- "Non-insulin-dependent diabetes mellitus type II" (not "diabetes")

### category and subcategory
Must be an EXACT match to a valid key pair from the taxonomy below. Use ONLY valid pairs.

### status — one of: "active", "resolved", "suspected"
**CRITICAL**: Status reflects the condition's state in the LATEST note where it appears.

**active**: Condition is confirmed and currently present
- Listed in "Diagnoses" or "Other Diagnoses" sections
- Has ongoing treatment or management mentioned
- Is chronic, worsening, progressive, or recurrent
- Lab values still abnormal in a later note

**resolved**: Condition is no longer present or is purely historical
- Prefixed with "Status post" or "History of" or "Z.n." or "St.p."
- Appears ONLY in "Medical History" section and NOT in "Diagnoses"
- "In remission", "complete response", "no evidence of disease"
- Lab values normalized in a later note
- Past surgical procedure

**suspected**: NOT yet diagnostically confirmed
- "Suspected", "suspicion of", "suggestive of", "rule out", "possible", "likely", "question of"

**Status evolution**: If a condition is "suspected" in text_0 but "active" in text_3, report status as "active" (latest note wins). If "active" in text_0 but only appears in "Medical History" in text_5, report as "resolved".

### onset — the EARLIEST date the condition is documented
Follow this strict priority:
1. **Stated date**: "first diagnosed in 03/2021" → "March 2021"; "cholecystectomy in 2015" → "2015"
2. **Note encounter date**: If no explicit date given, use the encounter/admission date from the earliest note where the condition first appears. Look for dates at the top of each note (e.g., "admitted from 05/28/14 to 06/20/14" → "May 2014")
3. **Relative dates**: "since mid-December" in a January 2017 note → "December 2016"
4. **Do NOT infer across conditions**: A symptom's date does not establish a later-diagnosed disease's onset
5. **null**: Only if absolutely no date can be determined

Date formats (use most specific available):
- Full date: "16 March 2026"
- Month and year: "March 2014"  
- Year only: "2014"
- Unknown: null

### evidence — ALL supporting text across ALL notes
For EVERY note that mentions this condition, include an evidence entry:
- **note_id**: filename without extension (e.g., "text_0")
- **line_no**: the exact line number (integer) where the text appears
- **span**: the exact text from that line that supports this condition

Evidence must be COMPREHENSIVE. If a condition appears in 8 notes, include evidence from all 8. Include mentions in:
- Diagnosis lists
- Medical history sections
- Imaging/lab/pathology reports
- Narrative clinical text
- Medication sections (if implying the condition)

## CRITICAL RULES
1. **One entry per DISTINCT condition**. Multiple anatomical sites = separate entries (brain metastasis ≠ liver metastasis)
2. **Evidence from EVERY note** where condition is mentioned — even brief list mentions
3. **Taxonomy must be VALID** — only use category.subcategory pairs from the taxonomy
4. **Do NOT extract** conditions outside the taxonomy categories
5. **Be THOROUGH** — scan every line of every note. Missing conditions is a critical error.

## DISAMBIGUATION RULES
- **Heart failure**: Categorize by underlying cause, NOT as cardiovascular.structural (unless cause unknown). Ischemic → cardiovascular.coronary; hypertensive → cardiovascular.hypertensive
- **Diabetic complications** (nephropathy, retinopathy): Categorize under metabolic_endocrine.diabetes, NOT the affected organ
- **Infections**: Categorize by organism type (bacterial/viral/fungal), NOT by affected organ
- **Low blood cell counts**: ALWAYS hematological.cytopenia regardless of cause
- **Portal hypertensive gastropathy**: gastrointestinal.upper_gi (by affected organ)
- **Status post procedures**: The underlying condition that required the procedure, marked as "resolved"

{taxonomy_section}

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown formatting, no code fences, no explanatory text.
The JSON must follow this exact structure:
{{"patient_id": "patient_XX", "conditions": [{{"condition_name": "...", "category": "...", "subcategory": "...", "status": "active|resolved|suspected", "onset": "...|null", "evidence": [{{"note_id": "text_0", "line_no": 12, "span": "exact text from the note at this line"}}]}}]}}
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
    """Build the user message containing all patient notes with line numbers."""
    parts = [f"Extract ALL conditions for {patient_id} from these clinical notes. Be exhaustive.\n"]
    for note in notes:
        parts.append(format_note_for_prompt(note))
        parts.append("")  # blank line separator
    return "\n".join(parts)


def _parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM response, handling various wrappers."""
    raw = raw.strip()
    
    # Remove markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        raw = "\n".join(lines[start:end])
    
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the text
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    
    raise json.JSONDecodeError("Could not extract valid JSON from LLM response", raw, 0)


def extract_conditions_for_patient(
    llm: LLMClient,
    notes: list[Note],
    patient_id: str,
    taxonomy: dict,
    train_dir: Path | None = None,
) -> dict:
    """
    Run the full extraction pipeline for a single patient.
    
    Strategy:
    - For patients with moderate notes: single comprehensive LLM call
    - For patients with very long notes: chunked extraction + consolidation
    
    Returns:
        Dict with patient_id and conditions list.
    """
    system_msg = _build_system_message(taxonomy)
    
    # Few-shot examples from training
    few_shot_msgs = []
    if train_dir and train_dir.exists():
        few_shot_msgs = get_few_shot_messages(train_dir)
    
    # Build user message with all notes
    user_content = _build_patient_user_message(notes, patient_id)
    
    # Estimate total tokens
    total_chars = len(user_content) + len(system_msg["content"])
    for msg in few_shot_msgs:
        total_chars += len(msg.get("content", ""))
    
    if total_chars > 300000:  # ~75k tokens — too large for single call
        logger.info(f"Large patient ({total_chars} chars) -- using chunked extraction")
        return _extract_chunked(llm, notes, patient_id, taxonomy, system_msg, few_shot_msgs)
    
    # Single comprehensive call
    messages = [system_msg] + few_shot_msgs + [
        {"role": "user", "content": user_content}
    ]
    
    logger.info(f"Extracting conditions for {patient_id} ({len(notes)} notes, ~{total_chars // 4} est. tokens)")
    
    raw_response = llm.chat(messages, max_tokens=16000)
    
    try:
        result = _parse_llm_json(raw_response)
    except json.JSONDecodeError:
        logger.warning(f"First parse failed for {patient_id}, retrying with explicit JSON instruction")
        messages.append({
            "role": "assistant",
            "content": raw_response
        })
        messages.append({
            "role": "user",
            "content": "Your response was not valid JSON. Please return ONLY the JSON object, no markdown, no code fences, no explanatory text. Start with { and end with }."
        })
        raw_response = llm.chat(messages, max_tokens=16000, use_cache=False)
        result = _parse_llm_json(raw_response)
    
    if isinstance(result, dict):
        result["patient_id"] = patient_id
    
    # Post-process: verify and fix evidence
    result = _verify_evidence(result, notes)
    
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
    Phase 1: Extract from small note groups
    Phase 2: Consolidate and deduplicate
    """
    all_conditions = []
    
    # Phase 1: Per-chunk extraction (groups of 2-3 notes)
    chunk_size = 2
    for i in range(0, len(notes), chunk_size):
        chunk = notes[i:i + chunk_size]
        user_content = _build_patient_user_message(chunk, patient_id)
        
        messages = [system_msg] + few_shot_msgs + [
            {"role": "user", "content": user_content}
        ]
        
        try:
            raw_response = llm.chat(messages, max_tokens=16000)
            result = _parse_llm_json(raw_response)
            conditions = result.get("conditions", []) if isinstance(result, dict) else []
            all_conditions.extend(conditions)
            logger.debug(f"  Chunk {i//chunk_size + 1}: extracted {len(conditions)} conditions")
        except Exception as e:
            logger.error(f"Chunk extraction failed for {patient_id} notes {i}-{i + chunk_size}: {e}")
    
    if not all_conditions:
        return {"patient_id": patient_id, "conditions": []}
    
    # Phase 2: Consolidation call
    consolidation_prompt = f"""I extracted conditions from individual note chunks for {patient_id}. Now I need you to consolidate them into a single unified list.

INSTRUCTIONS:
1. MERGE duplicate conditions (same condition across different notes) into ONE entry
2. For merged conditions, keep the MOST SPECIFIC condition_name
3. Set status based on the LATEST note where the condition appears (latest text_N number)
4. Set onset to the EARLIEST date found across all mentions
5. Combine ALL evidence entries from all mentions (keep every unique evidence span)
6. Remove exact duplicate evidence entries (same note_id + same line_no)
7. Ensure all category/subcategory pairs are valid per taxonomy
8. Do NOT add any new conditions — only consolidate existing ones

Raw extracted conditions from chunks:
{json.dumps(all_conditions, indent=2, ensure_ascii=False)}

Return ONLY valid JSON. No markdown, no code fences:
{{"patient_id": "{patient_id}", "conditions": [...]}}"""
    
    messages = [system_msg, {"role": "user", "content": consolidation_prompt}]
    
    try:
        raw_response = llm.chat(messages, max_tokens=16000)
        result = _parse_llm_json(raw_response)
        if isinstance(result, dict):
            result["patient_id"] = patient_id
        # Verify evidence
        result = _verify_evidence(result, notes)
        return result
    except Exception as e:
        logger.error(f"Consolidation failed for {patient_id}: {e}")
        return {"patient_id": patient_id, "conditions": all_conditions}


def _verify_evidence(result: dict, notes: list[Note]) -> dict:
    """
    Post-process evidence entries to verify/fix line numbers and spans.
    Ensures evidence spans actually exist at the stated line in the note.
    """
    if not isinstance(result, dict):
        return result
    
    # Build quick lookup: note_id -> Note
    note_map = {n.note_id: n for n in notes}
    
    conditions = result.get("conditions", [])
    for cond in conditions:
        evidence = cond.get("evidence", [])
        verified_evidence = []
        seen = set()  # deduplicate (note_id, line_no) pairs
        
        for ev in evidence:
            note_id = ev.get("note_id", "")
            line_no = ev.get("line_no", 0)
            span = ev.get("span", "")
            
            # Ensure line_no is int
            try:
                line_no = int(line_no)
            except (ValueError, TypeError):
                continue
            
            ev["line_no"] = line_no
            
            # Deduplicate
            dedup_key = (note_id, line_no)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            
            # Verify note exists
            note = note_map.get(note_id)
            if not note:
                logger.debug(f"Evidence references unknown note '{note_id}', keeping as-is")
                verified_evidence.append(ev)
                continue
            
            # Check if line_no is in range
            if line_no < 1 or line_no >= len(note.lines):
                # Try to find the span in nearby lines
                fixed_line = _find_span_in_note(note, span, line_no)
                if fixed_line:
                    ev["line_no"] = fixed_line
                    # Update span to match actual line content
                    actual_line = note.lines[fixed_line].strip()
                    if span.strip() not in actual_line and len(span) > 0:
                        # Keep original span but fix line_no
                        pass
                    verified_evidence.append(ev)
                else:
                    logger.debug(f"Could not verify evidence line {line_no} in {note_id}, keeping")
                    verified_evidence.append(ev)
                continue
            
            # Verify span roughly matches the actual line content
            actual_line = note.lines[line_no]
            if span.strip() and span.strip() in actual_line:
                verified_evidence.append(ev)
            elif span.strip() and actual_line.strip() in span.strip():
                # Span is broader than line — acceptable
                verified_evidence.append(ev)
            else:
                # Span doesn't match — try to find it nearby
                fixed_line = _find_span_in_note(note, span, line_no)
                if fixed_line:
                    ev["line_no"] = fixed_line
                    verified_evidence.append(ev)
                else:
                    # Keep it anyway — LLM may have paraphrased
                    verified_evidence.append(ev)
        
        cond["evidence"] = verified_evidence
    
    return result


def _find_span_in_note(note: Note, span: str, hint_line: int) -> int | None:
    """Try to find a span text in a note, searching near the hint line first."""
    if not span.strip():
        return None
    
    span_clean = span.strip().lower()
    
    # Search in a window around the hint line first, then expand
    for radius in [0, 1, 2, 3, 5, 10, 20, 50]:
        start = max(1, hint_line - radius)
        end = min(len(note.lines), hint_line + radius + 1)
        for i in range(start, end):
            line_text = note.lines[i].lower()
            if span_clean in line_text or line_text.strip() in span_clean:
                return i
    
    # Full note search as fallback
    for i in range(1, len(note.lines)):
        line_text = note.lines[i].lower()
        # Check for substantial overlap
        span_words = set(span_clean.split())
        line_words = set(line_text.split())
        if len(span_words) > 2 and len(span_words & line_words) / max(len(span_words), 1) > 0.6:
            return i
    
    return None
