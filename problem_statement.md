# Clinical Condition Extraction from Longitudinal Patient Notes

## Overview

You are given a dataset of clinical notes for multiple patients. Each patient has between 2 and 13 notes, written over months to years, documenting their medical history, diagnoses, treatments, and outcomes.

Your task is to build a system that extracts a **structured condition summary** for each patient from their clinical notes. A condition summary is an inventory of every medical diagnosis and clinically significant finding mentioned across a patient's notes.

## Task

For each patient, produce a JSON file containing every condition found across all their notes, with the following fields per condition:

```json
{
  "patient_id": "patient_XX",
  "conditions": [
    {
      "condition_name": "Descriptive name of the condition",
      "category": "category_key",
      "subcategory": "subcategory_key",
      "status": "status_key",
      "onset": "March 2014",
      "evidence": [
        {
          "note_id": "text_0",
          "line_no": 12,
          "span": "The exact text from the note supporting this condition"
        }
      ]
    }
  ]
}
```

### Field Definitions

- **condition_name**: A human-readable name for the condition. Use the most specific name supported by the notes.
- **category** and **subcategory**: Must be a valid key from the provided taxonomy (see `taxonomy.json`).
- **status**: One of `active`, `resolved`, or `suspected` (see `taxonomy.json` for definitions). Reflects the condition's state as of the **latest note** where it appears.
- **onset**: The earliest date the condition is **explicitly documented** in the notes — either by name or by a clearly equivalent clinical description. Follow this priority:
  1. **Stated date**: If the notes give a specific date for the condition (e.g., "first diagnosis of chronic lymphocytic leukemia: 03/2021", "cholecystectomy in 2015"), use that date.
  2. **Note date**: If no date is explicitly stated for the condition, use the encounter date of the earliest note where the condition is named or described.
  3. **Relative dates**: Convert relative references to absolute dates using the note's context (e.g., "since mid-December" in a January 2017 note → `"December 2016"`).
  4. **Do not infer across conditions**: A symptom's onset date does not set the onset of a different underlying condition (e.g., "persistent cough since January" does not establish the onset of a later-diagnosed lung carcinoma — the carcinoma's onset is its own diagnosis or documentation date).
  5. **Unknown**: Use `null` only when no date can be determined.
  - Use the most specific format available:
    - Full date: `"16 March 2026"`
    - Month and year: `"March 2014"`
    - Year only: `"2014"`
    - Unknown: `null`
- **evidence**: An array of all supporting excerpts across all notes where this condition is mentioned or referenced. Each entry requires:
  - `note_id`: The filename without extension (e.g., `"text_0"`)
  - `line_no`: The line number in the file where the supporting text appears
  - `span`: The exact text from the note that supports the condition's presence or status

### Key Rules

1. **One entry per distinct condition**: If the same condition affects multiple anatomical sites, create a separate entry for each site (e.g., brain metastasis and liver metastasis are two entries).
2. **Status reflects the latest note**: If a condition evolves from "suspected" in text_0 to "active" in text_3, report status as "active". The evidence array should include entries from both notes.
3. **Evidence must be comprehensive**: Include evidence from every note where the condition is mentioned. This means citing the condition in admission notes, discharge notes, follow-up notes, and any note that references it — even if the mention is brief or in a list.
4. **Taxonomy is strict**: Every condition must map to exactly one `category.subcategory` pair from the taxonomy.
5. **Scope**: Do not extract categories outside the taxonomy.

## Taxonomy

The full taxonomy is provided in `taxonomy.json`. It contains:

- **Condition categories and subcategories**: 13 categories with specific subcategories. Each subcategory has a description and examples.
- **Status values**: 3 possible statuses (`active`, `resolved`, `suspected`) with definitions and signals to look for in the notes.
- **Disambiguation rules and notes**: A small number of rules for handling edge cases. Read these carefully. You may add your own disambiguation rules if you find additional edge cases that benefit from explicit handling — document any additions in your report.

## Data

### Directory Structure

```
data/
  train/           # Labeled examples — use these to understand the task
    patient_XX/
      text_0.md
      text_1.md
      ...
    labels/
      patient_XX.json   # Ground truth extractions
  dev/             # Unlabeled — use for development and tuning
    patient_XX/
      text_0.md
      ...
```

### Notes Format

Each note is a markdown file representing a clinical document (discharge summary, consultation letter, follow-up report, surgical note, etc.). Notes are ordered chronologically by filename (`text_0.md` is the earliest).

Notes are written as referral letters between physicians and typically contain sections such as:

- **Diagnoses** — primary diagnoses for the current encounter
- **Other Diagnoses** — comorbidities and secondary conditions
- **Medical History** — past conditions and prior procedures
- **Current Presentation** — reason for visit, symptoms
- **Physical Examination** — examination findings
- **Medications** — current medication lists
- **Lab results** — laboratory values
- **Imaging** — radiology reports
- **Therapy and Progression** — treatment course
- **Recommendations** — follow-up plans

Not all sections appear in every note. Section headers may vary.

## Submission

Submit a **code repository** and a **PDF report**.

### Code

Your code must include:

- **`main.py`** — The entrypoint to your pipeline. It must accept the following **mandatory** CLI arguments:
  - `--data-dir` — Path to the data directory (same structure as `data/train/` or `data/dev/`)
  - `--patient-list` — Path to a JSON file containing a list of patient IDs to process. Example:
    ```json
    ["patient_01", "patient_03", "patient_17"]
    ```
  - `--output-dir` — Path to the directory where output files will be written. Your pipeline must write one JSON file per patient, named `patient_XX.json`, following the output schema defined in the Task section above.

  You may add any additional CLI arguments you find useful (e.g., `--temperature`, `--batch-size`, `--verbose`). Document these in your report.

- **`requirements.txt`** — All Python dependencies required to run your pipeline. Your code should be runnable after `pip install -r requirements.txt`.

- **LLM calls** — All LLM calls must use an **OpenAI-compatible API**. Your code must read the following environment variables:
  - `OPENAI_BASE_URL` — The API base URL
  - `OPENAI_API_KEY` — The API key
  - `OPENAI_MODEL` — The model identifier

  Do **not** hardcode model names or API endpoints.

Example invocation:
```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="model-name"

python main.py \
  --data-dir ./data/dev \
  --patient-list ./patients.json \
  --output-dir ./output
```

### Report

Submit a PDF detailing:

- Your approach and design decisions
- Experiments performed and results
- What worked, what didn't, and why
- Instructions for running the code, including any additional CLI arguments

## Evaluation

Your submission will be evaluated both quantitatively and qualitatively.

### Quantitative Evaluation

Your pipeline will be run on a held-out test set using the **same model for all participants**. The model will be provided via the `OPENAI_MODEL` environment variable at evaluation time. Results will be scored on:

1. **Condition identification** — Did you find all conditions? Did you avoid false positives?
2. **Status accuracy** — For correctly identified conditions, did you assign the right status?
3. **Date accuracy** — For correctly identified conditions, did you assign the correct onset date?
4. **Evidence quality** — For correctly identified conditions, did you cite the right evidence without noise?
5. **Speed** — Wall-clock time to process the test set.
6. **Cost** — Total tokens consumed (input + output) across all LLM calls.

### Qualitative Evaluation

Based on your submitted report:

1. **Experimentation rigor** — How systematically did you explore, evaluate, and iterate on your approach?
2. **Creativity of approach** — Did you bring thoughtful ideas to the problem?

## Tips

- Start by thoroughly exploring the data before building your extraction pipeline.
- Pay attention to data quality — not all data may be as clean as you expect.
- Read the taxonomy carefully, including the notes and disambiguation rules.
- Conditions often appear differently across notes for the same patient. A condition in the "Diagnoses" section of one note may appear in "Other Diagnoses" or "Medical History" in a later note.
- Some conditions are mentioned only in narrative text, not in structured sections.

