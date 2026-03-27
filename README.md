# Clinical LLM Extractor

A production-ready NLP pipeline for extracting structured clinical condition summaries from longitudinal patient notes using Large Language Models.

## Overview

Clinical notes tell a complex story over time. A single patient may have dozens of notes spanning months or years, detailing primary diagnoses, active comorbidities, resolved historical conditions, and incidental findings.

This project uses an LLM-driven approach to:
1. Parse longitudinal patient notes (Markdown).
2. Exhaustively extract every medical condition, diagnosis, and significant finding.
3. Classify each condition into a strict taxonomy of 13 categories.
4. Determine the clinical status (`active`, `resolved`, `suspected`) based on the most recent chronological note.
5. Determine the earliest onset date across all encounters.
6. Provide specific, reproducible evidence spans (exact note and line number) for *every* mention of the condition across all notes.

## Project Structure

```text
Clinical-LLM-Extractor/
│
├── src/
│   ├── main.py                  # (Invoked via root main.py)
│   ├── config.py                # Environment configuration & taxonomy loading
│   ├── document_loader.py       # MD parsing & robust date extraction
│   ├── extractor.py             # Core LLM extraction & exact-span evidence verification
│   ├── few_shot_builder.py      # Dynamic, representative few-shot prompt construction
│   ├── llm_client.py            # Robust OpenAI client wrapper (caching, retries, token tracking)
│   ├── output_formatter.py      # Output serialization & unstructured date normalization
│   └── taxonomy_mapper.py       # Strict ML-fallback taxonomy validation (fuzzy + keyword)
│
├── scripts/
│   ├── test_smoke.py            # Local validation of pipeline core methods
│   └── validate_output.py       # Standalone mathematical JSON/Taxonomy structural validator
│
├── main.py                      # CLI Entry point
├── requirements.txt             # Python dependencies
├── taxonomy.json                # Expected taxonomy mapping dictionary
├── report.md                    # Detailed design approach and experiment tracking
└── README.md
```

## Setup & Installation

**1. Create a Virtual Environment (Optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install Dependencies**
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

**3. Set Environment Variables**
The pipeline requires an OpenAI-compatible API to function. Set the following variables:
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Or custom proxy
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"  # Or your specific model
```

## Usage

Run the main pipeline CLI to process a batch of patients. 

```bash
python main.py \
  --data-dir ./dev \
  --patient-list ./dev_patients.json \
  --output-dir ./output \
  --concurrency 4
```

### CLI Arguments
* `--data-dir` (Required): Path to the input dataset directory containing `patient_XX` subdirectories.
* `--patient-list` (Required): Path to a JSON array file containing patient identifiers to process.
* `--output-dir` (Required): Path to the directory where JSON output per patient will be written.
* `--concurrency` (Optional): Number of parallel processing threads. Dramatically improves throughput. Default: `1`.
* `--temperature` (Optional): LLM Temperature. Set low (e.g., `0.1`) for deterministic data extraction. Default: `0.1`.
* `--verbose` (Optional): Enable streaming debug logs.

## Pipeline Highlights

* **Multi-note Chunking**: The Extractor automatically detects if a patient's history exceeds standard context windows. It intelligently slices processing into multi-note chunks, followed by a programmatic + LLM consolidation phase.
* **Evidence Validation Check**: LLMs occasionally hallucinate exact line numbers. The pipeline includes a post-processing pass that strictly verifies mathematical line numbers against the raw `document_loader` text, automatically fixing slight line skews.
* **Taxonomy Fallback Mechanism**: If the LLM produces a non-standard subcategory, `taxonomy_mapper` engages a cascading fallback: Direct Match -> Fuzzy Ratio Match -> Intelligent Keyword Mapping.

## Validation Strategy

To ensure extracted records mathematically conform to the required JSON schema, run the standalone validator against your output directory:

```bash
python scripts/validate_output.py \
  --output-dir ./output \
  --taxonomy taxonomy.json
```
