# Clinical LLM Extractor — Final Report

## 1. Approach and Design Decisions

The goal of this project was to build a robust, comprehensive system for extracting detailed medical condition summaries from longitudinal clinical notes. Given the complexity of clinical texts, the variability of how conditions are documented (narrative vs. lists), and the strict constraints around taxonomy, status, onset date, and evidence, my approach heavily relied on **context-aware LLM extraction augmented with strict programmatic validation.**

### 1.1 Pipeline Architecture
The pipeline is divided into five core modular components:
1. **Document Loading (`document_loader.py`)**: Parses markdown notes, enforces 1-indexed lines, and extracts encounter dates from note headers using an array of robust regex patterns.
2. **Context-Aware LLM Extraction (`extractor.py`)**: The engine of the pipeline. It constructs a highly detailed prompt embedding taxonomy constraints directly into the LLM instructions.
3. **Dynamic Few-Shot Builder (`few_shot_builder.py`)**: Instead of relying purely on zero-shot behavior, the pipeline dynamically loads labeled training examples to construct highly representative few-shot prompts. This teaches the model the nuance between "active", "resolved", and "suspected" conditions.
4. **Taxonomy & Schema Validation (`taxonomy_mapper.py` & `output_formatter.py`)**: Post-processes the LLM's output. It enforces exact mappings to the 13 provided categories and normalizes dates (e.g. converting "03/2021" or "March 2021" reliably).
5. **Parallel Orchestration (`main.py`)**: A `ThreadPoolExecutor` dispatches patient processing concurrently, drastically reducing wall-clock time on large datasets.

### 1.2 Two-Phase Extraction Strategy
A key design challenge was managing the context window size and "lost in the middle" phenomena for patients with massive longitudinal histories (e.g., 10+ notes). 
- **Standard Strategy**: For moderately sized patient histories (under ~75,000 tokens), all notes are concatenated chronologically and processed in a single LLM call. This gives the model maximal context.
- **Chunked Strategy**: For extremely long histories, the notes are processed in chunks (2-3 notes at a time) to extract condition candidates. A second "Consolidation Phase" LLM call is then invoked to merge duplicate conditions, determine the most recent status, resolve the earliest onset date, and compile all evidence across the individual chunks.

### 1.3 Strict Evidence Verification
LLMs are prone to hallucinating line numbers or slightly paraphrasing evidence spans. To ensure compliance, I implemented a programmatic _Evidence Verification_ pass in `extractor.py`:
- It cross-references the LLM-provided `line_no` and `span` against the actual loaded document.
- If the span isn't on the exact line, it searches within an expanding radius (±1, ±3, ±10 lines) and ultimately the entire note to find the exact matching span and correct the line number.
- It deduplicates identical evidence entries.

## 2. Experiments Performed & Results

### 2.1 Prompt Engineering Variations
- **Experiment 1 (Zero-Shot)**: Initial testing with zero-shot prompts resulted in the model frequently missing comorbidities mentioned only briefly in "Medical History" sections.
- **Experiment 2 (Few-Shot)**: Adding condensed examples from `patient_06` dramatically improved recall and formatting consistency, though the LLM occasionally grouped unrelated conditions.
- **Refinement**: The final `SYSTEM_PROMPT` explicitly instructs the model: *"If same condition affects multiple anatomical sites, create SEPARATE entries."* and heavily delineates the differences between statuses.

### 2.2 Date Extraction
- **Issue**: The LLM struggled with translating relative dates ("since mid-December").
- **Solution**: The prompt was updated to explicitly prioritize stated dates, followed by note dates, and finally relative conversions. The `output_formatter.py` handles the deterministic translation of unstructured date fragments into unified `Month YYYY` or `DD Month YYYY` formats.

## 3. What Worked, What Didn't, and Why

### What Worked:
- **Taxonomy Embedding**: Providing the entire taxonomy (with descriptions and rules) in the prompt almost eliminated category mismatch errors natively in the LLM output.
- **Multi-level Taxonomy Fallback**: The `taxonomy_mapper.py` uses direct matching, then fuzzy string matching for subcategories, and finally an intelligent keyword map derived analytically. This serves as a safety net ensuring valid JSON schema outputs.

### What Didn't:
- **Requiring the LLM to strictly output valid JSON block**: Occasionally, the model would prepend or append conversational filler despite instructions. 
- **The Fix**: Built a robust `_parse_llm_json` string-stripping function that scans for `{` and `}` boundaries and unwraps varying levels of markdown code fences.

## 4. Instructions for Running the Code

### Requirements
Ensure your environment satisfies the dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Export the following required variables prior to execution:
```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="model-name"
```

### Execution
Run the pipeline via `main.py`. The required arguments track the problem statement directives.

```bash
python main.py \
  --data-dir ./dev \
  --patient-list ./dev_patients.json \
  --output-dir ./output \
  --concurrency 4
```

### CLI Arguments
- `--data-dir` (Required): Path to the input dataset directory containing `patient_XX` subdirectories.
- `--patient-list` (Required): Path to a JSON array file containing patient identifiers to process.
- `--output-dir` (Required): Path to the directory where JSON output per patient will be deposited.
- `--concurrency` (Optional): Number of parallel threads to use. Dramatically improves throughput. Default is 1.
- `--temperature` (Optional): Temperature scaling for the LLM. Default is 0.1 (recommended for deterministic extraction).
- `--verbose` (Optional): Enable verbose debug logging to stderr.

### Validation
To mathematically guarantee schema and taxonomy validity before evaluation, a standalone validator is included:
```bash
python validate_output.py --output-dir ./output --taxonomy taxonomy.json
```
