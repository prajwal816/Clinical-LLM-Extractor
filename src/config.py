"""Configuration and taxonomy loading for Clinical LLM Extractor."""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-based LLM settings
# ---------------------------------------------------------------------------
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.json"


def load_taxonomy(path: Path | None = None) -> dict:
    """Load and return the full taxonomy dict from taxonomy.json."""
    path = path or TAXONOMY_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_category_subcategory_pairs(taxonomy: dict) -> set[tuple[str, str]]:
    """Return every valid (category, subcategory) pair."""
    pairs: set[tuple[str, str]] = set()
    for cat_key, cat_val in taxonomy.get("condition_categories", {}).items():
        for sub_key in cat_val.get("subcategories", {}):
            pairs.add((cat_key, sub_key))
    return pairs


def get_status_values(taxonomy: dict) -> list[str]:
    """Return all valid status keys."""
    return list(taxonomy.get("status_values", {}).keys())


def build_taxonomy_prompt_section(taxonomy: dict) -> str:
    """Build a concise taxonomy reference for inclusion in LLM prompts."""
    lines = ["## Taxonomy Categories & Subcategories\n"]
    for cat_key, cat_val in taxonomy["condition_categories"].items():
        lines.append(f"### {cat_key}: {cat_val['description']}")
        for sub_key, sub_desc in cat_val["subcategories"].items():
            lines.append(f"  - **{sub_key}**: {sub_desc}")
        lines.append("")

    lines.append("## Status Values\n")
    for status_key, status_val in taxonomy["status_values"].items():
        lines.append(f"- **{status_key}**: {status_val['description']}")
        lines.append(f"  Signals: {', '.join(status_val['signals'])}")
    lines.append("")

    lines.append("## Disambiguation Rules\n")
    for rule in taxonomy.get("disambiguation_rules", []):
        lines.append(f"- **{rule['rule']}**: {rule['explanation']}")
    lines.append("")

    lines.append("## Notes\n")
    for note in taxonomy.get("notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines)
