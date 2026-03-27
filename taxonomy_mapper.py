"""Taxonomy validation and mapping for extracted conditions."""

import logging
import re

from config import get_valid_category_subcategory_pairs, get_status_values

logger = logging.getLogger(__name__)


def validate_and_fix_conditions(conditions: list[dict], taxonomy: dict) -> list[dict]:
    """
    Validate and fix taxonomy compliance for all conditions.
    
    - Validates category/subcategory pairs against taxonomy
    - Validates status values
    - Fixes common issues (wrong subcategory, typos)
    - Removes conditions that can't be mapped
    
    Returns:
        List of validated conditions.
    """
    valid_pairs = get_valid_category_subcategory_pairs(taxonomy)
    valid_statuses = set(get_status_values(taxonomy))
    categories = taxonomy.get("condition_categories", {})
    
    validated = []
    for cond in conditions:
        cat = cond.get("category", "")
        sub = cond.get("subcategory", "")
        status = cond.get("status", "")
        name = cond.get("condition_name", "")
        
        # -- Fix status --
        if status not in valid_statuses:
            status_lower = status.lower().strip()
            if status_lower in valid_statuses:
                cond["status"] = status_lower
            elif "suspect" in status_lower or "possible" in status_lower:
                cond["status"] = "suspected"
            elif "resolv" in status_lower or "history" in status_lower or "remission" in status_lower:
                cond["status"] = "resolved"
            else:
                cond["status"] = "active"
            logger.debug(f"Fixed status '{status}' -> '{cond['status']}' for '{name}'")
        
        # -- Validate category/subcategory --
        if (cat, sub) in valid_pairs:
            validated.append(cond)
            continue
        
        # Try to fix: correct category exists, wrong subcategory
        if cat in categories:
            valid_subs = list(categories[cat]["subcategories"].keys())
            # Try fuzzy match on subcategory
            match = _fuzzy_match_subcategory(sub, valid_subs)
            if match:
                logger.debug(f"Fixed subcategory '{sub}' -> '{match}' for '{name}'")
                cond["subcategory"] = match
                validated.append(cond)
                continue
        
        # Try to find correct category based on subcategory
        for c_key, c_val in categories.items():
            if sub in c_val.get("subcategories", {}):
                logger.debug(f"Fixed category '{cat}' -> '{c_key}' for '{name}'")
                cond["category"] = c_key
                validated.append(cond)
                break
        else:
            # Try intelligent re-mapping based on condition name
            mapped = _remap_by_name(name, categories)
            if mapped:
                cond["category"] = mapped[0]
                cond["subcategory"] = mapped[1]
                logger.debug(f"Remapped '{name}' -> {mapped}")
                validated.append(cond)
            else:
                logger.warning(
                    f"Dropping condition '{name}' — invalid pair ({cat}, {sub}) "
                    f"and couldn't remap"
                )
    
    return validated


def _fuzzy_match_subcategory(sub: str, valid_subs: list[str]) -> str | None:
    """Try to fuzzy-match a subcategory string."""
    sub_lower = sub.lower().strip().replace(" ", "_").replace("-", "_")
    for vs in valid_subs:
        if sub_lower == vs or sub_lower in vs or vs in sub_lower:
            return vs
    return None


def _remap_by_name(name: str, categories: dict) -> tuple[str, str] | None:
    """Try to map a condition name to a category/subcategory based on keywords."""
    name_lower = name.lower()
    
    keyword_map = {
        # Cancer
        ("carcinoma", "cancer", "malignant", "malignancy", "tumor", "neoplasm", "sarcoma"): {
            "default": ("cancer", "primary_malignancy"),
            "metasta": ("cancer", "metastasis"),
            "benign": ("cancer", "benign"),
            "dysplasia": ("cancer", "pre_malignant"),
        },
        # Cardiovascular
        ("hypertension", "blood pressure"): {"default": ("cardiovascular", "hypertensive")},
        ("atrial fibrillation", "arrhythmia", "tachycardia", "bradycardia"): {"default": ("cardiovascular", "rhythm")},
        ("dissection", "aneurysm", "thrombosis", "atherosclerosis", "sclerosis", "peripheral artery", "varicose"): 
            {"default": ("cardiovascular", "vascular")},
        ("coronary", "myocardial infarction", "angina"): {"default": ("cardiovascular", "coronary")},
        # Infectious
        ("pneumonia", "sepsis", "infection", "abscess"): {"default": ("infectious", "bacterial")},
        ("covid", "hepatitis b", "hepatitis c", "hiv"): {"default": ("infectious", "viral")},
        ("aspergill", "candid", "fungal"): {"default": ("infectious", "fungal")},
        # Metabolic
        ("diabetes", "diabetic"): {"default": ("metabolic_endocrine", "diabetes")},
        ("hypothyroid", "hyperthyroid", "thyroid"): {"default": ("metabolic_endocrine", "thyroid")},
        ("cholesterol", "lipid", "hyperlipid"): {"default": ("metabolic_endocrine", "lipid")},
        # Neurological
        ("stroke", "infarct", "hemorrhage", "cerebrovascular"): {"default": ("neurological", "cerebrovascular")},
        ("seizure", "epilepsy"): {"default": ("neurological", "seizure")},
        ("parkinson", "alzheimer", "dementia", "sclerosis"): {"default": ("neurological", "degenerative")},
        ("aphasia", "dysarthria", "cognitive", "palsy"): {"default": ("neurological", "functional")},
        ("traumatic brain", "skull fracture"): {"default": ("neurological", "traumatic")},
        # Pulmonary
        ("copd", "asthma", "emphysema"): {"default": ("pulmonary", "obstructive")},
        ("ards", "respiratory failure"): {"default": ("pulmonary", "acute_respiratory")},
        ("pleural effusion", "pneumothorax", "fibrosis"): {"default": ("pulmonary", "structural")},
        # GI
        ("liver", "hepat", "cirrho"): {"default": ("gastrointestinal", "hepatic")},
        ("gallbladder", "cholecyst", "biliary"): {"default": ("gastrointestinal", "biliary")},
        # Renal
        ("kidney", "renal failure", "nephro"): {"default": ("renal", "renal_failure")},
        ("nephrolithiasis", "kidney stone", "hydronephrosis"): {"default": ("renal", "structural")},
        # Hematological
        ("anemia", "pancytopenia", "thrombocytopenia", "neutropenia", "lymphopenia"): 
            {"default": ("hematological", "cytopenia")},
        ("coagul", "hemophilia", "bleeding"): {"default": ("hematological", "coagulation")},
        # Immunological
        ("immunodeficiency", "igg deficiency"): {"default": ("immunological", "immunodeficiency")},
        ("allergy", "allergic", "hay fever"): {"default": ("immunological", "allergic")},
        ("autoimmune", "rheumatoid", "lupus"): {"default": ("immunological", "autoimmune")},
        # Musculoskeletal
        ("fracture",): {"default": ("musculoskeletal", "fracture")},
        ("osteopor", "osteoarthr", "degenerative"): {"default": ("musculoskeletal", "degenerative")},
        ("gout",): {"default": ("musculoskeletal", "crystal_arthropathy")},
    }
    
    for keywords, mappings in keyword_map.items():
        for kw in keywords:
            if kw in name_lower:
                # Check for specific sub-mappings
                for sub_kw, mapping in mappings.items():
                    if sub_kw != "default" and sub_kw in name_lower:
                        return mapping
                return mappings["default"]
    
    return None
