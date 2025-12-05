"""
Simple rule-based parser that matches query strings to category / input type columns.
Returns a list of lightweight records if matches are found.
"""

import os
import pandas as pd
from typing import List, Dict, Any

BASE_DATA_PATH = os.environ.get("AI_RECOMMENDER_DATA", "data")
DATA_PATH = os.path.join(BASE_DATA_PATH, "ai_database.csv")  # fallback path

def simple_parse(query: str) -> List[Dict[str, Any]]:
    query_lower = (query or "").lower()
    if not query_lower:
        return []

    # Try processed/clean csv first (consistent with vectorizer)
    proc = os.path.join(BASE_DATA_PATH, "processed", "ai_database_clean1.csv")
    if os.path.exists(proc):
        df = pd.read_csv(proc).fillna("")
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).fillna("")
    else:
        return []

    # check against Category column
    matches = df[df["Category"].str.lower().str.contains(query_lower, na=False)]
    if not matches.empty:
        return matches[["Name", "Company", "Category", "Subcategory",
                        "Input Type", "Output Type", "Task Description"]].to_dict(orient="records")

    # check against Input Type column
    matches = df[df["Input Type"].str.lower().str.contains(query_lower, na=False)]
    if not matches.empty:
        return matches[["Name", "Company", "Category", "Subcategory",
                        "Input Type", "Output Type", "Task Description"]].to_dict(orient="records")

    return []
