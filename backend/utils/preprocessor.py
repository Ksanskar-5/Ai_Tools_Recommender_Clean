import pandas as pd

# Load dataset
DATA_PATH = "data/processed/ai_database_clean1.csv"
df = pd.read_csv(DATA_PATH)

def simple_parse(query: str):
    """
    Rule-based: try exact keyword match in known categories
    """
    query_lower = query.lower()

    # check against Category column
    matches = df[df["Category"].str.lower().str.contains(query_lower, na=False)]
    if not matches.empty:
        return matches[["Name", "Company", "Category", "Subcategory"]].to_dict(orient="records")

    # check against Input Type column
    matches = df[df["Input Type"].str.lower().str.contains(query_lower, na=False)]
    if not matches.empty:
        return matches[["Name", "Company", "Category", "Subcategory"]].to_dict(orient="records")

    return None
