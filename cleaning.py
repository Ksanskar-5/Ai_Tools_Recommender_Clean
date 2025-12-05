import pandas as pd

# Load the CSV file
file_path_csv = "EDI-3_DATABASE - Database.csv"   # change to your file path
df = pd.read_csv(file_path_csv)

# 1. Strip spaces and lowercase categories
text_cols = ["Category", "Subcategory", "Input Type", "Output Type", 
             "Ease of Use", "Integration", "Languages", "Training Domain", 
             "Company", "Name"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# 2. Normalize "Cost"
def normalize_cost(x):
    x = str(x).lower()
    if "free" in x and "premium" not in x:
        return "free"
    elif "freemium" in x:
        return "freemium"
    elif "open" in x:
        return "open-source"
    elif any(c.isdigit() for c in x):
        return "paid"
    else:
        return "unknown"

if "Cost" in df.columns:
    df["Cost"] = df["Cost"].apply(normalize_cost)

# 3. Normalize Ease of Use
ease_map = {"beginner": 1, "intermediate": 2, "developer": 3, "enterprise": 4}
if "Ease of Use" in df.columns:
    df["Ease of Use"] = df["Ease of Use"].map(ease_map).fillna(0)

# 4. Normalize Rating (scale 0–1)
if "Rating" in df.columns:
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Rating"] = df["Rating"].apply(lambda x: x/5 if pd.notnull(x) and x <= 5 else x)

# 5. Normalize Popularity
pop_map = {"low": 0.3, "medium": 0.6, "high": 1}
if "Popularity" in df.columns:
    df["Popularity"] = df["Popularity"].map(pop_map).fillna(0.5)

# 6. Normalize Accuracy (% → 0–1)
if "Accuracy" in df.columns:
    df["Accuracy"] = df["Accuracy"].astype(str).str.replace("%", "").str.strip()
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    df["Accuracy"] = df["Accuracy"].apply(lambda x: x/100 if pd.notnull(x) and x > 1 else x)

# 7. Normalize Speed
speed_map = {"fast": 1, "medium": 0.5, "slow": 0.2}
if "Speed" in df.columns:
    df["Speed"] = df["Speed"].map(speed_map).fillna(0.5)

# 8. Fill missing values
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].fillna("unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# Save cleaned dataset
df.to_csv("ai_database_clean.csv", index=False)
print("Cleaned dataset saved as ai_database_clean.csv")
