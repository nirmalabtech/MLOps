import pandas as pd

RAW_PATH = "data/bank_customer_churn.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the columns needed downstream (by prep.py / train.py) are present
expected_columns = [
    "CreditScore", "Geography", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Churn (Exited) distribution:")
print(df["Exited"].value_counts())
