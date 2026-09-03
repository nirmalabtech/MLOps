# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Load the dataset that was uploaded locally into data/
bank_dataset = pd.read_csv("data/bank_customer_churn.csv")
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Exited'

# List of numerical features in the dataset
numeric_features = [
    'CreditScore',       # Customer's credit score
    'Age',               # Customer's age
    'Tenure',            # Number of years the customer has been with the bank
    'Balance',           # Customer’s account balance
    'NumOfProducts',     # Number of products the customer has with the bank
    'HasCrCard',         # Whether the customer has a credit card (binary: 0 or 1)
    'IsActiveMember',    # Whether the customer is an active member (binary: 0 or 1)
    'EstimatedSalary'    # Customer’s estimated salary
]

# List of categorical features in the dataset
categorical_features = [
    'Geography',         # Country where the customer resides
]

# Define predictor matrix (X) using selected numeric and categorical features
X = bank_dataset[numeric_features + categorical_features]

# Define target variable
y = bank_dataset[target]

# Split dataset into train and test
# stratify=y keeps the (imbalanced) churn ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
