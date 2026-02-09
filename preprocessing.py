import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    """
    Preprocess Telco Customer Churn dataset.

    Steps:
    - Remove identifier column
    - Convert target to binary
    - Handle missing values
    - Standardize numerical features
    - One-hot encode categorical features

    Same preprocessing is applied to all models for fair comparison.
    """

    df = df.drop(columns=["customerID"])

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return X, y, preprocessor