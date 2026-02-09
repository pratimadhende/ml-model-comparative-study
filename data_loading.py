import pandas as pd

def load_dataset(path):
    """
    Loads dataset from CSV.

    Keeping data loading separate improves reproducibility
    and keeps the experimental pipeline clean.
    """
    df = pd.read_csv(path)
    return df