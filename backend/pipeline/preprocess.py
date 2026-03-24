import pandas as pd
import os


def load_df():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "../../data/churn_features.csv")

    # Normalize csv path
    csv_path = os.path.normpath(csv_path)

    df = pd.read_csv(csv_path)

    # ML Models need numeric values so change boolean values in "senior_citizen" and "churn"
    df["churn"] = df["churn"].apply(lambda x:1 if x == "t" else 0)
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    return df
