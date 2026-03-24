from preprocess import load_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.pipeline import Pipeline
import os


def train_model():
    df = load_df()

    # Select independent and dependent variables
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Split into 80% train and 20% temp, then split temp into validation and test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Create a pipeline with standard scaler and logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val) # get predictions for validation set
    test_predictions = pipeline.predict(X_test) # get predictions for test set

    print(f"Validation accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.4f}")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "churn_model.joblib")
    joblib.dump(pipeline, model_path)

    print("Model trained successfully.")
    print(f"Model saved to: {model_path}")
    return pipeline


if __name__ == "__main__":
    train_model()



