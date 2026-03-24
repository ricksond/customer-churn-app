import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "churn_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train.py first to create it."
        )
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    model = load_model()
    input_df = pd.DataFrame([customer_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": float(probability),
    }


if __name__ == "__main__":
    sample_customer = {
        "tenure": 12,
        "monthly_charges": 70.5,
        "total_charges": 846.0,
        "senior_citizen": 0,
    }

    result = predict_churn(sample_customer)
    print(result)
