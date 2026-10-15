from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
THRESHOLD = 0.5


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_customer(customer):
    model = load_model()
    columns = list(model.feature_names_in_)

    missing = [c for c in columns if c not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    data = pd.DataFrame([customer])[columns]
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce").fillna(0)

    probability = float(model.predict_proba(data)[0, 1])
    return {
        "will_churn": probability >= THRESHOLD,
        "churn_probability": probability,
    }


if __name__ == "__main__":
    risky_customer = {
        "Gender": "Female", "Senior Citizen": "No", "Partner": "No",
        "Dependents": "No", "Tenure Months": 2, "Phone Service": "Yes",
        "Multiple Lines": "No", "Internet Service": "Fiber optic",
        "Online Security": "No", "Online Backup": "No",
        "Device Protection": "No", "Tech Support": "No",
        "Streaming TV": "No", "Streaming Movies": "No",
        "Contract": "Month-to-month", "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.0, "Total Charges": 170.0,
    }

    loyal_customer = {
        "Gender": "Male", "Senior Citizen": "No", "Partner": "Yes",
        "Dependents": "Yes", "Tenure Months": 60, "Phone Service": "Yes",
        "Multiple Lines": "No", "Internet Service": "DSL",
        "Online Security": "Yes", "Online Backup": "Yes",
        "Device Protection": "Yes", "Tech Support": "Yes",
        "Streaming TV": "No", "Streaming Movies": "No",
        "Contract": "Two year", "Paperless Billing": "No",
        "Payment Method": "Credit card (automatic)",
        "Monthly Charges": 65.0, "Total Charges": 3900.0,
    }

    for name, customer in [("Risky", risky_customer), ("Loyal", loyal_customer)]:
        result = predict_customer(customer)
        print(name, result)