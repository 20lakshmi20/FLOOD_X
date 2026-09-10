import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/processed/flood_x_features.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    "models/flood_model.pkl"
)

features = joblib.load(
    "models/feature_columns.pkl"
)


# --------------------------------------------------
# PREDICT FLOOD PROBABILITY
# --------------------------------------------------

X = df[features]

df["flood_probability"] = (
    model.predict_proba(X)[:, 1] * 100
)


# --------------------------------------------------
# RISK LEVEL
# --------------------------------------------------

def get_risk_level(probability):

    if probability < 25:
        return "LOW"

    elif probability < 50:
        return "MODERATE"

    elif probability < 75:
        return "HIGH"

    else:
        return "CRITICAL"


df["risk_level"] = df[
    "flood_probability"
].apply(get_risk_level)


# --------------------------------------------------
# ALERT MESSAGE
# --------------------------------------------------

def create_alert(row):

    risk = row["risk_level"]
    probability = row["flood_probability"]

    location = row["location"]

    if risk == "CRITICAL":

        return {
            "alert_type": "CRITICAL",
            "message": (
                f"CRITICAL FLOOD RISK detected at "
                f"{location}. "
                f"Predicted probability: "
                f"{probability:.1f}%."
            ),
            "action": (
                "Immediate emergency response. "
                "Initiate evacuation procedures "
                "and monitor the area continuously."
            )
        }

    elif risk == "HIGH":

        return {
            "alert_type": "HIGH",
            "message": (
                f"HIGH FLOOD RISK detected at "
                f"{location}. "
                f"Predicted probability: "
                f"{probability:.1f}%."
            ),
            "action": (
                "Issue early warning, prepare "
                "evacuation resources and "
                "monitor rainfall and river levels."
            )
        }

    elif risk == "MODERATE":

        return {
            "alert_type": "MODERATE",
            "message": (
                f"MODERATE FLOOD RISK at "
                f"{location}. "
                f"Predicted probability: "
                f"{probability:.1f}%."
            ),
            "action": (
                "Increase monitoring and "
                "prepare emergency resources."
            )
        }

    else:

        return {
            "alert_type": "LOW",
            "message": (
                f"LOW FLOOD RISK at "
                f"{location}. "
                f"Predicted probability: "
                f"{probability:.1f}%."
            ),
            "action": (
                "Continue routine monitoring."
            )
        }


# --------------------------------------------------
# GENERATE ALERTS
# --------------------------------------------------

alerts = df.apply(
    create_alert,
    axis=1,
    result_type="expand"
)


df["alert_type"] = alerts[
    "alert_type"
]

df["alert_message"] = alerts[
    "message"
]

df["recommended_action"] = alerts[
    "action"
]


# --------------------------------------------------
# SELECT ALERT COLUMNS
# --------------------------------------------------

alert_columns = [
    "timestamp",
    "location",
    "latitude",
    "longitude",
    "flood_probability",
    "risk_level",
    "alert_type",
    "alert_message",
    "recommended_action"
]


alert_data = df[
    alert_columns
]


# --------------------------------------------------
# SAVE ALERT DATA
# --------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

alert_data.to_csv(
    "models/flood_alerts.csv",
    index=False
)


# --------------------------------------------------
# ALERT SUMMARY
# --------------------------------------------------

print("\n==============================")
print("FLOOD-X ALERT ENGINE")
print("==============================")


print("\nAlert Distribution:")

print(
    alert_data["alert_type"]
    .value_counts()
)


# --------------------------------------------------
# SHOW HIGH / CRITICAL ALERTS
# --------------------------------------------------

priority_alerts = alert_data[
    alert_data["alert_type"].isin(
        ["HIGH", "CRITICAL"]
    )
]


priority_alerts = priority_alerts.sort_values(
    by="flood_probability",
    ascending=False
)


print("\n==============================")
print("PRIORITY ALERTS")
print("==============================")


print(
    priority_alerts[
        [
            "location",
            "flood_probability",
            "risk_level",
            "alert_message",
            "recommended_action"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nAlert engine completed!")

print(
    "\nSaved to:"
    "\nmodels/flood_alerts.csv"
)