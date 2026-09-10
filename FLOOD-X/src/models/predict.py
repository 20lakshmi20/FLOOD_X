import pandas as pd
import joblib


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "models/flood_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_flood(input_data):

    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------
    # CREATE ENGINEERED FEATURES
    # --------------------------------------------------

    input_df["rainfall_intensity"] = (
        input_df["rainfall_1h_mm"] /
        (input_df["rainfall_24h_mm"] + 1)
    )

    input_df["river_rise_intensity"] = (
        input_df["river_rise_rate_m_per_hr"] *
        input_df["river_level_m"]
    )

    input_df["terrain_exposure"] = (
        input_df["slope_degrees"] /
        (input_df["distance_from_river_km"] + 0.1)
    )

    # --------------------------------------------------
    # SELECT MODEL FEATURES
    # --------------------------------------------------

    X = input_df[features]

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    probability = model.predict_proba(X)[0][1]

    prediction = model.predict(X)[0]

    # Convert probability to percentage
    probability_percent = probability * 100

    # --------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------

    risk_score = probability_percent

    # Risk level
    if risk_score < 25:
        risk_level = "LOW"

    elif risk_score < 50:
        risk_level = "MODERATE"

    elif risk_score < 75:
        risk_level = "HIGH"

    else:
        risk_level = "CRITICAL"

    return {
        "prediction": int(prediction),
        "flood_probability": round(probability_percent, 2),
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level
    }


# --------------------------------------------------
# TEST THE MODEL
# --------------------------------------------------

if __name__ == "__main__":

    sample_input = {
        "rainfall_1h_mm": 45,
        "rainfall_3h_mm": 90,
        "rainfall_6h_mm": 140,
        "rainfall_24h_mm": 220,
        "soil_moisture_percent": 78,
        "river_level_m": 4.2,
        "river_rise_rate_m_per_hr": 0.8,
        "elevation_m": 1800,
        "slope_degrees": 32,
        "distance_from_river_km": 0.5
    }

    result = predict_flood(sample_input)

    print("\n==============================")
    print("FLOOD-X PREDICTION")
    print("==============================")

    print("Flood Prediction:",
          "FLOOD" if result["prediction"] == 1
          else "NO FLOOD")

    print(
        f"Flood Probability: "
        f"{result['flood_probability']}%"
    )

    print(
        f"Risk Score: "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )