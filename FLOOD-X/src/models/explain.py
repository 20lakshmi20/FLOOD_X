import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "models/flood_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/flood_x_features.csv"
)


# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

FEATURES = [
    "rainfall_1h_mm",
    "rainfall_3h_mm",
    "rainfall_6h_mm",
    "rainfall_24h_mm",
    "soil_moisture_percent",
    "river_level_m",
    "river_rise_rate_m_per_hr",
    "elevation_m",
    "slope_degrees",
    "distance_from_river_km",
    "rainfall_intensity",
    "river_rise_intensity",
    "terrain_exposure"
]


# --------------------------------------------------
# SELECT SAMPLE FOR SHAP
# --------------------------------------------------

# Instead of calculating SHAP for all 10,000 rows,
# use 500 representative samples for faster execution.

X = df[FEATURES].sample(
    n=500,
    random_state=42
).reset_index(drop=True)

print("SHAP samples:", len(X))


# --------------------------------------------------
# CREATE SHAP EXPLAINER
# --------------------------------------------------

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X)

print("SHAP calculation completed!")


# --------------------------------------------------
# HANDLE SHAP OUTPUT
# --------------------------------------------------

# Different SHAP versions can return different formats.
# We want the SHAP values for the FLOOD class.

if isinstance(shap_values, list):

    flood_shap_values = shap_values[1]

else:

    flood_shap_values = shap_values

    if flood_shap_values.ndim == 3:
        flood_shap_values = flood_shap_values[:, :, 1]


# --------------------------------------------------
# FEATURE IMPORTANCE USING SHAP
# --------------------------------------------------

mean_abs_shap = np.abs(
    flood_shap_values
).mean(axis=0)


shap_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
})


shap_importance = shap_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)


print("\n==============================")
print("SHAP FEATURE IMPORTANCE")
print("==============================")


print(shap_importance.to_string(index=False))


# --------------------------------------------------
# SAVE SHAP IMPORTANCE
# --------------------------------------------------

shap_importance.to_csv(
    "models/shap_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# SHAP SUMMARY PLOT
# --------------------------------------------------

print("\nCreating SHAP summary plot...")


plt.figure(figsize=(10, 7))


shap.summary_plot(
    flood_shap_values,
    X,
    show=False
)


plt.title(
    "FLOOD-X Explainable AI - SHAP Summary"
)


plt.tight_layout()


plt.savefig(
    "models/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# --------------------------------------------------
# SINGLE SAMPLE EXPLANATION
# --------------------------------------------------

sample_index = 0


sample = X.iloc[
    sample_index:sample_index + 1
]


sample_shap = flood_shap_values[
    sample_index
]


explanation = pd.DataFrame({
    "feature": FEATURES,
    "value": sample.iloc[0].values,
    "shap_value": sample_shap
})


# Positive SHAP = increases flood prediction
# Negative SHAP = decreases flood prediction

explanation["impact"] = np.where(
    explanation["shap_value"] > 0,
    "Increases flood risk",
    "Decreases flood risk"
)


explanation["absolute_impact"] = np.abs(
    explanation["shap_value"]
)


explanation = explanation.sort_values(
    by="absolute_impact",
    ascending=False
)


explanation = explanation.drop(
    columns=["absolute_impact"]
)


print("\n==============================")
print("SAMPLE EXPLANATION")
print("==============================")


print(
    explanation.to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE SAMPLE EXPLANATION
# --------------------------------------------------

explanation.to_csv(
    "models/sample_explanation.csv",
    index=False
)


# --------------------------------------------------
# COMPLETED
# --------------------------------------------------

print("\n==============================")
print("SHAP ANALYSIS COMPLETED")
print("==============================")


print("\nFiles created:")

print("models/shap_summary.png")

print("models/shap_feature_importance.csv")

print("models/sample_explanation.csv")