import pandas as pd
import joblib


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
# ESTIMATE POPULATION AT RISK
# --------------------------------------------------

# We estimate exposed population based on
# the model's flood probability.

df["population_at_risk"] = (
    df["population"] *
    df["flood_probability"] / 100
)


df["households_at_risk"] = (
    df["households"] *
    df["flood_probability"] / 100
)


# --------------------------------------------------
# INFRASTRUCTURE AT RISK
# --------------------------------------------------

df["hospitals_at_risk"] = (
    df["hospitals"] *
    df["flood_probability"] / 100
)


df["schools_at_risk"] = (
    df["schools"] *
    df["flood_probability"] / 100
)


# --------------------------------------------------
# SAVE ANALYSIS DATA
# --------------------------------------------------

output_columns = [
    "timestamp",
    "location",
    "latitude",
    "longitude",
    "flood_probability",
    "risk_level",
    "population",
    "population_at_risk",
    "households",
    "households_at_risk",
    "hospitals",
    "hospitals_at_risk",
    "schools",
    "schools_at_risk",
    "shelters"
]


population_risk = df[
    output_columns
]


population_risk.to_csv(
    "models/population_risk.csv",
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

total_population = df[
    "population_at_risk"
].sum()


total_households = df[
    "households_at_risk"
].sum()


total_hospitals = df[
    "hospitals_at_risk"
].sum()


total_schools = df[
    "schools_at_risk"
].sum()


# --------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------

risk_population = (
    df.groupby("risk_level")[
        "population_at_risk"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("FLOOD-X POPULATION AT RISK")
print("==============================")


print(
    f"\nEstimated Population at Risk:"
    f" {total_population:,.0f}"
)


print(
    f"Estimated Households at Risk:"
    f" {total_households:,.0f}"
)


print(
    f"Estimated Hospitals at Risk:"
    f" {total_hospitals:,.0f}"
)


print(
    f"Estimated Schools at Risk:"
    f" {total_schools:,.0f}"
)


print("\nPopulation Risk Distribution:")

print(
    risk_population
)


print("\nRisk Levels:")

print(
    df["risk_level"].value_counts()
)


print("\n==============================")
print("TOP HIGH-RISK LOCATIONS")
print("==============================")


top_locations = (
    df.sort_values(
        "population_at_risk",
        ascending=False
    )
    [
        [
            "location",
            "flood_probability",
            "risk_level",
            "population_at_risk"
        ]
    ]
    .head(10)
)


print(
    top_locations.to_string(
        index=False
    )
)


print("\nPopulation risk analysis completed!")

print(
    "\nSaved to:"
    "\nmodels/population_risk.csv"
)