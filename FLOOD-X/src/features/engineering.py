import pandas as pd
import numpy as np
from pathlib import Path


def create_features(input_path, output_path):

    # Load cleaned dataset
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # --------------------------------------------------
    # FEATURE 1: Rainfall Intensity
    # --------------------------------------------------
    df["rainfall_intensity"] = (
        df["rainfall_1h_mm"] /
        (df["rainfall_24h_mm"] + 1)
    )

    # --------------------------------------------------
    # FEATURE 2: River Rise Intensity
    # --------------------------------------------------
    df["river_rise_intensity"] = (
        df["river_rise_rate_m_per_hr"] *
        df["river_level_m"]
    )

    # --------------------------------------------------
    # FEATURE 3: Terrain Exposure
    # --------------------------------------------------
    df["terrain_exposure"] = (
        df["slope_degrees"] /
        (df["distance_from_river_km"] + 0.1)
    )

    # --------------------------------------------------
    # FEATURE 4: Population Density per Household
    # --------------------------------------------------
    df["population_per_household"] = (
        df["population"] /
        (df["households"] + 1)
    )

    # --------------------------------------------------
    # FEATURE 5: Infrastructure Exposure
    # --------------------------------------------------
    df["infrastructure_exposure"] = (
        df["hospitals"] +
        df["schools"] +
        df["shelters"]
    )

    # --------------------------------------------------
    # Save feature-engineered dataset
    # --------------------------------------------------
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("New shape:", df.shape)
    print("\nNew features created:")

    new_features = [
        "rainfall_intensity",
        "river_rise_intensity",
        "terrain_exposure",
        "population_per_household",
        "infrastructure_exposure"
    ]

    print(new_features)

    print("\nFeature engineering completed!")
    print("Saved to:", output_path)


if __name__ == "__main__":

    create_features(
        "data/processed/flood_x_cleaned_dataset.csv",
        "data/processed/flood_x_features.csv"
    )