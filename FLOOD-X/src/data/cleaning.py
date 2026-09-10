import pandas as pd
from pathlib import Path


def clean_flood_data(input_path, output_path):

    # Load raw dataset
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Fix invalid soil moisture values
    df.loc[
        df["soil_moisture_percent"] > 100,
        "soil_moisture_percent"
    ] = pd.NA

    # Fix invalid negative slope values
    df.loc[
        df["slope_degrees"] < 0,
        "slope_degrees"
    ] = pd.NA

    # Numerical columns
    numerical_columns = [
        "latitude",
        "longitude",
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
        "population",
        "households",
        "hospitals",
        "schools",
        "shelters"
    ]

    # Fill missing numerical values
    # First use location-wise median
    for column in numerical_columns:

        df[column] = df.groupby("location")[column].transform(
            lambda x: x.fillna(x.median())
        )

        # Remaining missing values use global median
        df[column] = df[column].fillna(
            df[column].median()
        )

    # Fill missing categorical values
    for column in ["location", "severity"]:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    # Make flood target integer
    df["flood"] = df["flood"].astype(int)

    # Create output folder if needed
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_csv(
        output_path,
        index=False
    )

    print("Cleaned shape:", df.shape)
    print("Missing values remaining:", df.isnull().sum().sum())
    print("Cleaning completed!")
    print("Saved to:", output_path)


if __name__ == "__main__":

    clean_flood_data(
        "data/raw/flood_x_unprocessed_dataset.csv",
        "data/processed/flood_x_cleaned_dataset.csv"
    )