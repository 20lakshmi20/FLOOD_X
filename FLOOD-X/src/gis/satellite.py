import pandas as pd
import folium
from pathlib import Path


# ==================================================
# FLOOD-X SATELLITE / GLOF RISK MODULE
# ==================================================

# IMPORTANT:
# This is a prototype using lake-monitoring data.
# In the final system, these values can come from
# satellite-derived measurements such as Sentinel-2
# or other remote-sensing products.


# --------------------------------------------------
# SAMPLE GLACIAL LAKE DATA
# --------------------------------------------------

lake_data = pd.DataFrame({

    "lake_id": [
        "GL001",
        "GL002",
        "GL003",
        "GL004",
        "GL005"
    ],

    "lake_name": [
        "Lake Alpha",
        "Lake Beta",
        "Lake Gamma",
        "Lake Delta",
        "Lake Epsilon"
    ],

    "latitude": [
        30.250,
        30.420,
        30.180,
        30.520,
        30.350
    ],

    "longitude": [
        79.100,
        79.250,
        78.950,
        79.400,
        79.000
    ],

    # Previous observed lake area
    "previous_area_km2": [
        0.82,
        1.20,
        0.55,
        0.95,
        1.40
    ],

    # Current observed lake area
    "current_area_km2": [
        0.91,
        1.32,
        0.57,
        1.12,
        1.43
    ],

    # Approximate elevation
    "elevation_m": [
        4700,
        4850,
        4550,
        5100,
        4900
    ]
})


# --------------------------------------------------
# CALCULATE LAKE AREA CHANGE
# --------------------------------------------------

lake_data["area_change_percent"] = (

    (
        lake_data["current_area_km2"]
        -
        lake_data["previous_area_km2"]
    )
    /
    lake_data["previous_area_km2"]
    * 100

)


# --------------------------------------------------
# GLOF RISK SCORE
# --------------------------------------------------

def calculate_glof_risk(area_change):

    if area_change >= 20:
        return 85

    elif area_change >= 10:
        return 65

    elif area_change >= 5:
        return 40

    else:
        return 20


lake_data["glof_risk_score"] = (
    lake_data["area_change_percent"]
    .apply(calculate_glof_risk)
)


# --------------------------------------------------
# GLOF RISK LEVEL
# --------------------------------------------------

def get_glof_level(score):

    if score < 25:
        return "LOW"

    elif score < 50:
        return "MODERATE"

    elif score < 75:
        return "HIGH"

    else:
        return "CRITICAL"


lake_data["glof_risk_level"] = (
    lake_data["glof_risk_score"]
    .apply(get_glof_level)
)


# --------------------------------------------------
# IDENTIFY CHANGING LAKES
# --------------------------------------------------

lake_data["significant_change"] = (

    lake_data["area_change_percent"] >= 10

)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==============================")
print("FLOOD-X SATELLITE / GLOF ANALYSIS")
print("==============================")


print("\nGlacial Lake Monitoring:")

print(
    lake_data[
        [
            "lake_id",
            "lake_name",
            "previous_area_km2",
            "current_area_km2",
            "area_change_percent",
            "glof_risk_score",
            "glof_risk_level"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# --------------------------------------------------
# HIGH-RISK LAKES
# --------------------------------------------------

high_risk_lakes = lake_data[
    lake_data["glof_risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )
]


print("\n==============================")
print("PRIORITY GLOF MONITORING")
print("==============================")


if len(high_risk_lakes) > 0:

    print(
        high_risk_lakes[
            [
                "lake_name",
                "area_change_percent",
                "glof_risk_score",
                "glof_risk_level"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

else:

    print("No high-risk lakes detected.")


# --------------------------------------------------
# CREATE SATELLITE MONITORING MAP
# --------------------------------------------------

center_lat = lake_data[
    "latitude"
].mean()

center_lon = lake_data[
    "longitude"
].mean()


satellite_map = folium.Map(

    location=[
        center_lat,
        center_lon
    ],

    zoom_start=9,

    tiles="OpenStreetMap"
)


# --------------------------------------------------
# ADD LAKES TO MAP
# --------------------------------------------------

for _, row in lake_data.iterrows():

    risk = row[
        "glof_risk_level"
    ]

    score = row[
        "glof_risk_score"
    ]

    change = row[
        "area_change_percent"
    ]


    if risk == "LOW":

        color = "green"

    elif risk == "MODERATE":

        color = "orange"

    elif risk == "HIGH":

        color = "red"

    else:

        color = "darkred"


    popup_html = f"""

    <b>{row['lake_name']}</b><br><br>

    Lake ID:
    {row['lake_id']}<br>

    Previous Area:
    {row['previous_area_km2']:.2f} km²<br>

    Current Area:
    {row['current_area_km2']:.2f} km²<br>

    Area Change:
    {change:.2f}%<br>

    Elevation:
    {row['elevation_m']} m<br>

    GLOF Risk Score:
    {score}/100<br>

    GLOF Risk Level:
    {risk}

    """


    folium.CircleMarker(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=9,

        color=color,

        fill=True,

        fill_opacity=0.7,

        popup=folium.Popup(
            popup_html,
            max_width=300
        ),

        tooltip=(
            f"{row['lake_name']} | "
            f"{risk}"
        )

    ).add_to(
        satellite_map
    )


# --------------------------------------------------
# SAVE MAP
# --------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True
)


output_path = (
    "models/glof_monitoring_map.html"
)


satellite_map.save(
    output_path
)


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

lake_data.to_csv(
    "models/glof_lake_analysis.csv",
    index=False
)


print("\n==============================")
print("SATELLITE ANALYSIS COMPLETED")
print("==============================")


print(
    "\nMap saved to:"
    "\nmodels/glof_monitoring_map.html"
)


print(
    "\nData saved to:"
    "\nmodels/glof_lake_analysis.csv"
)