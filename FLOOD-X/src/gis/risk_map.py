import pandas as pd
import folium
from folium.plugins import MarkerCluster


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/flood_x_features.csv"
)

# Load trained model
import joblib

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
# CREATE RISK LEVEL
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
# MAP CENTER
# --------------------------------------------------

center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()


flood_map = folium.Map(
    location=[
        center_lat,
        center_lon
    ],
    zoom_start=8,
    tiles="OpenStreetMap"
)


# --------------------------------------------------
# MARKER CLUSTER
# --------------------------------------------------

marker_cluster = MarkerCluster().add_to(
    flood_map
)


# --------------------------------------------------
# ADD LOCATIONS
# --------------------------------------------------

for _, row in df.iterrows():

    probability = row[
        "flood_probability"
    ]

    risk_level = row[
        "risk_level"
    ]

    # Marker color
    if risk_level == "LOW":
        color = "green"

    elif risk_level == "MODERATE":
        color = "orange"

    elif risk_level == "HIGH":
        color = "red"

    else:
        color = "darkred"


    # Popup information
    popup_html = f"""
    <b>Location:</b> {row['location']}<br>
    <b>Flood Probability:</b> {probability:.2f}%<br>
    <b>Risk Level:</b> {risk_level}<br>
    <b>Rainfall (24h):</b> {row['rainfall_24h_mm']:.2f} mm<br>
    <b>River Level:</b> {row['river_level_m']:.2f} m<br>
    <b>River Rise Rate:</b>
    {row['river_rise_rate_m_per_hr']:.2f} m/hr<br>
    <b>Population:</b> {int(row['population'])}
    """


    folium.Marker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        popup=folium.Popup(
            popup_html,
            max_width=300
        ),
        tooltip=(
            f"{row['location']} | "
            f"{risk_level} | "
            f"{probability:.1f}%"
        ),
        icon=folium.Icon(
            color=color,
            icon="warning-sign"
        )
    ).add_to(marker_cluster)


# --------------------------------------------------
# LEGEND
# --------------------------------------------------

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 180px;
z-index: 9999;
background-color: white;
border: 2px solid grey;
padding: 10px;
font-size: 14px;
">

<b>FLOOD-X Risk Level</b><br><br>

<span style="color:green;">●</span>
Low<br>

<span style="color:orange;">●</span>
Moderate<br>

<span style="color:red;">●</span>
High<br>

<span style="color:darkred;">●</span>
Critical

</div>
"""

flood_map.get_root().html.add_child(
    folium.Element(legend_html)
)


# --------------------------------------------------
# SAVE MAP
# --------------------------------------------------

output_path = "models/flood_risk_map.html"

flood_map.save(
    output_path
)


print("\n==============================")
print("FLOOD-X RISK MAP")
print("==============================")

print(
    f"Locations mapped: {len(df)}"
)

print(
    "\nRisk distribution:"
)

print(
    df["risk_level"].value_counts()
)

print(
    f"\nMap saved to:"
    f"\n{output_path}"
)