import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
import streamlit.components.v1 as components
from pathlib import Path

# ============================================================
# FLOOD-X | AI Flash Flood Prediction & Risk Intelligence
# ============================================================

st.set_page_config(
    page_title="FLOOD-X | Disaster Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', Arial, sans-serif;
}

.stApp {
    background: #08111f;
    color: #e8f0fa;
}

[data-testid="stHeader"] {
    background: #08111f;
}

[data-testid="stSidebar"] {
    background: #06101d;
    border-right: 1px solid #1d334d;
}

[data-testid="stSidebar"] * {
    color: #e6eef8 !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 7px;
    border-radius: 8px;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #f4f8ff;
    margin-bottom: 2px;
}

.subtitle {
    color: #9fb1c8;
    font-size: 14px;
    margin-bottom: 22px;
}

.hero {
    background: linear-gradient(135deg, #0b1d34 0%, #123b63 55%, #075b70 100%);
    padding: 28px 30px;
    border: 1px solid #285477;
    border-radius: 18px;
    color: #f8fbff;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,.22);
}

.hero h1 {
    color: #ffffff;
    margin: 0;
    font-size: 30px;
}

.hero p {
    color: #c8d8ea;
    margin: 7px 0 0 0;
}

.kpi {
    background: #0e1b2d;
    border: 1px solid #213954;
    border-radius: 15px;
    padding: 18px;
    min-height: 120px;
    box-shadow: 0 5px 18px rgba(0,0,0,.18);
}

.kpi-label {
    color: #9fb1c8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .6px;
}

.kpi-value {
    color: #f5f9ff;
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}

.kpi-sub {
    color: #8298b2;
    font-size: 12px;
    margin-top: 4px;
}

.section {
    color: #f2f7fd;
    font-size: 20px;
    font-weight: 750;
    margin: 25px 0 12px;
}

.card {
    background: #0e1b2d;
    border: 1px solid #213954;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(0,0,0,.18);
}

.card h3 {
    color: #f2f7fd !important;
}

.card p, .card div, .card span {
    color: #c1cfdf;
}

.risk-critical {
    color: #fecaca;
    background: #49151b;
    border: 1px solid #8f2932;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.risk-high {
    color: #fed7aa;
    background: #4a2411;
    border: 1px solid #9a4b1e;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.risk-moderate {
    color: #fef08a;
    background: #40380b;
    border: 1px solid #827219;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.risk-low {
    color: #bbf7d0;
    background: #103b25;
    border: 1px solid #247047;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 700;
    display: inline-block;
}

.alert-box {
    background: #281d10;
    border: 1px solid #67421c;
    border-left: 5px solid #f59e0b;
    color: #f5dfb4;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 15px;
}

.info-box {
    background: #0d2239;
    border: 1px solid #24557c;
    border-left: 5px solid #38bdf8;
    color: #c9def0;
    border-radius: 12px;
    padding: 14px 16px;
}

.info-box b, .alert-box b {
    color: #ffffff;
}

.footer {
    text-align: center;
    color: #71869f;
    font-size: 12px;
    padding: 30px 0 15px;
}

[data-testid="stMetricValue"] {
    color: #f5f9ff !important;
}

[data-testid="stMetricLabel"] {
    color: #9fb1c8 !important;
}

.stMarkdown, .stText, p, label {
    color: #d5dfeb;
}

div[data-baseweb="select"] > div {
    background: #0e1b2d !important;
    border-color: #2a425e !important;
    color: #f2f7fd !important;
}

div[data-baseweb="select"] span {
    color: #f2f7fd !important;
}

input, textarea {
    background: #0e1b2d !important;
    color: #f2f7fd !important;
    border-color: #2a425e !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263f5a;
    border-radius: 12px;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0f766e);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #14b8a6);
    color: white;
}

.stSlider label, .stNumberInput label, .stTextInput label,
.stTextArea label, .stSelectbox label {
    color: #cbd8e7 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "flood_x_features.csv"
MODEL_PATH = ROOT / "models" / "flood_model.pkl"
FEATURE_PATH = ROOT / "models" / "feature_columns.pkl"
SHAP_PATH = ROOT / "models" / "shap_feature_importance.csv"
POP_PATH = ROOT / "models" / "population_risk.csv"
ALERT_PATH = ROOT / "models" / "flood_alerts.csv"

# -----------------------------
# Load data/model
# -----------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    if not FEATURE_PATH.exists():
        return None
    return joblib.load(FEATURE_PATH)

df = load_data()
model = load_model()
model_features = load_features()

# -----------------------------
# Utility functions
# -----------------------------
def risk_level(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"

def risk_class(level):
    return {
        "CRITICAL": "risk-critical",
        "HIGH": "risk-high",
        "MODERATE": "risk-moderate",
        "LOW": "risk-low",
    }.get(level, "risk-low")

def predict_probability(dataframe):
    if model is None:
        return np.zeros(len(dataframe))
    cols = list(model_features) if model_features is not None else [
        "rainfall_1h_mm", "rainfall_3h_mm", "rainfall_6h_mm",
        "rainfall_24h_mm", "soil_moisture_percent", "river_level_m",
        "river_rise_rate_m_per_hr", "elevation_m", "slope_degrees",
        "distance_from_river_km", "rainfall_intensity",
        "river_rise_intensity", "terrain_exposure"
    ]
    available = [c for c in cols if c in dataframe.columns]
    X = dataframe[available].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    return model.predict_proba(X)[:, 1] * 100

def make_chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#e8f0fa",
            size=13
        ),
        title_font=dict(
            family="Inter, Arial, sans-serif",
            color="#f8fbff",
            size=17
        ),
        plot_bgcolor="#0e1b2d",
        paper_bgcolor="#0e1b2d",
        margin=dict(l=30, r=25, t=55, b=35),
        legend=dict(
            bgcolor="#0e1b2d",
            bordercolor="#29415d",
            borderwidth=1,
            font=dict(color="#dce8f5")
        ),
        hoverlabel=dict(
            bgcolor="#14263d",
            font=dict(color="#ffffff")
        )
    )
    fig.update_xaxes(
        color="#b7c7d9",
        title_font=dict(color="#dce8f5"),
        tickfont=dict(color="#aebfd2"),
        showgrid=True,
        gridcolor="#24384f",
        zerolinecolor="#334b66"
    )
    fig.update_yaxes(
        color="#b7c7d9",
        title_font=dict(color="#dce8f5"),
        tickfont=dict(color="#aebfd2"),
        showgrid=True,
        gridcolor="#24384f",
        zerolinecolor="#334b66"
    )
    return fig

def risk_color(level):
    return {
        "CRITICAL": "#b91c1c",
        "HIGH": "#ea580c",
        "MODERATE": "#ca8a04",
        "LOW": "#16a34a",
    }.get(level, "#64748b")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 2px 20px;">
        <div style="font-size:28px;font-weight:800;color:white;">🌊 FLOOD-X</div>
        <div style="font-size:12px;color:#93c5fd;">AI FLOOD INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Risk Map",
            "AI Prediction",
            "Risk Score",
            "Explainable AI",
            "Alerts",
            "Population at Risk",
            "Evacuation",
            "Satellite Analysis",
            "Citizen Reports",
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### System Status")

    if model is not None:
        st.success("AI Model Loaded")
    else:
        st.error("AI Model Missing")

    if not df.empty:
        st.success("Dataset Loaded")
        st.caption(f"{len(df):,} records available")
    else:
        st.error("Dataset Missing")

    st.markdown("---")
    st.caption("FLOOD-X Prototype")
    st.caption("Disaster Management • SIH 2026")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">FLOOD-X Disaster Intelligence</div>'
    '<div class="subtitle">AI-powered flash flood risk prediction for hilly and Himalayan regions</div>',
    unsafe_allow_html=True
)

if df.empty:
    st.error("Dataset not found. Expected: data/processed/flood_x_features.csv")
    st.stop()

# Calculate probabilities
if "flood_probability" not in df.columns:
    df["flood_probability"] = predict_probability(df)

df["risk_score_ai"] = df["flood_probability"]
df["risk_level_ai"] = df["risk_score_ai"].apply(risk_level)

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":

    latest = df.sort_values("timestamp").iloc[-1] if "timestamp" in df.columns else df.iloc[-1]

    avg_prob = float(df["flood_probability"].mean())
    high_count = int((df["risk_score_ai"] >= 50).sum())
    critical_count = int((df["risk_score_ai"] >= 75).sum())
    max_prob = float(df["flood_probability"].max())

    st.markdown("""
    <div class="hero">
        <h1>Mountain Flood Risk Command Center</h1>
        <p>Monitor environmental signals, understand AI predictions, identify exposed areas and support early response decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Current Risk</div>'
            f'<div class="kpi-value">{latest["flood_probability"]:.1f}%</div>'
            f'<div class="kpi-sub">AI flood probability</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">High Risk Records</div>'
            f'<div class="kpi-value">{high_count:,}</div>'
            f'<div class="kpi-sub">Probability ≥ 50%</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Critical Records</div>'
            f'<div class="kpi-value">{critical_count:,}</div>'
            f'<div class="kpi-sub">Probability ≥ 75%</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Peak Probability</div>'
            f'<div class="kpi-value">{max_prob:.1f}%</div>'
            f'<div class="kpi-sub">Highest model output</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section">Current Situation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        level = latest["risk_level_ai"]
        st.markdown(
            f'<div class="card"><h3 style="color:#f2f7fd;">Risk Level</h3>'
            f'<div class="{risk_class(level)}">{level}</div>'
            f'<br><br><b>Location:</b> {latest.get("location","Unknown")}<br>'
            f'<b>Rainfall 24h:</b> {latest.get("rainfall_24h_mm",0):.1f} mm<br>'
            f'<b>River Level:</b> {latest.get("river_level_m",0):.2f} m<br>'
            f'<b>River Rise:</b> {latest.get("river_rise_rate_m_per_hr",0):.2f} m/hr</div>',
            unsafe_allow_html=True
        )

    with col2:
        trend = df.copy()
        if "timestamp" in trend.columns:
            trend = trend.sort_values("timestamp").tail(100)

            fig = px.line(
                trend,
                x="timestamp",
                y="flood_probability",
                markers=True,
                template="plotly_dark",
                labels={
                    "flood_probability": "Probability (%)",
                    "timestamp": "Time"
                }
            )
            fig.add_hline(y=50, line_dash="dash", annotation_text="High-risk threshold")
            fig.add_hline(y=75, line_dash="dot", annotation_text="Critical threshold")
            fig = make_chart_layout(fig, 370)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Why is the risk changing?</div>', unsafe_allow_html=True)

    factor_cols = [
        "rainfall_1h_mm",
        "rainfall_3h_mm",
        "rainfall_6h_mm",
        "rainfall_24h_mm",
        "soil_moisture_percent",
        "river_level_m",
        "river_rise_rate_m_per_hr",
        "slope_degrees",
    ]

    available_factors = [x for x in factor_cols if x in latest.index]
    factor_values = []
    for col in available_factors:
        value = pd.to_numeric(latest[col], errors="coerce")
        median = pd.to_numeric(df[col], errors="coerce").median()
        if pd.isna(value) or pd.isna(median) or median == 0:
            score = 0
        else:
            score = min(abs(float(value)) / abs(float(median)) * 50, 100)
        factor_values.append((col.replace("_", " ").title(), score))

    factors = pd.DataFrame(factor_values, columns=["Indicator", "Factor"])
    factors = factors.sort_values("Factor", ascending=True)

    fig = px.bar(
        factors,
        x="Factor",
        y="Indicator",
        orientation="h",
        template="plotly_dark",
        labels={"Factor": "Relative indicator score", "Indicator": ""}
    )
    fig = make_chart_layout(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="info-box"><b>Decision support:</b> FLOOD-X combines rainfall, soil moisture, river behaviour and terrain indicators. The displayed outputs are prototype AI predictions and should be validated against operational sensor and emergency-management data before deployment.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# RISK MAP
# ============================================================
elif page == "Risk Map":

    st.markdown('<div class="section">Regional Flood Risk Map</div>', unsafe_allow_html=True)

    locations = sorted(df["location"].dropna().unique().tolist())
    selected = st.selectbox("Select monitored location", ["All locations"] + locations)

    map_df = df.copy()
    if selected != "All locations":
        map_df = map_df[map_df["location"] == selected]

    map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()

    if map_df.empty:
        st.warning("No geographic records available.")
    else:
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles="CartoDB positron",
            control_scale=True
        )

        heat_data = [
            [row.latitude, row.longitude, max(row.flood_probability / 100, 0.01)]
            for row in map_df.itertuples()
        ]

        HeatMap(
            heat_data,
            radius=24,
            blur=18,
            min_opacity=0.25,
            max_zoom=10
        ).add_to(m)

        cluster = MarkerCluster(name="Risk locations").add_to(m)

        latest_by_location = (
            map_df.sort_values("timestamp")
            .groupby("location", as_index=False)
            .tail(1)
        )

        for row in latest_by_location.itertuples():
            level = risk_level(row.flood_probability)
            color = risk_color(level)

            popup = f"""
            <b>{row.location}</b><br>
            Risk: <b>{level}</b><br>
            Flood probability: {row.flood_probability:.1f}%<br>
            Rainfall 24h: {getattr(row, 'rainfall_24h_mm', 0):.1f} mm<br>
            River level: {getattr(row, 'river_level_m', 0):.2f} m<br>
            River rise: {getattr(row, 'river_rise_rate_m_per_hr', 0):.2f} m/hr
            """

            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=9,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"{row.location} • {level}"
            ).add_to(cluster)

        folium.LayerControl().add_to(m)

        components.html(m._repr_html_(), height=650, scrolling=False)

    st.markdown('<div class="section">Regional Risk Distribution</div>', unsafe_allow_html=True)

    dist = (
        df["risk_level_ai"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["risk_level", "count"]

    fig = px.bar(
        dist,
        x="risk_level",
        y="count",
        template="plotly_dark",
        labels={"risk_level": "Risk level", "count": "Records"},
        category_orders={"risk_level": ["LOW", "MODERATE", "HIGH", "CRITICAL"]}
    )
    fig = make_chart_layout(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# AI PREDICTION
# ============================================================
elif page == "AI Prediction":

    st.markdown('<div class="section">AI Flood Prediction</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Flood model not found.")
        st.stop()

    st.markdown(
        '<div class="info-box">Enter environmental conditions to estimate flood probability. This is a decision-support prototype, not an official warning system.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        rainfall_1h = st.number_input("Rainfall — 1 hour (mm)", 0.0, 500.0, 25.0)
        rainfall_3h = st.number_input("Rainfall — 3 hours (mm)", 0.0, 800.0, 55.0)
        rainfall_6h = st.number_input("Rainfall — 6 hours (mm)", 0.0, 1000.0, 90.0)
        rainfall_24h = st.number_input("Rainfall — 24 hours (mm)", 0.0, 1500.0, 160.0)

    with c2:
        soil = st.number_input("Soil moisture (%)", 0.0, 100.0, 55.0)
        river = st.number_input("River level (m)", 0.0, 20.0, 2.5)
        rise = st.number_input("River rise rate (m/hr)", 0.0, 10.0, 0.4)
        elevation = st.number_input("Elevation (m)", 0.0, 8000.0, 1800.0)

    with c3:
        slope = st.number_input("Slope (degrees)", 0.0, 90.0, 25.0)
        distance = st.number_input("Distance from river (km)", 0.0, 50.0, 1.5)

    input_df = pd.DataFrame([{
        "rainfall_1h_mm": rainfall_1h,
        "rainfall_3h_mm": rainfall_3h,
        "rainfall_6h_mm": rainfall_6h,
        "rainfall_24h_mm": rainfall_24h,
        "soil_moisture_percent": soil,
        "river_level_m": river,
        "river_rise_rate_m_per_hr": rise,
        "elevation_m": elevation,
        "slope_degrees": slope,
        "distance_from_river_km": distance,
        "rainfall_intensity": rainfall_1h / (rainfall_24h + 1),
        "river_rise_intensity": rise * river,
        "terrain_exposure": slope / (distance + 0.1)
    }])

    if st.button("Run AI Prediction", type="primary", use_container_width=True):

        try:
            probability = float(predict_probability(input_df)[0])
            level = risk_level(probability)

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">Flood Probability</div>'
                    f'<div class="kpi-value">{probability:.1f}%</div>'
                    f'<div class="kpi-sub">Random Forest prediction</div></div>',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">Risk Level</div>'
                    f'<div class="kpi-value">{level}</div>'
                    f'<div class="kpi-sub">Prototype threshold</div></div>',
                    unsafe_allow_html=True
                )

            with c3:
                confidence = max(probability, 100 - probability)
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">Model Confidence</div>'
                    f'<div class="kpi-value">{confidence:.1f}%</div>'
                    f'<div class="kpi-sub">Class probability margin</div></div>',
                    unsafe_allow_html=True
                )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={"suffix": "%"},
                title={"text": "Flood Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"thickness": 0.25},
                    "steps": [
                        {"range": [0, 25], "color": "#dcfce7"},
                        {"range": [25, 50], "color": "#fef9c3"},
                        {"range": [50, 75], "color": "#ffedd5"},
                        {"range": [75, 100], "color": "#fee2e2"},
                    ],
                }
            ))
            gauge = make_chart_layout(gauge, 360)
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown('<div class="section">What-if Rainfall Scenario</div>', unsafe_allow_html=True)

            increase = st.slider("Increase rainfall by", 0, 100, 20, step=5)

            scenario = input_df.copy()
            for col in [
                "rainfall_1h_mm",
                "rainfall_3h_mm",
                "rainfall_6h_mm",
                "rainfall_24h_mm"
            ]:
                scenario[col] *= (1 + increase / 100)

            scenario["rainfall_intensity"] = (
                scenario["rainfall_1h_mm"] /
                (scenario["rainfall_24h_mm"] + 1)
            )

            scenario_probability = float(predict_probability(scenario)[0])

            comparison = pd.DataFrame({
                "Scenario": ["Current", f"+{increase}% rainfall"],
                "Probability": [probability, scenario_probability]
            })

            fig = px.bar(
                comparison,
                x="Scenario",
                y="Probability",
                template="plotly_dark",
                labels={"Probability": "Flood probability (%)"}
            )
            fig = make_chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ============================================================
# RISK SCORE
# ============================================================
elif page == "Risk Score":

    st.markdown('<div class="section">Composite Risk Score</div>', unsafe_allow_html=True)

    latest = df.sort_values("timestamp").iloc[-1]

    ai_score = float(latest["flood_probability"])
    rainfall_score = min(float(latest.get("rainfall_24h_mm", 0)) / 250 * 100, 100)
    river_score = min(float(latest.get("river_rise_rate_m_per_hr", 0)) / 2 * 100, 100)
    soil_score = min(float(latest.get("soil_moisture_percent", 0)), 100)
    population_score = min(float(latest.get("population", 0)) / 5000 * 100, 100)
    infrastructure_score = min(
        (float(latest.get("hospitals", 0)) + float(latest.get("schools", 0))) / 20 * 100,
        100
    )

    total = (
        ai_score * .50 +
        rainfall_score * .15 +
        river_score * .15 +
        soil_score * .05 +
        population_score * .10 +
        infrastructure_score * .05
    )

    level = risk_level(total)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Overall Risk Score</div>'
            f'<div class="kpi-value">{total:.1f}/100</div>'
            f'<div class="kpi-sub">{level}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.metric("AI Probability", f"{ai_score:.1f}%")

    with c3:
        st.metric("Rainfall 24h", f'{latest.get("rainfall_24h_mm",0):.1f} mm')

    factors = pd.DataFrame({
        "Indicator": [
            "AI probability",
            "Rainfall",
            "River rise",
            "Soil moisture",
            "Population",
            "Infrastructure"
        ],
        "Factor": [
            ai_score,
            rainfall_score,
            river_score,
            soil_score,
            population_score,
            infrastructure_score
        ]
    }).sort_values("Factor", ascending=True)

    fig = px.bar(
        factors,
        x="Factor",
        y="Indicator",
        orientation="h",
        template="plotly_dark",
        labels={"Factor": "Indicator score", "Indicator": ""}
    )
    fig = make_chart_layout(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="info-box"><b>Method:</b> The composite score combines the AI probability with environmental and exposure indicators. The weights are initial engineering assumptions for the prototype and are not official disaster-management thresholds.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# EXPLAINABLE AI
# ============================================================
elif page == "Explainable AI":

    st.markdown('<div class="section">Explainable AI — Why is the risk high?</div>', unsafe_allow_html=True)

    if SHAP_PATH.exists():
        shap_df = pd.read_csv(SHAP_PATH)

        # Handle common column names
        feature_col = next(
            (c for c in shap_df.columns if c.lower() in ["feature", "features", "variable"]),
            shap_df.columns[0]
        )
        value_col = next(
            (c for c in shap_df.columns if "shap" in c.lower() or "importance" in c.lower()),
            shap_df.columns[-1]
        )

        chart = shap_df[[feature_col, value_col]].copy()
        chart.columns = ["Feature", "Importance"]
        chart["Importance"] = pd.to_numeric(chart["Importance"], errors="coerce")
        chart = chart.dropna().sort_values("Importance", ascending=True).tail(12)

        fig = px.bar(
            chart,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark",
            labels={"Importance": "Mean |SHAP value|", "Feature": ""}
        )
        fig = make_chart_layout(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

        top_feature = chart.iloc[-1]["Feature"]

        st.markdown(
            f'<div class="card"><h3 style="color:#f2f7fd;">Top contributing factor</h3>'
            f'<div style="font-size:24px;font-weight:800;color:#38bdf8;">{top_feature}</div>'
            f'<p style="color:#aebfd2;">Higher SHAP importance means the model relied more heavily on that feature across the analysed samples.</p></div>',
            unsafe_allow_html=True
        )

    else:
        st.warning("SHAP feature importance file not found: models/shap_feature_importance.csv")

    st.markdown('<div class="section">Model Interpretation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>How explainability works</b><br><br>
    SHAP (SHapley Additive exPlanations) helps show which input variables contribute most to the Random Forest prediction.
    <br><br>
    <b>Example interpretation:</b> if recent rainfall and soil moisture have high importance, the system can explain that the elevated flood probability is strongly associated with saturated ground and accumulated rainfall.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ALERTS
# ============================================================
elif page == "Alerts":

    st.markdown('<div class="section">Early Warning & Alerts</div>', unsafe_allow_html=True)

    alert_df = None
    if ALERT_PATH.exists():
        alert_df = pd.read_csv(ALERT_PATH)

    source = df.copy()

    counts = source["risk_level_ai"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MODERATE", "LOW"], fill_value=0
    )

    c1, c2, c3, c4 = st.columns(4)

    for col, name in zip(
        [c1, c2, c3, c4],
        ["CRITICAL", "HIGH", "MODERATE", "LOW"]
    ):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{name}</div>'
                f'<div class="kpi-value">{counts[name]:,}</div>'
                f'<div class="kpi-sub">records</div></div>',
                unsafe_allow_html=True
            )

    high_df = source[source["flood_probability"] >= 50].copy()

    st.markdown('<div class="section">Priority Alerts</div>', unsafe_allow_html=True)

    if high_df.empty:
        st.success("No high-risk records in the current dataset.")
    else:
        display_cols = [
            c for c in [
                "timestamp", "location", "flood_probability",
                "risk_level_ai", "rainfall_24h_mm",
                "river_level_m", "river_rise_rate_m_per_hr"
            ] if c in high_df.columns
        ]

        high_df = high_df.sort_values("flood_probability", ascending=False).head(25)
        st.dataframe(
            high_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        '<div class="alert-box"><b>Operational note:</b> These alerts are generated from the prototype dataset/model. Production alerts should be connected to validated sensors, official warning thresholds and an authorized emergency-management workflow.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# POPULATION
# ============================================================
elif page == "Population at Risk":

    st.markdown('<div class="section">Population & Infrastructure at Risk</div>', unsafe_allow_html=True)

    pop_df = None
    if POP_PATH.exists():
        pop_df = pd.read_csv(POP_PATH)

    if pop_df is not None and not pop_df.empty:
        numeric_candidates = [
            "population_at_risk",
            "households_at_risk",
            "hospitals_at_risk",
            "schools_at_risk",
            "shelters_at_risk"
        ]

        values = {}
        for c in numeric_candidates:
            values[c] = pd.to_numeric(pop_df[c], errors="coerce").fillna(0).sum() if c in pop_df.columns else 0

        c1, c2, c3, c4 = st.columns(4)

        labels = [
            ("Population", values["population_at_risk"]),
            ("Households", values["households_at_risk"]),
            ("Hospitals", values["hospitals_at_risk"]),
            ("Schools", values["schools_at_risk"]),
        ]

        for col, (label, value) in zip([c1, c2, c3, c4], labels):
            with col:
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{value:,.0f}</div>'
                    f'<div class="kpi-sub">estimated exposure</div></div>',
                    unsafe_allow_html=True
                )

        if "location" in pop_df.columns and "population_at_risk" in pop_df.columns:
            regional = (
                pop_df.groupby("location", as_index=False)["population_at_risk"]
                .sum()
                .sort_values("population_at_risk", ascending=False)
                .head(15)
            )

            fig = px.bar(
                regional,
                x="population_at_risk",
                y="location",
                orientation="h",
                template="plotly_dark",
                labels={"population_at_risk": "Estimated population at risk", "location": ""}
            )
            fig = make_chart_layout(fig, 450)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pop_df, use_container_width=True, hide_index=True)

    else:
        st.info("Population risk file not found. Showing a prototype exposure estimate from the current dataset.")

        exposure = df.copy()
        exposure["population_at_risk_est"] = (
            exposure["population"] * exposure["flood_probability"] / 100
        )

        regional = (
            exposure.groupby("location", as_index=False)["population_at_risk_est"]
            .sum()
            .sort_values("population_at_risk_est", ascending=False)
            .head(15)
        )

        fig = px.bar(
            regional,
            x="population_at_risk_est",
            y="location",
            orientation="h",
            template="plotly_dark",
            labels={"population_at_risk_est": "Estimated population at risk", "location": ""}
        )
        fig = make_chart_layout(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EVACUATION
# ============================================================
elif page == "Evacuation":

    st.markdown('<div class="section">Risk-Aware Evacuation Planning</div>', unsafe_allow_html=True)

    locations = sorted(df["location"].dropna().unique().tolist())
    selected = st.selectbox("Select high-risk location", locations)

    selected_df = df[df["location"] == selected].sort_values("timestamp")

    if selected_df.empty:
        st.warning("Location not found.")
    else:
        row = selected_df.iloc[-1]
        probability = float(row["flood_probability"])
        level = risk_level(probability)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Selected Area</div>'
                f'<div class="kpi-value" style="font-size:22px;">{selected}</div>'
                f'<div class="kpi-sub">monitored location</div></div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Flood Risk</div>'
                f'<div class="kpi-value">{probability:.1f}%</div>'
                f'<div class="kpi-sub">{level}</div></div>',
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Population</div>'
                f'<div class="kpi-value">{float(row.get("population",0)):,.0f}</div>'
                f'<div class="kpi-sub">local exposure base</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section">Recommended Response</div>', unsafe_allow_html=True)

        if level == "CRITICAL":
            action = "Initiate emergency evacuation procedures and move people to designated safe shelters."
        elif level == "HIGH":
            action = "Prepare evacuation teams, verify shelters and issue precautionary warnings."
        elif level == "MODERATE":
            action = "Increase monitoring and prepare local response resources."
        else:
            action = "Continue monitoring environmental conditions."

        st.markdown(
            f'<div class="alert-box"><b>Recommended action:</b><br>{action}</div>',
            unsafe_allow_html=True
        )

        route_file = ROOT / "models" / "risk_aware_evacuation.html"

        if route_file.exists():
            try:
                html = route_file.read_text(encoding="utf-8")
                components.html(html, height=600, scrolling=False)
            except Exception:
                st.info("Evacuation route map is available in models/risk_aware_evacuation.html")
        else:
            st.info("Risk-aware route map has not been generated yet.")

        st.markdown(
            '<div class="info-box"><b>Prototype limitation:</b> evacuation routing currently uses a demonstration road network. A production system should use validated road/shelter GIS data, road closures, terrain hazards and live accessibility information.</div>',
            unsafe_allow_html=True
        )

# ============================================================
# SATELLITE / GLOF
# ============================================================
elif page == "Satellite Analysis":

    st.markdown('<div class="section">Satellite & GLOF Monitoring</div>', unsafe_allow_html=True)

    lake_path = ROOT / "models" / "glof_lake_analysis.csv"
    map_path = ROOT / "models" / "glof_monitoring_map.html"

    if lake_path.exists():
        lake_df = pd.read_csv(lake_path)

        c1, c2, c3 = st.columns(3)

        area_change = pd.to_numeric(
            lake_df.get("area_change_percent", pd.Series(dtype=float)),
            errors="coerce"
        )

        significant = int((area_change.abs() >= 10).sum()) if not area_change.empty else 0
        max_change = float(area_change.abs().max()) if not area_change.empty else 0

        with c1:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Monitored Lakes</div>'
                f'<div class="kpi-value">{len(lake_df)}</div>'
                f'<div class="kpi-sub">prototype monitoring set</div></div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Significant Changes</div>'
                f'<div class="kpi-value">{significant}</div>'
                f'<div class="kpi-sub">≥ 10% area change</div></div>',
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Maximum Change</div>'
                f'<div class="kpi-value">{max_change:.1f}%</div>'
                f'<div class="kpi-sub">absolute area change</div></div>',
                unsafe_allow_html=True
            )

        st.dataframe(lake_df, use_container_width=True, hide_index=True)

        if map_path.exists():
            components.html(
                map_path.read_text(encoding="utf-8"),
                height=600,
                scrolling=False
            )

    else:
        st.warning("GLOF analysis file not found. Expected models/glof_lake_analysis.csv")

    st.markdown(
        '<div class="info-box"><b>GLOF interpretation:</b> the prototype monitors changes in glacial-lake characteristics and related indicators. It does not claim to predict the exact time of glacier or lake collapse. Production deployment should use validated satellite observations and specialist hydrological/glaciological models.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# CITIZEN REPORTS
# ============================================================
elif page == "Citizen Reports":

    st.markdown('<div class="section">Citizen Incident Reports</div>', unsafe_allow_html=True)

    if "reports" not in st.session_state:
        st.session_state.reports = []

    with st.form("citizen_report_form"):
        c1, c2 = st.columns(2)

        with c1:
            report_location = st.text_input("Location")
            report_type = st.selectbox(
                "Incident type",
                ["Flooding", "River rise", "Landslide", "Road blocked", "Bridge issue", "Other"]
            )
            severity = st.selectbox(
                "Observed severity",
                ["Low", "Moderate", "High", "Critical"]
            )

        with c2:
            description = st.text_area("Description")
            reporter = st.text_input("Reporter / organization")

        submitted = st.form_submit_button("Submit Incident Report", type="primary")

        if submitted:
            st.session_state.reports.append({
                "Location": report_location,
                "Type": report_type,
                "Severity": severity,
                "Description": description,
                "Reporter": reporter,
            })
            st.success("Citizen report submitted successfully.")

    st.markdown('<div class="section">Recent Reports</div>', unsafe_allow_html=True)

    if st.session_state.reports:
        reports_df = pd.DataFrame(st.session_state.reports)
        st.dataframe(reports_df, use_container_width=True, hide_index=True)

        distribution = (
            reports_df["Severity"]
            .value_counts()
            .reindex(["Low", "Moderate", "High", "Critical"], fill_value=0)
            .reset_index()
        )
        distribution.columns = ["Severity", "Count"]

        fig = px.bar(
            distribution,
            x="Severity",
            y="Count",
            template="plotly_dark",
            labels={"Severity": "Severity", "Count": "Reports"}
        )
        fig = make_chart_layout(fig, 330)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No citizen reports have been submitted during this session.")

    st.markdown(
        '<div class="info-box"><b>Future enhancement:</b> citizen reports can be geotagged, photo-verified and fused with AI risk predictions to improve situational awareness.</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">FLOOD-X • AI-Based Flash Flood Prediction & Risk Intelligence • SIH 2026 Prototype<br>'
    'For demonstration and research purposes. Not an operational emergency-warning system.</div>',
    unsafe_allow_html=True
)
