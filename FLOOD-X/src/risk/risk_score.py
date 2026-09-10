import pandas as pd


def calculate_risk_score(
    flood_probability,
    rainfall_24h,
    river_rise_rate,
    soil_moisture,
    population,
    hospitals,
    schools
):
    """
    Calculate FLOOD-X risk score from 0 to 100.

    Combines:
    - AI flood probability
    - rainfall hazard
    - river rise hazard
    - soil moisture
    - population exposure
    - critical infrastructure exposure
    """

    # --------------------------------------------------
    # 1. AI HAZARD SCORE
    # --------------------------------------------------

    ai_score = flood_probability * 100

    # --------------------------------------------------
    # 2. RAINFALL SCORE
    # --------------------------------------------------

    rainfall_score = min(
        (rainfall_24h / 250) * 100,
        100
    )

    # --------------------------------------------------
    # 3. RIVER RISE SCORE
    # --------------------------------------------------

    river_score = min(
        (river_rise_rate / 2) * 100,
        100
    )

    # --------------------------------------------------
    # 4. SOIL MOISTURE SCORE
    # --------------------------------------------------

    soil_score = min(
        (soil_moisture / 100) * 100,
        100
    )

    # --------------------------------------------------
    # 5. POPULATION EXPOSURE
    # --------------------------------------------------

    population_score = min(
        (population / 5000) * 100,
        100
    )

    # --------------------------------------------------
    # 6. INFRASTRUCTURE EXPOSURE
    # --------------------------------------------------

    infrastructure_count = (
        hospitals +
        schools
    )

    infrastructure_score = min(
        (infrastructure_count / 20) * 100,
        100
    )

    # --------------------------------------------------
    # 7. FINAL RISK SCORE
    # --------------------------------------------------

    risk_score = (
        ai_score * 0.50 +
        rainfall_score * 0.15 +
        river_score * 0.15 +
        soil_score * 0.05 +
        population_score * 0.10 +
        infrastructure_score * 0.05
    )

    risk_score = min(
        max(risk_score, 0),
        100
    )

    # --------------------------------------------------
    # 8. RISK LEVEL
    # --------------------------------------------------

    if risk_score < 25:
        risk_level = "LOW"

    elif risk_score < 50:
        risk_level = "MODERATE"

    elif risk_score < 75:
        risk_level = "HIGH"

    else:
        risk_level = "CRITICAL"

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "ai_score": round(ai_score, 2),
        "rainfall_score": round(rainfall_score, 2),
        "river_score": round(river_score, 2),
        "soil_score": round(soil_score, 2),
        "population_score": round(population_score, 2),
        "infrastructure_score": round(
            infrastructure_score,
            2
        )
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    result = calculate_risk_score(
        flood_probability=0.82,
        rainfall_24h=220,
        river_rise_rate=0.8,
        soil_moisture=78,
        population=2500,
        hospitals=3,
        schools=8
    )

    print("\n==============================")
    print("FLOOD-X RISK SCORE")
    print("==============================")

    print(
        f"Overall Risk Score: "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )

    print("\nRisk Factor Breakdown:")

    for key, value in result.items():

        if key not in ["risk_score", "risk_level"]:
            print(f"{key}: {value}")