import networkx as nx
import folium


# ==================================================
# FLOOD-X RISK-AWARE EVACUATION ROUTING
# ==================================================


# --------------------------------------------------
# CREATE ROAD NETWORK
# --------------------------------------------------

G = nx.Graph()


# --------------------------------------------------
# LOCATION DATA
# --------------------------------------------------

locations = {

    "Village_A": (30.1000, 78.3000),
    "Village_B": (30.1050, 78.3100),
    "Village_C": (30.1100, 78.3200),
    "Village_D": (30.1150, 78.3300),

    "Junction_1": (30.1030, 78.3050),
    "Junction_2": (30.1080, 78.3150),
    "Junction_3": (30.1120, 78.3250),

    "Shelter_1": (30.1200, 78.3400),
    "Shelter_2": (30.0950, 78.2900)
}


# --------------------------------------------------
# ADD NODES
# --------------------------------------------------

for node, coordinates in locations.items():

    G.add_node(
        node,
        pos=coordinates
    )


# --------------------------------------------------
# ROAD INFORMATION
# --------------------------------------------------

# distance = road distance
# flood_risk = estimated flood risk of road
# slope_risk = terrain/slope risk
#
# Values are prototype values.
# Later these can come from GIS/real road data.

roads = [

    ("Village_A", "Junction_1", 1.0, 20, 10),

    ("Village_B", "Junction_1", 1.2, 35, 15),

    ("Village_B", "Junction_2", 1.0, 45, 20),

    ("Village_C", "Junction_2", 1.1, 70, 35),

    ("Village_C", "Junction_3", 1.0, 25, 15),

    ("Village_D", "Junction_3", 1.3, 40, 25),

    ("Junction_1", "Junction_2", 1.0, 30, 15),

    ("Junction_2", "Junction_3", 1.0, 65, 30),

    ("Junction_3", "Shelter_1", 1.5, 15, 10),

    ("Village_A", "Shelter_2", 1.4, 10, 5)
]


# --------------------------------------------------
# ADD ROADS TO NETWORK
# --------------------------------------------------

for start, end, distance, flood_risk, slope_risk in roads:

    # Risk-aware cost
    #
    # Distance has a base weight.
    # Flood and slope increase the cost.
    
    risk_penalty = (
        flood_risk * 0.05 +
        slope_risk * 0.03
    )

    safe_cost = distance + risk_penalty

    G.add_edge(
        start,
        end,
        distance=distance,
        flood_risk=flood_risk,
        slope_risk=slope_risk,
        safe_cost=safe_cost
    )


# --------------------------------------------------
# START LOCATION
# --------------------------------------------------

start_location = "Village_C"


# --------------------------------------------------
# SHELTERS
# --------------------------------------------------

shelters = [
    "Shelter_1",
    "Shelter_2"
]


# --------------------------------------------------
# FIND RISK-AWARE ROUTES
# --------------------------------------------------

routes = []


for shelter in shelters:

    try:

        route = nx.shortest_path(
            G,
            source=start_location,
            target=shelter,
            weight="safe_cost"
        )

        safe_cost = nx.shortest_path_length(
            G,
            source=start_location,
            target=shelter,
            weight="safe_cost"
        )

        distance = nx.shortest_path_length(
            G,
            source=start_location,
            target=shelter,
            weight="distance"
        )

        routes.append({

            "shelter": shelter,

            "route": route,

            "safe_cost": safe_cost,

            "distance": distance

        })

    except nx.NetworkXNoPath:

        pass


# --------------------------------------------------
# SELECT SAFEST ROUTE
# --------------------------------------------------

routes = sorted(
    routes,
    key=lambda x: x["safe_cost"]
)


best_route = routes[0]


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\n==============================")
print("FLOOD-X RISK-AWARE EVACUATION")
print("==============================")


print(
    f"\nStarting location:"
    f" {start_location}"
)


print(
    f"Recommended shelter:"
    f" {best_route['shelter']}"
)


print(
    f"Actual route distance:"
    f" {best_route['distance']:.2f} km"
)


print(
    f"Risk-adjusted route cost:"
    f" {best_route['safe_cost']:.2f}"
)


print("\nRecommended Safe Route:")


print(
    " → ".join(
        best_route["route"]
    )
)


# --------------------------------------------------
# CREATE MAP
# --------------------------------------------------

start_coordinates = locations[
    start_location
]


evacuation_map = folium.Map(
    location=start_coordinates,
    zoom_start=13,
    tiles="OpenStreetMap"
)


# --------------------------------------------------
# ADD LOCATIONS
# --------------------------------------------------

for node, coordinates in locations.items():

    if node.startswith("Shelter"):

        icon_color = "green"

        popup_text = (
            f"<b>{node}</b><br>"
            "Emergency Shelter"
        )

    elif node == start_location:

        icon_color = "red"

        popup_text = (
            f"<b>{node}</b><br>"
            "Evacuation Starting Point"
        )

    else:

        icon_color = "blue"

        popup_text = node


    folium.Marker(
        location=coordinates,
        popup=popup_text,
        tooltip=node,
        icon=folium.Icon(
            color=icon_color
        )
    ).add_to(
        evacuation_map
    )


# --------------------------------------------------
# DRAW ALL ROADS
# --------------------------------------------------

for start, end, distance, flood_risk, slope_risk in roads:

    start_point = locations[start]

    end_point = locations[end]

    folium.PolyLine(
        [
            start_point,
            end_point
        ],
        weight=3,
        opacity=0.5,
        tooltip=(
            f"Distance: {distance} km | "
            f"Flood Risk: {flood_risk}% | "
            f"Slope Risk: {slope_risk}%"
        )
    ).add_to(
        evacuation_map
    )


# --------------------------------------------------
# HIGHLIGHT SAFEST ROUTE
# --------------------------------------------------

route_coordinates = [

    locations[node]

    for node in best_route["route"]

]


folium.PolyLine(
    route_coordinates,
    weight=8,
    opacity=1,
    tooltip="FLOOD-X Recommended Safe Route"
).add_to(
    evacuation_map
)


# --------------------------------------------------
# SAVE MAP
# --------------------------------------------------

output_path = (
    "models/risk_aware_evacuation.html"
)


evacuation_map.save(
    output_path
)


print(
    f"\nRisk-aware evacuation map saved to:"
    f"\n{output_path}"
)


print(
    "\nRisk-aware evacuation engine completed!"
)