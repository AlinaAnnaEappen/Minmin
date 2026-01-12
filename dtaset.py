import pandas as pd
import random
from datetime import datetime

data = []

NUM_ROWS = 800  # recommended size

for _ in range(NUM_ROWS):
    # Temporal features
    travel_hour = random.randint(0, 23)
    is_night = 1 if (travel_hour >= 19 or travel_hour <= 5) else 0
    is_weekend = random.randint(0, 1)

    # Infrastructure features
    street_lighting = random.randint(0, 1)
    road_type_score = round(random.uniform(0.3, 1.0), 2)
    nearby_shops = random.randint(0, 20)

    # Emergency access
    police_distance = round(random.uniform(0.3, 5.0), 2)
    hospital_distance = round(random.uniform(0.3, 6.0), 2)
    emergency_count = random.randint(0, 5)

    # Crowd proxies
    poi_count = random.randint(0, 30)
    bus_stop_count = random.randint(0, 8)
    commercial_density = round(random.uniform(0.0, 1.0), 2)

    # Rule-based safety calculation
    safety = 45
    safety += street_lighting * 10
    safety += (1 - is_night) * 10
    safety += road_type_score * 10
    safety += (poi_count / 30) * 10
    safety += (1 / police_distance) * 5
    safety += (commercial_density * 5)

    safety_probability = max(0, min(100, round(safety)))

    data.append([
        travel_hour, is_night, is_weekend,
        street_lighting, road_type_score, nearby_shops,
        police_distance, hospital_distance, emergency_count,
        poi_count, bus_stop_count, commercial_density,
        safety_probability
    ])

columns = [
    "travel_hour", "is_night", "is_weekend",
    "street_lighting", "road_type_score", "nearby_shops",
    "police_distance", "hospital_distance", "emergency_count",
    "poi_count", "bus_stop_count", "commercial_density",
    "safety_probability"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("women_travel_safety_dataset.csv", index=False)

print("Dataset generated successfully with", NUM_ROWS, "rows")
