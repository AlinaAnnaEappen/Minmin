import pandas as pd

# =========================
# STAGE 1: Load Dataset
# =========================
data = pd.read_csv("data/women_travel_safety_dataset_updated.csv")

print(data.head())
print("Dataset shape:", data.shape)


# =========================
# STAGE 2: Prepare Data
# =========================
from sklearn.model_selection import train_test_split

X = data.drop("safety_probability", axis=1)
y = data["safety_probability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
# STAGE 3: Logistic Regression (Baseline)
# =========================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

log_model = LogisticRegression(max_iter=1000)

# Convert safety score into binary (baseline model)
y_train_binary = (y_train >= 60).astype(int)
y_test_binary = (y_test >= 60).astype(int)

log_model.fit(X_train, y_train_binary)

log_preds = log_model.predict(X_test)

print("\nLogistic Regression MAE:",
      mean_absolute_error(y_test_binary, log_preds))


# =========================
# STAGE 4: Gradient Boosting (Final Model)
# =========================
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict(X_test)

print("Gradient Boosting MAE:",
      mean_absolute_error(y_test, gb_preds))


# =========================
# STAGE 5: Safety Category Function
# =========================
def categorize_safety(score):
    if score < 40:
        return "Unsafe"
    elif score < 70:
        return "Caution"
    else:
        return "Safe"



# =========================
# STAGE 7: Predict for New Location
# =========================
sample_location = pd.DataFrame([[
    21,     # travel_hour
    1,      # is_night
    0,      # is_weekend
    0.3,    # street_lighting
    0.4,    # road_type_score
    3,      # nearby_shops
    2.5,    # police_distance
    1.8,    # hospital_distance
    2,      # emergency_count
    6,      # poi_count
    1,      # bus_stop_count
    0.45    # commercial_density
]], columns=X_train.columns)

predicted_score = gb_model.predict(sample_location)[0]
predicted_category = categorize_safety(predicted_score)

print("\nPredicted Safety Probability:", round(predicted_score, 2), "%")
print("Safety Category:", predicted_category)
