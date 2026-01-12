import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# --- STEP 1: LOAD YOUR DATA ---
df = pd.read_csv('data/women_travel_safety_dataset_updated.csv')

# Features (X) are your inputs (lighting, emergency count, etc.)
# Target (y) is the Safety Probability
X = df.drop('safety_probability', axis=1)
y = df['safety_probability']

# --- STEP 2: PREPARE DATA FOR LOGISTIC REGRESSION ---
# Logistic Regression needs Categories (0, 1, 2) instead of 0-100%
# We create a new target 'y_class' based on your project requirements
def categorize_safety(p):
    if p < 40: return 0    # Unsafe
    if p < 70: return 1    # Cautious
    return 2               # Safe

y_class = y.apply(categorize_safety)

# --- STEP 3: SPLIT DATA ---
# We keep 20% of data aside to test if the models actually learned
X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
    X, y, y_class, test_size=0.2, random_state=42
)

# --- STEP 4: TRAIN GRADIENT BOOSTING (For the % Score) ---
# This is a powerful model that learns from its own mistakes
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# --- STEP 5: TRAIN LOGISTIC REGRESSION ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# The updated line:
log_model = LogisticRegression(max_iter=1000) 
log_model.fit(X_train_scaled, yc_train)
# --- STEP 6: SAVE EVERYTHING FOR THE UI TEAM ---
# Your teammates need these files to build the website
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Success! Models trained and saved.")


# --- STEP 7: EVALUATE (For your Project Report) ---
# Check Gradient Boosting (Percentage Error)
from sklearn.metrics import mean_absolute_error
gb_preds = gb_model.predict(X_test)
mae = mean_absolute_error(y_test, gb_preds)

# Check Logistic Regression (Classification Accuracy)
X_test_scaled = scaler.transform(X_test) # Scale the test data too!
log_accuracy = log_model.score(X_test_scaled, yc_test)

print(f"\n--- PROJECT METRICS ---")
print(f"Gradient Boosting Error: {mae:.2f}% (Lower is better)")
print(f"Logistic Regression Accuracy: {log_accuracy * 100:.2f}% (Higher is better)")