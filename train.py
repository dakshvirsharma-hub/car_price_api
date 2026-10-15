import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("cardekho_data (1).csv")

# Drop target leakage or unnecessary columns if any
# Assuming 'Selling_Price' is the target
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# One-hot encode categorical columns
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission", "Owner", "Car_Name"]
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Save feature columns for inference
feature_columns = list(X_encoded.columns)
joblib.dump(feature_columns, "feature_columns.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "random_forest_model.pkl")

print(f"✅ Model trained! R² score: {model.score(X_test, y_test):.4f}")
print("✅ Saved: random_forest_model.pkl, feature_columns.pkl")
