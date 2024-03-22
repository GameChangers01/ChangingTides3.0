import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file
df = pd.read_csv("data\B_temp_modifies")

# Extract year from 'Date' column
df["Year"] = pd.to_datetime(df["Year"]).dt.year

# Select features and target variables
X = df[["Year"]]
y_max = df["Temp Max"]
y_min = df["Temp Min"]
y_avg = df["Temp Avg"]

# Split the data into training and testing sets
(
    X_train,
    X_test,
    y_max_train,
    y_max_test,
    y_min_train,
    y_min_test,
    y_avg_train,
    y_avg_test,
) = train_test_split(X, y_max, y_min, y_avg, test_size=0.2, random_state=42)

# Train Gradient Boosting models for 'Temp Max', 'Temp Min', and 'Temp Avg'
model_max = GradientBoostingRegressor(n_estimators=1000, random_state=42)
model_max.fit(X_train, y_max_train)

model_min = GradientBoostingRegressor(n_estimators=1000, random_state=42)
model_min.fit(X_train, y_min_train)

model_avg = GradientBoostingRegressor(n_estimators=1000, random_state=42)
model_avg.fit(X_train, y_avg_train)

# Make predictions for the year 2050
year_2050 = pd.DataFrame({"Year": [2050]})
temp_max_2050 = model_max.predict(year_2050)
temp_min_2050 = model_min.predict(year_2050)
temp_avg_2050 = model_avg.predict(year_2050)

# Print the predicted temperatures for 2050
print(f"Predicted Temp Max for 2050: {temp_max_2050[0]}")
print(f"Predicted Temp Min for 2050: {temp_min_2050[0]}")
print(f"Predicted Temp Avg for 2050: {temp_avg_2050[0]}")

# Evaluate the models on the test set
test_preds_max = model_max.predict(X_test)
test_preds_min = model_min.predict(X_test)
test_preds_avg = model_avg.predict(X_test)

test_mse_max = mean_squared_error(y_max_test, test_preds_max)
test_r2_max = r2_score(y_max_test, test_preds_max)

test_mse_min = mean_squared_error(y_min_test, test_preds_min)
test_r2_min = r2_score(y_min_test, test_preds_min)

test_mse_avg = mean_squared_error(y_avg_test, test_preds_avg)
test_r2_avg = r2_score(y_avg_test, test_preds_avg)

# Print accuracy metrics for 'Temp Max'
print(f"\nAccuracy for Temp Max:")
print(f" {1-test_mse_max}")

# Print accuracy metrics for 'Temp Min'
print(f"\nAccuracy for Temp Min:")
print(f" {1-test_mse_min}")

# Print accuracy metrics for 'Temp Avg'
print(f"\nAccuracy for Temp Avg:")
print(f" {1-test_mse_avg}")
