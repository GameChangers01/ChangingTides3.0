from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler


# Load the CSV file
df = pd.read_csv(r"data\rain.csv")

# Drop rows with NaN value
df = df.dropna()

# Extract relevant features and normalize data
years = df["Year"].values.reshape(-1, 1)
total_rainfall = df["Total"].values.reshape(-1, 1)

scaler_year = MinMaxScaler()
scaler_total = MinMaxScaler()

years_scaled = scaler_year.fit_transform(years)
total_scaled = scaler_total.fit_transform(total_rainfall)

# Prepare sequences
X, y = [], []
for i in range(len(years_scaled) - 1):
    X.append(np.hstack((years_scaled[i], total_scaled[i])))
    y.append(total_scaled[i + 1])

X, y = np.array(X), np.array(y)

# Build Gradient Boosting model
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gradient_boosting_model.fit(X, y.ravel())

# Make predictions on the training set
train_preds_scaled_gb = gradient_boosting_model.predict(X)
train_preds_gb = scaler_total.inverse_transform(train_preds_scaled_gb.reshape(-1, 1))
train_mse_gb = mean_squared_error(total_rainfall[1:], train_preds_gb)
train_r2_gb = r2_score(total_rainfall[1:], train_preds_gb)
# print(f"Mean Squared Error on Training Data (Gradient Boosting): {train_mse_gb}")

print(f"R-squared (R2) Score on Training Data (Gradient Boosting): {train_r2_gb}")


user_year = int(input("Enter the year for rainfall prediction: "))
user_input = np.array(
    [[user_year, 0]]
)  # Assuming 0 for total rainfall (you can adjust this based on your data)

# Scale the user input features
user_input_scaled = np.hstack(
    (
        scaler_year.transform(user_input[:, :1]),
        scaler_total.transform(user_input[:, 1:]),
    )
)

# Reshape for the model input
user_input_reshaped = user_input_scaled.reshape(1, -1)

# Predict the total rainfall for the user input year using Gradient Boosting
predicted_total_scaled = gradient_boosting_model.predict(user_input_reshaped)
predicted_total = scaler_total.inverse_transform(predicted_total_scaled.reshape(-1, 1))

# Print the predicted total rainfall for the user input year
print(f"Predicted Total Rainfall for {user_year}: {predicted_total[0, 0] / 12}")
