import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
url = r"data\co-emissions-per-capita.csv"
df = pd.read_csv(url)

# Filter data for the IND code
df_ind = df[df["Code"] == "IND"]

# Select relevant features and target
features = ["Year"]
target = "Annual"

X = df_ind[features]
y = df_ind[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Predict the emissions for the year 2070

di = {}
for i in range(2020, 2100):
    year_2070 = np.array([[i]])
    predicted_emissions_2070 = gb_model.predict(year_2070)[0]
    di[i] = predicted_emissions_2070

    print(f"Predicted CO2 emissions per capita for 2070: {predicted_emissions_2070}")

print(di)
