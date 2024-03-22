import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = r"data\antarctic-ozone-hole-area.csv"
df = pd.read_csv(url)

# # Filter data for the IND code
# df_ind = df[df['Code'] == 'IND']

# Select relevant features and targets
features = ["Year"]
targets = ["Maximum", "Mean"]

X = df[features]
y_max = df["Maximum"]
y_mean = df["Mean"]

# Split the data into training and testing sets for Maximum
X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(
    X, y_max, test_size=0.3, random_state=4
)

# Create and train the Gradient Boosting model for Maximum
gb_model_max = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb_model_max.fit(X_train_max, y_train_max)

# Make predictions on the test set for Maximum
y_pred_max = gb_model_max.predict(X_test_max)

# Evaluate the model for Maximum
mse_max = mean_squared_error(y_test_max, y_pred_max)
r2_max = r2_score(y_test_max, y_pred_max)

print(f"Mean Squared Error for Maximum: {mse_max}")
print(f"R-squared for Maximum: {r2_max}")
