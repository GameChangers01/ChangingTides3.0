import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the sea level data
url = r"data\Sea level.xlsx"  # Replace with the actual path to your Excel file
sea_level_df = pd.read_excel(url)

# Display the columns to check their names
print(sea_level_df.columns)

# Assuming 'Year' is present in the columns, select relevant features and target
features = ["Year "]
target = "Level"

# Check if the 'Year' column is present in the DataFrame
if "Year " in sea_level_df.columns:
    X = sea_level_df[features]
    y = sea_level_df[target]

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

    # Predict the sea level for the year 2100
    year_2100 = np.array([[2100]])
    predicted_sea_level_2100 = gb_model.predict(year_2100)[0]

    print(f"Predicted sea level for the year 2100: {predicted_sea_level_2100}")
else:
    print("The 'Year ' column is not present in the DataFrame.")
