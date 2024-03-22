# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC


# def rainfall_prediction():
#     df = pd.read_csv(r"data\rain.csv")
#     df = df.dropna()

#     years = df["Year"].values.reshape(-1, 1)
#     total_rainfall = df["Total"].values.reshape(-1, 1)
#     scaler_year = MinMaxScaler()
#     scaler_total = MinMaxScaler()
#     years_scaled = scaler_year.fit_transform(years)
#     total_scaled = scaler_total.fit_transform(total_rainfall)
#     X, y = [], []
#     for i in range(len(years_scaled) - 1):
#         X.append(np.hstack((years_scaled[i], total_scaled[i])))
#         y.append(total_scaled[i + 1])
#     X, y = np.array(X), np.array(y)
#     gradient_boosting_model = GradientBoostingRegressor(
#         n_estimators=100, random_state=42
#     )
#     gradient_boosting_model.fit(X, y.ravel())
#     train_preds_scaled_gb = gradient_boosting_model.predict(X)
#     train_preds_gb = scaler_total.inverse_transform(
#         train_preds_scaled_gb.reshape(-1, 1)
#     )
#     train_mse_gb = mean_squared_error(total_rainfall[1:], train_preds_gb)
#     train_r2_gb = r2_score(total_rainfall[1:], train_preds_gb)

#     print(f"R-squared (R2) Score on Training Data (Gradient Boosting): {train_r2_gb}")
#     user_year = 2045
#     user_input = np.array([[user_year, 0]])
#     user_input_scaled = np.hstack(
#         (
#             scaler_year.transform(user_input[:, :1]),
#             scaler_total.transform(user_input[:, 1:]),
#         )
#     )
#     user_input_reshaped = user_input_scaled.reshape(1, -1)
#     predicted_total_scaled = gradient_boosting_model.predict(user_input_reshaped)
#     predicted_total = scaler_total.inverse_transform(
#         predicted_total_scaled.reshape(-1, 1)
#     )
#     print(f"Predicted Total Rainfall for {user_year}: {predicted_total[0, 0] / 12}")

#     acc = train_r2_gb
#     value = predicted_total[0, 0] / 12
#     return acc, value


# def train_and_predict_sea_level():
#     # Load the sea level data
#     url = r"data\Sea level.xlsx"  # Replace with the actual path to your Excel file
#     sea_level_df = pd.read_excel(url)

#     # Display the columns to check their names
#     print(sea_level_df.columns)

#     features = ["Year "]
#     target = "Level"

#     # Check if the 'Year' column is present in the DataFrame
#     if "Year " in sea_level_df.columns:
#         X = sea_level_df[features]
#         y = sea_level_df[target]

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # Create and train the Gradient Boosting model
#         gb_model = GradientBoostingRegressor(
#             n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
#         )
#         gb_model.fit(X_train, y_train)

#         # Make predictions on the test set
#         y_pred = gb_model.predict(X_test)

#         # Evaluate the model
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         print(f"R-squared: {r2}")

#         # Predict the sea level for the year 2100
#         year_2100 = np.array([[2100]])
#         predicted_sea_level = gb_model.predict(year_2100)[0]

#         print(f"Predicted sea level for the year 2100: {predicted_sea_level}")

#         return r2, predicted_sea_level
#     else:
#         print("The 'Year ' column is not present in the DataFrame.")


# def train_and_predict_ozone_hole_area():
#     url = r"data\antarctic-ozone-hole-area.csv"
#     df = pd.read_csv(url)

#     features = ["Year"]
#     targets = ["Maximum", "Mean"]

#     X = df[features]
#     y_max = df["Maximum"]
#     y_mean = df["Mean"]

#     X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(
#         X, y_max, test_size=0.3, random_state=4
#     )

#     gb_model_max = GradientBoostingRegressor(
#         n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
#     )
#     gb_model_max.fit(X_train_max, y_train_max)

#     y_pred_max = gb_model_max.predict(X_test_max)

#     mse_max = mean_squared_error(y_test_max, y_pred_max)
#     r2_max = r2_score(y_test_max, y_pred_max)

#     return r2_max, y_pred_max


# def train_and_predict_co2_emissions():
#     url = r"data\co-emissions-per-capita.csv"
#     df = pd.read_csv(url)

#     df_ind = df[df["Code"] == "IND"]

#     features = ["Year"]
#     target = "Annual"

#     X = df_ind[features]
#     y = df_ind[target]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     gb_model = GradientBoostingRegressor(
#         n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
#     )
#     gb_model.fit(X_train, y_train)

#     y_pred = gb_model.predict(X_test)

#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"Mean Squared Error: {mse}")
#     print(f"R-squared: {r2}")

#     year_2070 = np.array([[2090]])
#     predicted_emissions = gb_model.predict(year_2070)[0]

#     print(f"Predicted CO2 emissions per capita for 2070: {predicted_emissions}")

#     return r2, predicted_emissions


# def train_and_predict_temperature_models():
#     df = pd.read_csv("data\B_temp_modifies")
#     df["Year"] = pd.to_datetime(df["Year"]).dt.year

#     X = df[["Year"]]
#     y_max = df["Temp Max"]
#     y_min = df["Temp Min"]
#     y_avg = df["Temp Avg"]

#     (
#         X_train,
#         X_test,
#         y_max_train,
#         y_max_test,
#         y_min_train,
#         y_min_test,
#         y_avg_train,
#         y_avg_test,
#     ) = train_test_split(X, y_max, y_min, y_avg, test_size=0.2, random_state=42)

#     model_max = GradientBoostingRegressor(n_estimators=1000, random_state=42)
#     model_max.fit(X_train, y_max_train)

#     model_min = GradientBoostingRegressor(n_estimators=1000, random_state=42)
#     model_min.fit(X_train, y_min_train)

#     model_avg = GradientBoostingRegressor(n_estimators=1000, random_state=42)
#     model_avg.fit(X_train, y_avg_train)

#     year_2050 = pd.DataFrame({"Year": [2050]})
#     temp_max_2050 = model_max.predict(year_2050)
#     temp_min_2050 = model_min.predict(year_2050)
#     temp_avg_2050 = model_avg.predict(year_2050)

#     test_preds_max = model_max.predict(X_test)
#     test_preds_min = model_min.predict(X_test)
#     test_preds_avg = model_avg.predict(X_test)

#     test_mse_max = mean_squared_error(y_max_test, test_preds_max)
#     test_r2_max = r2_score(y_max_test, test_preds_max)

#     test_mse_min = mean_squared_error(y_min_test, test_preds_min)
#     test_r2_min = r2_score(y_min_test, test_preds_min)

#     test_mse_avg = mean_squared_error(y_avg_test, test_preds_avg)
#     test_r2_avg = r2_score(y_avg_test, test_preds_avg)

#     return {
#         "accuracy_max": 1 - test_mse_max,
#         "accuracy_min": 1 - test_mse_min,
#         "accuracy_avg": 1 - test_mse_avg,
#         "predictions_max": temp_max_2050[0],
#         "predictions_min": temp_min_2050[0],
#         "predictions_avg": temp_avg_2050[0],
#     }


# def train_and_predict_air_quality_model():
#     df = pd.read_excel("data\Air Qualtity Index.xlsx")
#     df = df.dropna(subset=["Year "])

#     X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
#     y = df[" Approximate AQI Category"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     model = SVC(kernel="linear", random_state=42)
#     model.fit(X_train_scaled, y_train)

#     predictions = model.predict(X_test_scaled)

#     accuracy = accuracy_score(y_test, predictions)

#     return {"accuracy": accuracy, "predictions": predictions}
#


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def multi_model_predictions(year_to_predict):
    def rainfall_prediction():
        df = pd.read_csv(r"data\rain.csv")
        df = df.dropna()

        years = df["Year"].values.reshape(-1, 1)
        total_rainfall = df["Total"].values.reshape(-1, 1)
        scaler_year = MinMaxScaler()
        scaler_total = MinMaxScaler()
        years_scaled = scaler_year.fit_transform(years)
        total_scaled = scaler_total.fit_transform(total_rainfall)
        X, y = [], []
        for i in range(len(years_scaled) - 1):
            X.append(np.hstack((years_scaled[i], total_scaled[i])))
            y.append(total_scaled[i + 1])
        X, y = np.array(X), np.array(y)
        gradient_boosting_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        gradient_boosting_model.fit(X, y.ravel())
        train_preds_scaled_gb = gradient_boosting_model.predict(X)
        train_preds_gb = scaler_total.inverse_transform(
            train_preds_scaled_gb.reshape(-1, 1)
        )
        train_mse_gb = mean_squared_error(total_rainfall[1:], train_preds_gb)
        train_r2_gb = r2_score(total_rainfall[1:], train_preds_gb)

        print(
            f"R-squared (R2) Score on Training Data (Gradient Boosting): {train_r2_gb}"
        )
        user_year = 2045
        user_input = np.array([[user_year, 0]])
        user_input_scaled = np.hstack(
            (
                scaler_year.transform(user_input[:, :1]),
                scaler_total.transform(user_input[:, 1:]),
            )
        )
        user_input_reshaped = user_input_scaled.reshape(1, -1)
        predicted_total_scaled = gradient_boosting_model.predict(user_input_reshaped)
        predicted_total = scaler_total.inverse_transform(
            predicted_total_scaled.reshape(-1, 1)
        )
        print(f"Predicted Total Rainfall for {user_year}: {predicted_total[0, 0] / 12}")

        acc = train_r2_gb
        value = predicted_total[0, 0] / 12
        return acc, value

    def train_and_predict_sea_level():
        url = r"data\Sea level.xlsx"
        sea_level_df = pd.read_excel(url)
        print(sea_level_df.columns)

        features = ["Year "]
        target = "Level"

        if "Year " in sea_level_df.columns:
            X = sea_level_df[features]
            y = sea_level_df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            gb_model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
            gb_model.fit(X_train, y_train)

            y_pred = gb_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"R-squared: {r2}")

            year_2100 = np.array([[2100]])
            predicted_sea_level = gb_model.predict(year_2100)[0]

            print(f"Predicted sea level for the year 2100: {predicted_sea_level}")

            return r2, predicted_sea_level
        else:
            print("The 'Year ' column is not present in the DataFrame.")

    def train_and_predict_ozone_hole_area():
        url = r"data\antarctic-ozone-hole-area.csv"
        df = pd.read_csv(url)

        features = ["Year"]
        targets = ["Maximum", "Mean"]

        X = df[features]
        y_max = df["Maximum"]
        y_mean = df["Mean"]

        X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(
            X, y_max, test_size=0.3, random_state=4
        )

        gb_model_max = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        gb_model_max.fit(X_train_max, y_train_max)

        y_pred_max = gb_model_max.predict(X_test_max)

        mse_max = mean_squared_error(y_test_max, y_pred_max)
        r2_max = r2_score(y_test_max, y_pred_max)

        return r2_max, y_pred_max

    def train_and_predict_co2_emissions():
        url = r"data\co-emissions-per-capita.csv"
        df = pd.read_csv(url)

        df_ind = df[df["Code"] == "IND"]

        features = ["Year"]
        target = "Annual"

        X = df_ind[features]
        y = df_ind[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        gb_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        gb_model.fit(X_train, y_train)

        y_pred = gb_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        year_2070 = np.array([[2090]])
        predicted_emissions = gb_model.predict(year_2070)[0]

        print(f"Predicted CO2 emissions per capita for 2070: {predicted_emissions}")

        return r2, predicted_emissions

    def train_and_predict_temperature_models():
        df = pd.read_csv("data\B_temp_modifies")
        df["Year"] = pd.to_datetime(df["Year"]).dt.year

        X = df[["Year"]]
        y_max = df["Temp Max"]
        y_min = df["Temp Min"]
        y_avg = df["Temp Avg"]

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

        model_max = GradientBoostingRegressor(n_estimators=1000, random_state=42)
        model_max.fit(X_train, y_max_train)

        model_min = GradientBoostingRegressor(n_estimators=1000, random_state=42)
        model_min.fit(X_train, y_min_train)

        model_avg = GradientBoostingRegressor(n_estimators=1000, random_state=42)
        model_avg.fit(X_train, y_avg_train)

        year_2050 = pd.DataFrame({"Year": [2050]})
        temp_max_2050 = model_max.predict(year_2050)
        temp_min_2050 = model_min.predict(year_2050)
        temp_avg_2050 = model_avg.predict(year_2050)

        test_preds_max = model_max.predict(X_test)
        test_preds_min = model_min.predict(X_test)
        test_preds_avg = model_avg.predict(X_test)

        test_mse_max = mean_squared_error(y_max_test, test_preds_max)
        test_r2_max = r2_score(y_max_test, test_preds_max)

        test_mse_min = mean_squared_error(y_min_test, test_preds_min)
        test_r2_min = r2_score(y_min_test, test_preds_min)

        test_mse_avg = mean_squared_error(y_avg_test, test_preds_avg)
        test_r2_avg = r2_score(y_avg_test, test_preds_avg)

        return {
            "accuracy_max": 1 - test_mse_max,
            "accuracy_min": 1 - test_mse_min,
            "accuracy_avg": 1 - test_mse_avg,
            "predictions_max": temp_max_2050[0],
            "predictions_min": temp_min_2050[0],
            "predictions_avg": temp_avg_2050[0],
        }

    def train_and_predict_air_quality_model():
        df = pd.read_excel("data\Air Qualtity Index.xlsx")
        df = df.dropna(subset=["Year "])

        X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
        y = df[" Approximate AQI Category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)

        return {"accuracy": accuracy, "predictions": predictions}

    # Call individual functions to get predictions and accuracies
    acc_rainfall, pred_rainfall = rainfall_prediction()
    acc_sea_level, pred_sea_level = train_and_predict_sea_level()
    acc_ozone, pred_ozone = train_and_predict_ozone_hole_area()
    acc_co2, pred_co2 = train_and_predict_co2_emissions()
    temp_predictions = train_and_predict_temperature_models()
    air_quality_predictions = train_and_predict_air_quality_model()

    # Print Accuracies
    print(f"Rainfall Prediction Accuracy: {acc_rainfall}")
    print(f"Sea Level Prediction Accuracy: {acc_sea_level}")
    print(f"Ozone Hole Area Prediction Accuracy: {acc_ozone}")
    print(f"CO2 Emissions Prediction Accuracy: {acc_co2}")
    print(f"Temperature Models Accuracy (Max): {temp_predictions['accuracy_max']}")
    print(f"Temperature Models Accuracy (Min): {temp_predictions['accuracy_min']}")
    print(f"Temperature Models Accuracy (Avg): {temp_predictions['accuracy_avg']}")
    print(f"Air Quality Model Accuracy: {air_quality_predictions['accuracy']}")

    accuracies = [
        acc_rainfall,
        acc_sea_level,
        acc_ozone,
        acc_co2,
        temp_predictions["accuracy_max"],
        temp_predictions["accuracy_min"],
        temp_predictions["accuracy_avg"],
        air_quality_predictions["accuracy"],
    ]

    # avg_accuracy = sum(accuracies) / len(accuracies)
    # print(f"Average Accuracy: {avg_accuracy}")

    # Return Predictions
    return {
        "Rainfall": pred_rainfall,
        "Sea Level": pred_sea_level,
        "Ozone Hole Area": pred_ozone,
        "CO2 Emissions": pred_co2,
        "Temperature Models (Max)": temp_predictions["predictions_max"],
        "Temperature Models (Min)": temp_predictions["predictions_min"],
        "Temperature Models (Avg)": temp_predictions["predictions_avg"],
        "Air Quality Model": air_quality_predictions["predictions"],
    }


# Example Usage
year_to_predict = 2045  # Replace with the desired year
predictions = multi_model_predictions(year_to_predict)
print(f"Predictions for {year_to_predict}:\n{predictions}")
for x in predictions:
    print(x, "----->", predictions[x])
