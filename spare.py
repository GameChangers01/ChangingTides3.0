# from flask import Flask, render_template, url_for, request
# import numpy as np
# import tensorflow as tf
# import os
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     RandomForestRegressor,
#     AdaBoostRegressor,
#     VotingRegressor,
# )
# from sklearn.svm import SVR
# from sklearn.svm import SVR
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     RandomForestRegressor,
#     VotingRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import VotingRegressor
# from xgboost import XGBRegressor
# from sklearn.svm import SVR
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.svm import SVR
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score


# app = Flask(__name__, static_url_path="/static")


# # index page
# @app.route("/")
# def index():
#     return render_template("index.html")


# # SignUp page
# @app.route("/signup")
# def signup():
#     return render_template("Signup.html")


# # Login page
# @app.route("/login")
# def login():
#     return render_template("Login.html")


# @app.route("/detect")
# def detect():
#     return render_template("detect.html")


# @app.route("/contactus")
# def contactus():
#     return render_template("contactus.html")


# @app.route("/result", methods=["POST", "GET"])
# def result():
#     if request.method == "POST":
#         ans = ""
#         max_t = 49
#         min_t = -2
#         avg_t = 30
#         rain = 50
#         aqi = 12
#         co2 = 0.5
#         sealevel = 5
#         ozone = 10000
#         city = request.form["city"]
#         year = request.form["year"]
#         ans = city + year
#         if city == "Bangalore":
#             # aqi = predict_air("AQI_prediction_model.pkl", year)
#             # sealevel = predict_sea("sealevel_stacking_model.pkl", year)
#             aqi = air_prediction(year)
#             co2 = co2_emissions(year)
#             max_t, min_t = predict_temp(year)
#             sealevel = predict_sea_level(year)
#             rain = predict_total_rainfall_for_year(year)
#             ozone = predict_maximum_ozone(year)

#         elif city == "Hyderabad":
#             pass

#     return render_template(
#         "result.html",
#         max_t=max_t,
#         min_t=min_t,
#         rain=rain,
#         aqi=aqi,
#         co2=co2,
#         sealevel=sealevel,
#         ozone=ozone,
#         year=year,
#     )


# @app.route("/home")
# def home():
#     return render_template("home.html")


# def predict_pm10_level(year):
#     # Load your dataset
#     df = pd.read_excel("data\Air Qualtity Index.xlsx")

#     df = df.dropna(subset=["Year "])

#     X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
#     y = df[" Estimated Avg PM10 Levels (μg/m3) "]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = SVR(kernel="linear")
#     model.fit(X_scaled, y)

#     input_pm10_level = 0

#     input_data_scaled = scaler.transform([[year, input_pm10_level]])
#     predicted_pm10_level = model.predict(input_data_scaled)[0]

#     return predicted_pm10_level


# def predict_air(model_path, year):
#     with open(model_path, "rb") as file:
#         loaded_model = pickle.load(file)
#     input_data = np.array([[year]])
#     scaler = StandardScaler()
#     input_data_scaled = scaler.fit_transform(input_data)
#     prediction = loaded_model.predict(input_data_scaled)
#     prediction_inv = scaler.inverse_transform(prediction)
#     adjusted_prediction = prediction_inv[0, 0] / 11
#     return adjusted_prediction


# def predict_sea(model_path, year):
#     with open(model_path, "rb") as file:
#         loaded_model = pickle.load(file)
#     input_data = np.array([[year]])
#     predicted_sea_level = loaded_model.predict(
#         pd.DataFrame({"Year": input_data.flatten()})
#     )
#     adjusted_prediction = predicted_sea_level + 50
#     return adjusted_prediction[0]


# def co2_emissions(year_to_predict):
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
#     year_to_predict = np.array([[year_to_predict]])
#     predicted_emissions = gb_model.predict(year_to_predict)[0]
#     return predicted_emissions


# def predict_temp(year):
#     # Load the dataset
#     url = r"data\Temp_exp1.xlsx"
#     df = pd.read_excel(url)

#     # Extract year from 'Date' column
#     df["Year"] = pd.to_datetime(df["Year"]).dt.year

#     # Select features and target variables
#     X = df[["Year"]]
#     y_max = df["Temp Max"]
#     y_min = df["Temp Min"]

#     # Split the data into training and testing sets
#     (
#         X_train,
#         X_test,
#         y_max_train,
#         y_max_test,
#         y_min_train,
#         y_min_test,
#     ) = train_test_split(X, y_max, y_min, test_size=0.2, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Instantiate individual models for Temp Max
#     gb_model_max = GradientBoostingRegressor(n_estimators=10000, random_state=42)
#     rf_model_max = RandomForestRegressor(n_estimators=10000, random_state=42)
#     svm_model_max = SVR(kernel="linear")
#     lr_model_max = LinearRegression()
#     nn_model_max = MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=5000)
#     ensemble_max = VotingRegressor(
#         [
#             ("gb", gb_model_max),
#             ("rf", rf_model_max),
#             ("svm", svm_model_max),
#             ("lr", lr_model_max),
#             ("nn", nn_model_max),
#         ]
#     )

#     ensemble_max.fit(X_train_scaled, y_max_train)

#     year_scaled = scaler.transform(np.array([[year]]))
#     predicted_temp_max = ensemble_max.predict(year_scaled)[0] - 10

#     gb_model_min = GradientBoostingRegressor(n_estimators=500, random_state=42)
#     rf_model_min = RandomForestRegressor(n_estimators=500, random_state=42)
#     svm_model_min = SVR(kernel="linear")
#     lr_model_min = LinearRegression()
#     nn_model_min = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
#     ensemble_min = VotingRegressor(
#         [
#             ("gb", gb_model_min),
#             ("rf", rf_model_min),
#             ("svm", svm_model_min),
#             ("lr", lr_model_min),
#             ("nn", nn_model_min),
#         ]
#     )
#     ensemble_min.fit(X_train_scaled, y_min_train)
#     predicted_temp_min = ensemble_min.predict(year_scaled)[0]

#     return predicted_temp_min, predicted_temp_max


# def air_prediction(year):
#     df = pd.read_excel("data\AQI.xlsx")
#     df = df.dropna(subset=["Year "])
#     X = df["Year "].values.reshape(-1, 1)
#     y = df["Estimated PM10 (μg/m3)"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1, random_state=42
#     )
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     model = keras.Sequential(
#         [
#             layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
#             layers.Dropout(0.2),
#             layers.Dense(128, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(64, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(32, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(1),
#         ]
#     )

#     model.compile(optimizer="adam", loss="mean_squared_error")
#     model.fit(X_train_scaled, y_train, epochs=100000, batch_size=32, verbose=1)
#     year_array = np.array([[year]])
#     year_scaled = scaler.transform(year_array)
#     prediction = model.predict(year_scaled)
#     prediction_inv = scaler.inverse_transform(prediction)
#     return prediction_inv[0, 0] / 12


# def predict_sea_level(year):
#     url = r"data\Sea level (1).xlsx"
#     sea_level_df = pd.read_excel(url)
#     features = ["Year "]
#     target = "Level"
#     if "Year " in sea_level_df.columns:
#         X = sea_level_df[features]
#         y = sea_level_df[target]
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#         models = [
#             (
#                 "gb",
#                 GradientBoostingRegressor(
#                     n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
#                 ),
#             ),
#             (
#                 "rf",
#                 RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42),
#             ),
#             (
#                 "xgb",
#                 XGBRegressor(
#                     n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
#                 ),
#             ),
#             ("svm", SVR(kernel="rbf")),
#             (
#                 "mlp",
#                 MLPRegressor(
#                     hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
#                 ),
#             ),
#             ("hist_gb", HistGradientBoostingRegressor(max_iter=500, random_state=42)),
#         ]

#         stacking_model = StackingRegressor(
#             estimators=models, final_estimator=LinearRegression(), cv=5
#         )

#         stacking_model.fit(X_train, y_train)
#         year_array = np.array([[year]])
#         future_prediction = stacking_model.predict(
#             pd.DataFrame({"Year ": year_array.flatten()})
#         )[0]

#         print(f"Year: {year}, Predicted Sea Level: {future_prediction}")

#         return future_prediction


# def predict_total_rainfall_for_year(year):
#     df = pd.read_csv("data\rain (1).csv")
#     features = df[
#         [
#             "Jan",
#             "Feb",
#             "Mar",
#             "Apr",
#             "May",
#             "June",
#             "July",
#             "Aug",
#             "Sept",
#             "Oct",
#             "Nov",
#             "Dec",
#         ]
#     ]
#     imputer = SimpleImputer(strategy="mean")
#     features_imputed = imputer.fit_transform(features)
#     scaler = MinMaxScaler()
#     features_scaled = scaler.fit_transform(features_imputed)
#     df[
#         [
#             "Jan",
#             "Feb",
#             "Mar",
#             "Apr",
#             "May",
#             "June",
#             "July",
#             "Aug",
#             "Sept",
#             "Oct",
#             "Nov",
#             "Dec",
#         ]
#     ] = features_scaled
#     target_total = df["Total"]
#     X_train, X_test, y_train_total, y_test_total = train_test_split(
#         features_scaled, target_total, test_size=0.5, random_state=42
#     )

#     model = keras.Sequential(
#         [
#             layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
#             layers.Dropout(0.2),  # Add dropout layer with a dropout rate of 20%
#             layers.Dense(64, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(32, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(1),
#         ]
#     )

#     model.compile(optimizer="adam", loss="mean_squared_error")

#     model.fit(X_train, y_train_total, epochs=100, batch_size=32, verbose=1)
#     future_features = scaler.transform(
#         imputer.transform(df.loc[:, "Jan":"Dec"])
#     )  # Impute and scale
#     year_index = df.columns.get_loc("Jan") + (year - df["Year"].min()) * 12
#     future_features_year = future_features[year_index].reshape(1, -1)
#     future_total_prediction_nn = model.predict(future_features_year)[0, 0]
#     print(f"Year: {year}, Predicted Total Rainfall: {future_total_prediction_nn/12}")
#     return future_total_prediction_nn / 12


# def predict_maximum_ozone(year):
#     # Load the dataset
#     url = r"data\Air Qualtity Index.xlsx"
#     df = pd.read_csv(url)
#     features = ["Year"]
#     targets = ["Maximum"]
#     X = df[features]
#     y_max = df["Maximum"]
#     X_train_max, X_test_max, y_train_max, y_test_max = train_test_split(
#         X, y_max, test_size=0.2, random_state=4
#     )
#     scaler = StandardScaler()
#     X_train_max_scaled = scaler.fit_transform(X_train_max)
#     X_test_max_scaled = scaler.transform(X_test_max)
#     gb_model_max = GradientBoostingRegressor(
#         n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42
#     )
#     rf_model_max = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
#     ada_model_max = AdaBoostRegressor(
#         n_estimators=500, learning_rate=0.1, random_state=42
#     )
#     svm_model_max = SVR(kernel="linear")
#     lr_model_max = LinearRegression()
#     ensemble_model_max = VotingRegressor(
#         estimators=[
#             ("gb", gb_model_max),
#             ("rf", rf_model_max),
#             ("ada", ada_model_max),
#             ("svm", svm_model_max),
#             ("lr", lr_model_max),
#         ]
#     )
#     ensemble_model_max.fit(X_train_max_scaled, y_train_max)
#     year_array = np.array([[year]])
#     year_scaled = scaler.transform(year_array)
#     future_prediction_max = ensemble_model_max.predict(year_scaled)[0]
#     print(f"Year: {year}, Predicted Maximum Ozone Hole Area: {future_prediction_max}")
#     return future_prediction_max


# if __name__ == "__main__":
#     app.run(debug=True)
