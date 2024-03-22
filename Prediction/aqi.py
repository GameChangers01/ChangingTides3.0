# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your dataset
# df = pd.read_excel("data\Air Qualtity Index.xlsx")

# # Assuming your data has columns: 'Year', 'Estimated Avg PM10 Levels (μg/m3)', 'Approximate AQI Category'

# # Drop rows with missing values in the 'Year' column
# df = df.dropna(subset=["Year "])

# # Reshape the data
# X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
# y = df[" Approximate AQI Category"]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Create and train the SVM model
# model = SVC(
#     kernel="linear", random_state=42
# )  # You can adjust the kernel type (linear, rbf, etc.)
# model.fit(X_train_scaled, y_train)

# # Make predictions on the test set
# predictions = model.predict(X_test_scaled)


# print(predictions)

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on Test Set: {accuracy:.2f}")


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load your dataset
df = pd.read_excel("data\Air Qualtity Index.xlsx")

df = df.dropna(subset=["Year "])


X = df[["Year ", " Estimated Avg PM10 Levels (μg/m3) "]]
y = df[" Estimated Avg PM10 Levels (μg/m3) "]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVR(kernel="linear")  #
model.fit(X_scaled, y)


input_year = 2021
input_pm10_level = 0
input_data_scaled = scaler.transform([[input_year, input_pm10_level]])
predicted_pm10_level = model.predict(input_data_scaled)[0]
print(
    f"Predicted PM10 Level for the year {input_year}: {predicted_pm10_level:.2f} μg/m3"
)
