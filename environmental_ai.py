print("hello, ai world!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import StringIO
csv_data = """
date,PM2.5,PM10,NO2,SO2,CO,AQI
2023-01-01,55,120,30,12,0.6,140
2023-01-02,60,110,35,15,0.7,150
2023-01-03,70,130,38,16,0.9,165
2023-01-04,80,140,40,18,1.0,175
2023-01-05,50,100,25,10,0.5,120
2023-01-06,65,125,32,13,0.8,155
2023-01-07,75,135,37,17,0.95,170
"""
data = pd.read_csv(StringIO(csv_data))
print("Dataset loaded:")
print(data.head())
X = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model trained.")
predictions = model.predict(X_test)
print("\n Predictions vs Actual:")
for i in range(len(predictions)):
    print(f"Predicted: {round(predictions[i], 1)} | Actual: {y_test.iloc[i]}")

plt.plot(range(len(y_test)), y_test.values, label='Actual AQI', marker='o')
plt.plot(range(len(predictions)), predictions, label='Predicted AQI', marker='x')
plt.title('Actual vs Predicted AQI')
plt.xlabel('Sample')
plt.ylabel('AQI')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from io import StringIO

air_csv = """
date,PM2.5,PM10,NO2,SO2,CO,AQI
2023-01-01,55,120,30,12,0.6,140
2023-01-02,60,110,35,15,0.7,150
2023-01-03,70,130,38,16,0.9,165
2023-01-04,80,140,40,18,1.0,175
2023-01-05,50,100,25,10,0.5,120
2023-01-06,65,125,32,13,0.8,155
2023-01-07,75,135,37,17,0.95,170
"""

air_data = pd.read_csv(StringIO(air_csv))
X_air = air_data[['PM2.5','PM10','NO2','SO2','CO']]
y_air = air_data['AQI']

X_train, X_test, y_train, y_test = train_test_split(X_air, y_air, test_size=0.3, random_state=42)
air_model = RandomForestRegressor(n_estimators=50, random_state=42)
air_model.fit(X_train, y_train)

pred_air = air_model.predict(X_test)
print(" Air Quality Predictions:")
for i in range(len(pred_air)):
    print(f"Predicted: {round(pred_air[i],1)} | Actual: {y_test.iloc[i]}")

plt.scatter(y_test, pred_air)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Air Quality Prediction")
plt.grid(True)
plt.show()

water_csv = """
pH,DO,Turbidity,WQI
7.2,6.5,3.0,80
7.5,7.0,2.5,85
6.8,6.0,4.0,75
7.0,6.8,3.5,78
7.4,7.2,2.8,83
6.9,6.3,3.7,76
7.1,6.7,3.2,79
"""

water_data = pd.read_csv(StringIO(water_csv))
X_water = water_data[['pH','DO','Turbidity']]
y_water = water_data['WQI']

X_train, X_test, y_train, y_test = train_test_split(X_water, y_water, test_size=0.3, random_state=42)
water_model = RandomForestRegressor(n_estimators=50, random_state=42)
water_model.fit(X_train, y_train)

pred_water = water_model.predict(X_test)
print("\n Water Quality Predictions:")
for i in range(len(pred_water)):
    print(f"Predicted: {round(pred_water[i],1)} | Actual: {y_test.iloc[i]}")

plt.scatter(y_test, pred_water)
plt.xlabel("Actual WQI")
plt.ylabel("Predicted WQI")
plt.title("Water Quality Prediction")
plt.grid(True)
plt.show()
soil_csv = """
Nitrogen,Phosphorus,Potassium,pH,OM,Soil_Quality_Index
0.3,0.2,0.25,6.8,3.0,75
0.35,0.25,0.3,7.0,3.5,80
0.28,0.22,0.27,6.9,2.8,72
0.32,0.24,0.29,7.1,3.2,78
0.31,0.23,0.28,6.95,3.0,76
0.33,0.26,0.31,7.05,3.3,79
"""

soil_data = pd.read_csv(StringIO(soil_csv))
X_soil = soil_data[['Nitrogen','Phosphorus','Potassium','pH','OM']]
y_soil = soil_data['Soil_Quality_Index']

X_train, X_test, y_train, y_test = train_test_split(X_soil, y_soil, test_size=0.3, random_state=42)
soil_model = RandomForestRegressor(n_estimators=50, random_state=42)
soil_model.fit(X_train, y_train)

pred_soil = soil_model.predict(X_test)
print("\n Soil Quality Predictions:")
for i in range(len(pred_soil)):
    print(f"Predicted: {round(pred_soil[i],1)} | Actual: {y_test.iloc[i]}")

plt.scatter(y_test, pred_soil)
plt.xlabel("Actual Soil Quality")
plt.ylabel("Predicted Soil Quality")
plt.title("Soil Quality Prediction")
plt.grid(True)
plt.show()
