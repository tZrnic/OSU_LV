import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, max_error

data = pd.read_csv("data_C02_emission.csv")
data = data.drop(["Make", "Model"], axis=1)

input_variables = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders',
                   'Fuel Type']
output_variable = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder()
x_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type']=x_encoded

x = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2, random_state=1)
linearModel = lm.LinearRegression()
linearModel.fit(x_train,y_train)

y_test_p = linearModel.predict(x_test)
print("R2 score: ", r2_score(y_test, y_test_p))
print("MAE: ", mean_absolute_error(y_test, y_test_p))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_test_p))
print("MSE: ", mean_squared_error(y_test, y_test_p))
print("Max error: ", max_error(y_test, y_test_p))