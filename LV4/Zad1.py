import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn . linear_model as lm
from sklearn . metrics import mean_absolute_error

data = pd.read_csv('data_C02_emission.csv')
data = data.drop(["Make", "Model"], axis=1)

input_variables = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Engine Size (L)',
                   'Cylinders']

output_variable = ['CO2 Emissions (g/km)']
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)
f1 = plt.figure("Scatter ulaza")
plt.scatter(X_train[:,1], y_train, c="r")
plt.scatter(X_test[:,1], y_test, c="b")

sc = MinMaxScaler ()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

fig2, axes2 = plt.subplots(2)
axes2[0].hist(X_test[:,0])
axes2[1].hist(X_test_n[:,0])

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , y_train)

print(linearModel.coef_)

f3=plt.figure("Scatter izlaza")
y_test_p = linearModel.predict( X_test_n )
plt.scatter(y_test, y_test_p)
plt.show()

MAE = mean_absolute_error (y_test , y_test_p)