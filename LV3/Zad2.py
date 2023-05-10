import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist')
plt.xlabel(('CO2 emission'))
plt.ylabel(('Number of cars'))
plt.show()

emissions=data['CO2 Emissions (g/km)']
fuelC=data['Fuel Consumption City (L/100km)']
data["Fuel Color"] = data["Fuel Type"].map(
    {
        "X": "Red",
        "Z": "Green",
        "D": "Blue",
        "E": "Pink",
        "N": "Orange",
    }
)

plt.scatter(fuelC, emissions,c=data['Fuel Color'])
plt.show()


data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
data.groupby("Fuel Type")["Cylinders"].count().plot(kind="bar", ax=ax1)


data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar", ax=ax2)
fig.subplots_adjust(hspace=0.5)
plt.show()
