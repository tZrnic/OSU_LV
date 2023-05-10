import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print(len(data))
print (data.info())
data.dropna(axis =0)
data.drop_duplicates()
data = data.reset_index(drop=True)

for col in data:
    if(type(col)==object):
        data[col]=data[col].astype('Category')

#b)
consumption=data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False)[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
print("Highest fuel consumption\n", consumption.head(3))
print("Highest fuel consumption\n", consumption.tail(3))

#c)
engineSize = data[(data['Engine Size (L)']>=2.5)&(data['Engine Size (L)']<=3.5)]
print(len(engineSize))
print("Average Co2 for engine size 2.5 - 3.5: ", engineSize['CO2 Emissions (g/km)'].mean())

#d)
audi = data[(data['Make']=='Audi')]
print("Audi quantity: ", len(audi))
audi = audi[(audi['Cylinders']==4)]
print("Average 4cyl audi emissions: ", audi['CO2 Emissions (g/km)'].mean())

#e)
cyl = data[(data['Cylinders']%2==0)]
print("Pair cylinders: ", len(cyl))
print("Emissions: ", cyl['CO2 Emissions (g/km)'].mean())

#f)
diesel = data[(data['Fuel Type']=='D')]
print("Diesel consumption: ", diesel['Fuel Consumption City (L/100km)'].mean())
gasoline = data[(data['Fuel Type']=='X')]
print("Gasoline consumption: ", gasoline['Fuel Consumption City (L/100km)'].mean())

#g)
vehicle = data[(data['Cylinders']==4)&(data['Fuel Type']=='D')]
print("Max consumption: ", vehicle.sort_values(by='Fuel Consumption City (L/100km)').head(1))

#h)
manual = data[(data['Transmission'].str.startswith('M'))]
print("Number of manuals: ", len(manual))

#i)
print (data.corr(numeric_only = True))