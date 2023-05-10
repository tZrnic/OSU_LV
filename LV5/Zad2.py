import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

labels = {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1Min, x1Max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2Min, x2Max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x11, x22 = np.meshgrid(np.arange(x1Min, x1Max, resolution),
    np.arange(x2Min, x2Max, resolution))
    Z = classifier.predict(np.array([x11.ravel(), x22.ravel()]).T)
    Z = Z.reshape(x11.shape)
    
    plt.contourf(x11, x22, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x11.min(), x11.max())
    plt.ylim(x22.min(), x22.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

df = pd.read_csv("penguins.csv")
print(df.isnull().sum())
df = df.drop(columns=['sex'])
df.dropna(axis=0, inplace=True)
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)
print(df.info())

output_variable = ['species']
input_variables = ['bill_length_mm', 'flipper_length_mm']
X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)