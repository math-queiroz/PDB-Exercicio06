import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('winequality-red.csv')
ind = dataset.iloc[:, :-1].values
dep = dataset.iloc[:, -1].values

decisionTreeRegressor = DecisionTreeRegressor()
decisionTreeRegressor.fit(ind, dep)

plt.scatter(ind[:, -1], decisionTreeRegressor.predict(ind), color='red')
plt.xlabel('Álcool')
plt.ylabel('Qualidade')
plt.title('Álcool x Qualidade')
plt.show()

fig = plt.figure()
subplot = fig.add_subplot(111, projection='3d')
subplot.scatter(ind[:, -1], ind[:, -4], decisionTreeRegressor.predict(ind), color='red')
plt.show()