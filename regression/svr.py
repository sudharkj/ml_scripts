# From the course: Machine Learning A-Z™: Hands-On Python & R In Data Science
# https://www.udemy.com/machinelearning/
# dataset: https://www.superdatascience.com/machine-learning/

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Scale the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y).ravel()

# Fit the SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predict with SVR model
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualize SVR model
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualize SVR model (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
