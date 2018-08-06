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
y = dataset.iloc[:, 2].values

# Fit the decision tree regression model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict with decision tree polynomial model
y_pred = regressor.predict(6.5)

# Visualize decision tree regression model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff (Smoother Decision Tree Regressor Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
