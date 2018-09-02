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

# Fit the linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fit the polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualize linear model
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or bluff (Linear Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualize polynomial model
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.plot(X_poly, lin_reg_2.predict(X_poly), color='blue')
plt.title("Truth or bluff (Polynomial Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predict with linear model
print(lin_reg.predict(6.5))

# Predict with polynomial model
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))
