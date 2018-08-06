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

# Split the dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the dataset
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fit the regression model

# Predict with regression model
# y_pred = regressor.predict(6.5)

# Visualize regression model
plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff (Regressor Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualize regression model (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
# plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff (Smoother Regressor Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
