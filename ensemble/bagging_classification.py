# From the course: Ensemble Machine Learning in Python: Random Forest, AdaBoost
# Link: https://www.udemy.com/machine-learning-in-python-random-forest-adaboost/

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from .util import plot_decision_boundary


np.random.seed(10)

# create the data
N = 500
D = 2
X = np.random.randn(N, D)

# noisy XOR
sep = 2
X[:125] += np.array([sep, sep])
X[125:250] += np.array([sep, -sep])
X[250:375] += np.array([-sep, -sep])
X[375:] += np.array([-sep, sep])
Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125)

# plot the data
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

model = DecisionTreeClassifier()
model.fit(X, Y)
print("score for 1 tree:", model.score(X, Y))

# plot the data
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plot_decision_boundary(X, model)
plt.show()


class BaggedTreeClassifier:
    def __init__(self, B):
        self.B = B

    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            model = DecisionTreeClassifier(max_depth=2)
            model.fit(Xb, Yb)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return np.round(predictions / self.B)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


if __name__ == '__main__':

    model = BaggedTreeClassifier(200)
    model.fit(X, Y)

    print("score for bagged model:", model.score(X, Y))

    # plot the data
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plot_decision_boundary(X, model)
    plt.show()
