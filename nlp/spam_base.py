# course: https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://archive.ics.uci.edu/ml/datasets/Spambase

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('data/spambase/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

# last 100 rows for test everything else is for train
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

# naive bayes classifier
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))

# ada-boost classifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest))
