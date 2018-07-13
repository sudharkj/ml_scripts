# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
import numpy as np

from scipy import stats

N = 10
a = np.random.randn(N) + 2
b = np.random.randn(N)

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)
s = np.sqrt((var_a + var_b) / 2)
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N))
df = 2*N - 2
p = 1 - stats.t.cdf(t, df=df)
print("t:\t", t, "\tp:\t", 2*p)

t2, p2 = stats.ttest_ind(a, b)
print("t:\t", t2, "\tp:\t", p2)
