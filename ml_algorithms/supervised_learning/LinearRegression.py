import numpy as np
from sklearn.preprocessing import add_dummy_feature
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta = []  # weights

    def fit(self, X, y):
        X_b = add_dummy_feature(X)  # add x0 = 1 to each instance

        # Normal Equation
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X_new):
        X_new_b = add_dummy_feature(X_new)
        return X_new_b @ self.theta


lin_reg = LinearRegression()
m = 100  # number of examples

np.random.seed(42)  # same random number each run
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

lin_reg.fit(X, y)

new_X = np.array([[0], [2]])
y_predicted = lin_reg.predict(new_X)

print(y_predicted)

plt.plot(X, y, 'b.')
plt.plot(new_X, y_predicted, 'r-', label="predictions")
plt.legend()
plt.grid()
plt.show()
