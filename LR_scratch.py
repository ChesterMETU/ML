import numpy as np


class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        y = y.reshape(-1, 1)

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # 3b: Gradyanları Hesapla (Kayıp fonksiyonunun türevleri)
            # dw = (2/n) * X.T * (y_pred - y)
            # db = (2/n) * sum(y_pred - y)
            error = y_predicted - y
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X, y = make_regression(n_samples=100, n_features=3, noise=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions.flatten())
print(f"My Model MSE: {mse:.4f}")

# Karşılaştırma için Scikit-Learn modeline de bakalım
from sklearn.linear_model import LinearRegression

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)
print(f"Scikit-Learn MSE: {sklearn_mse:.4f}")
