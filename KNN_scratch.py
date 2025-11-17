from collections import Counter
import numpy as np


class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_Test):
        y_pred = [self._predict(x) for x in X_Test]
        return np.array(y_pred)

    def _predict(self, x):
        # (x2​−x1​)^2+(y2​−y1​)^2
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        k_nearest_indices = np.argsort(distances)[: self.k]

        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

knn_model = KNNFromScratch(k=20)
knn_model.fit(X_train, y_train)

predictions = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"My KNN Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.neighbors import KNeighborsClassifier

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
sklearn_predictions = sklearn_knn.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
print(f"Scikit-Learn KNN Model Accuracy: {sklearn_accuracy * 100:.2f}%")
