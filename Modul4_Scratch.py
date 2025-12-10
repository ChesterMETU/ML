import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


lr = 0.01
w = 0.0
b = 0.0

X = np.array([1, 5, 10, 50])
Y = np.array([0, 0, 1, 1])

epochs = 1000

for i in range(epochs):
    z = X * w + b

    p = sigmoid(z)

    loss = -np.mean(Y * np.log(p + 1e-15) + (1 - Y) * np.log(1 - p + 1e-15))

    diff = p - Y

    dw = np.mean(np.dot(X, diff))
    db = np.mean(diff)

    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print(f"Epoch {i}: Loss {loss:.4f}")

print(f"\nFinal Modeli: w={w:.4f}, b={b:.4f}")

test_mesaj = 20
test_olasilik = sigmoid(w * test_mesaj + b)
print(f"20 'Bedava' kelimeli mesajın Spam olma ihtimali: %{test_olasilik*100:.2f}")
