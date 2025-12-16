import numpy as np

import numpy as np

# Girdi: 3 Özellikli (Sütunlar: x1, x2, x3)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Hedef: Sınıflandırma (0 veya 1)
# İpucu: İlk iki sütuna dikkat et (XOR mantığı), 3. sütun yanıltıcı olabilir.
Y = np.array([[0], [1], [1], [0]])

input = 2
hidden = 4
output = 1


np.random.seed(42)
w1 = np.random.randn(input, hidden)
b1 = np.zeros((1, hidden))

np.random.seed(24)
w2 = np.random.randn(hidden, output)
b2 = np.zeros((1, output))

print("Başlangıç Ağırlıkları ve Biaslar:")
print("w1:\n", w1)
print("b1:\n", b1)
print("w2:\n", w2)
print("b2:\n", b2)

lr = 0.01
epochs = 6000


def Relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for i in range(epochs):
    # Forward Pass
    z1 = np.dot(X, w1) + b1
    h1 = Relu(z1)

    z2 = np.dot(h1, w2) + b2
    h2 = sigmoid(z2)

    loss = -np.mean(Y * np.log(h2 + 1e-15) + (1 - Y) * np.log(1 - h2 + 1e-15))

    # Backward Pass
    dz2 = h2 - Y
    dw2 = np.dot(h1.T, dz2) / len(X)

    dz1 = np.dot(dz2, w2.T) * (h1 > 0)
    dw1 = np.dot(X.T, dz1) / len(X)

    db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

    # update parameters

    w1 = w1 - (lr * dw1)
    w2 = w2 - (lr * dw2)

    b1 = b1 - (lr * db1)
    b2 = b2 - (lr * db2)

    if i % 1000 == 0:
        print(f"Loss: {loss} in epoch {i}")
