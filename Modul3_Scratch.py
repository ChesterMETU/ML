import numpy as np


lr = 0.01
w = 0.0
b = 0.0
epochs = 2000

X = np.array([1, 2, 3, 4, 5])
Y = np.array([13, 19, 25, 31, 37])


for i in range(epochs):
    predicted = X * w + b

    loss = np.mean(np.sum(Y - predicted) ** 2)

    diff = Y - predicted

    dw = -2 * np.mean(np.sum(X * diff))
    db = -2 * np.mean(np.sum(diff))

    w = w - lr * dw
    b = b - lr * db

    if i % 500 == 0:
        print(f"Epoch {i}: Loss {loss:.4f}")

print(f"Final w = {w} and Final b = {b}")

test_Data = 6
predicted = test_Data * w + b
print(f"Predicted value for {test_Data} is {predicted}")
