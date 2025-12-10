# Vector
import numpy as np

# 1. Verileri NumPy array olarak tanımla
x = np.array([120, 3, 10])
w = np.array([0.5, 20, -1])
b = 10

# 2. Dot Product işlemini yap (np.dot kullan) ve bias ekle
tahmin = np.dot(x, w) + b

print(f"Evin Tahmini Fiyatı: {tahmin}")

# Matris batch

x = np.array([[120, 3, 10], [80, 2, 5], [200, 4, 20]])
w = np.array([0.5, 20, -1])
b = 10

tahmin = np.dot(x, w) + b
print(f"Evin Tahmini Fiyatları: {tahmin}")
