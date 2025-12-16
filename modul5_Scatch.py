"""
Şimdi senden, hiçbir eğitim yapmadan, sadece 2 katmanlı bir sinir ağının yapısını (mimarisini) NumPy ile kurmanı istiyorum.

Bu görev, Matris Boyutlarını (Shapes) anlama sınavıdır. En çok burada hata yapılır.

Senaryo:

    Girdi (X): 4 örnek veri, her birinin 3 özelliği var. (Örn: Evin m2, oda sayısı, yaşı).

        Shape: (4, 3)

    Gizli Katman: 5 tane nöron olsun istiyoruz.

        Ağırlık Matrisi W1​ boyutu ne olmalı? → (3, 5)
    Çıktı Katmanı: Tek bir sonuç (Fiyat) istiyoruz.

        Ağırlık Matrisi W2​ boyutu ne olmalı? → (5, 1)

Görev: Aşağıdaki kodda boşlukları doldurarak veriyi ağın içinden geçir.
"""

import numpy as np

hiddenDim = 5
inputDim = 3
outputDim = 1

X = np.array([[10, 2, 5], [50, 4, 10], [20, 1, 2], [100, 5, 1]])

W1 = np.random.rand(inputDim, hiddenDim)
b1 = np.zeros((1, hiddenDim))

W2 = np.random.rand(hiddenDim, outputDim)
b2 = np.zeros((1, outputDim))

lr = 0.01
epochs = 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Relu(z):
    return np.maximum(0, z)


for i in range(epochs):
    z1 = np.dot(X, W1) + b1
    h1 = Relu(z1)

    z2 = np.dot(z1, W2)
    h2 = sigmoid(z2)

print(h2)
