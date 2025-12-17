import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Veriyi Yükle
data = load_breast_cancer()
X_org = data.data  # Özellikler (569 örnek, 30 özellik)
Y_org = data.target  # Etiketler (569 örnek,)

# 2. Şekil Düzenleme (Shape Fix)
# Y şu an (569,) şeklinde, bize (569, 1) lazım. Yoksa matris çarpımı patlar.
Y_org = Y_org.reshape(-1, 1)

# 3. Eğitim ve Test Olarak Ayır (%80 Eğitim, %20 Test)
# Modeli X_train ile eğiteceğiz, X_test ile başarısını ölçeceğiz.
X_train, X_test, Y_train, Y_test = train_test_split(
    X_org, Y_org, test_size=0.2, random_state=42
)

# 4. Ölçekleme (Standardization) - ÇOK KRİTİK!
# Verilerin ortalamasını 0, standart sapmasını 1 yapar.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class nnModule:
    def __init__(self, X_train, X_test, Y_train, Y_test, hiddenDim, lr):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.inputDim = X_train.shape[1]
        self.hiddenDim = hiddenDim
        self.outputDim = 1
        np.random.seed(42)
        self.w1 = np.random.randn(self.inputDim, self.hiddenDim)
        self.w2 = np.random.randn(self.hiddenDim, self.outputDim)
        self.b1 = np.zeros((1, self.hiddenDim))
        self.b2 = np.zeros((1, self.outputDim))
        self.loss = 0
        self.z1 = 0
        self.h1 = 0
        self.z2 = 0
        self.h2 = 0
        self.lr = lr

    def Relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculateLoss(self, p):
        return -np.mean(
            self.Y_train * np.log(p + 1e-15)
            + (1 - self.Y_train) * np.log(1 - p + 1e-15)
        )

    def forwardPass(self):
        self.z1 = np.dot(self.X_train, self.w1) + self.b1
        self.h1 = self.Relu(self.z1)

        self.z2 = np.dot(self.h1, self.w2) + self.b2
        self.h2 = self.sigmoid(self.z2)

        self.loss = self.calculateLoss(self.h2)

        return self.loss

    def backwardPass(self):
        dz2 = self.h2 - self.Y_train
        dw2 = np.dot(self.h1.T, dz2) / len(self.X_train)

        dz1 = np.dot(dz2, self.w2.T) * (self.h1 > 0)
        dw1 = np.dot(self.X_train.T, dz1) / len(self.X_train)

        db1 = np.sum(dz1, axis=0, keepdims=True) / len(self.X_train)
        db2 = np.sum(dz2, axis=0, keepdims=True) / len(self.X_train)

        self.w1 = self.w1 - (self.lr * dw1)
        self.w2 = self.w2 - (self.lr * dw2)
        self.b1 = self.b1 - (self.lr * db1)
        self.b2 = self.b2 - (self.lr * db2)

    def caculateAccuracy(self):
        z1_test = np.dot(self.X_test, self.w1) + self.b1
        h1_test = self.Relu(z1_test)

        z2_test = np.dot(h1_test, self.w2) + self.b2
        h2_test = self.sigmoid(z2_test)

        predictions = (h2_test > 0.5).astype(int)
        accuracy = np.mean(predictions == self.Y_test)

        return accuracy


print(f"Eğitim Verisi: {X_train.shape}")  # (455, 30) olmalı
print(f"Test Verisi: {X_test.shape}")  # (114, 30) olmalı

lr = 0.01
epochs = 10000

nn = nnModule(X_train, X_test, Y_train, Y_test, 16, lr)

for i in range(epochs):
    # Forward Pass
    loss = nn.forwardPass()

    # Backward Pass
    nn.backwardPass()

    if i % 2000 == 0:
        print(f"Loss: {loss}")
        print("--------------")

# Calculate Test Accuracy
accuracy = nn.caculateAccuracy()

print(f"Test Doğruluğu: {accuracy * 100:.2f}%")
