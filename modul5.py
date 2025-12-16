import numpy as np

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


# 1. VERİ SETİ (4 örnek, 3 özellik)
X = np.array([[10, 2, 5], [50, 4, 10], [20, 1, 2], [100, 5, 1]])
print(f"Girdi Boyutu: {X.shape}")  # (4, 3)

# 2. AĞIRLIKLAR VE BIAS (Rastgele başlatalım)
# Gizli Katman (5 nöron)
input_dim = 3
hidden_dim = 5
output_dim = 1

# W1: Girdiyi (3) alıp Gizliye (5) götürecek
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))

# W2: Gizliyi (5) alıp Çıktıya (1) götürecek
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))


# 3. YARDIMCI FONKSİYONLAR
def relu(z):
    # np.maximum kullanarak ReLU yaz
    return np.maximum(0, z)  # --- KODU DOLDUR ---


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 4. FORWARD PASS (Ağın içinden geçiş)

# ADIM A: Gizli Katman Hesaplaması
# Z1 = X . W1 + b1
Z1 = np.dot(X, W1) + b1
# A1 = ReLU(Z1) -> Aktivasyon fonksiyonundan geçir
print(f"z1 = {Z1} ")  # --- KODU DOLDUR ---
A1 = relu(Z1)  # --- KODU DOLDUR ---
print("\nA1 (ReLU çıktı):\n", A1)

print(f"Gizli Katman Çıktı Boyutu: {A1.shape}")  # (4, 5) olmalı

# ADIM B: Çıktı Katmanı Hesaplaması
# Z2 = A1 . W2 + b2 (Dikkat: Giriş artık A1 oldu)
Z2 = np.dot(A1, W2) + b2  # --- KODU DOLDUR ---
# Tahmin = Sigmoid(Z2)
tahmin = sigmoid(Z2)  # --- KODU DOLDUR ---

print(f"Final Tahmin Boyutu: {tahmin.shape}")  # (4, 1) olmalı
print("\nTahminler:\n", tahmin)
