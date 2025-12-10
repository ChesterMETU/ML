"""
Senden NumPy kullanarak bu formülü koda dökmeni istiyorum. (İpucu: e sayısı için np.exp() fonksiyonunu kullanmalısın).
"""

import numpy as np
import matplotlib.pyplot as plt


# GÖREV: Sigmoid Fonksiyonunu Tamamla
def sigmoid(z):
    # Formül: 1 / (1 + e^(-z))
    # np.exp(-z) kullanarak yaz
    sonuc = 1 / (1 + np.exp(-z))
    return sonuc


# Test Edelim
z_degerleri = np.linspace(-10, 10, 100)  # -10 ile 10 arasında 100 sayı
y_degerleri = sigmoid(z_degerleri)

# Görselleştirme (S şeklini görmeliyiz)
plt.plot(z_degerleri, y_degerleri)
plt.title("Sigmoid Aktivasyon Fonksiyonu")
plt.grid(True)
plt.show()

# Kritik Test (x=0 iken y=0.5 olmalı)
print(f"Sigmoid(0) = {sigmoid(0)}")

"""
Senden, NumPy'ın np.log() fonksiyonunu kullanarak yukarıdaki formülü kodlamanı istiyorum.

Küçük bir mühendislik hilesi: np.log(0) tanımsızdır (eksi sonsuz). O yüzden genellikle p değerlerinin içine çok küçük bir sayı (epsilon: 1e-15) ekleriz ama şimdilik manuel değerlerle test edeceğimiz için gerek yok.
"""

import numpy as np

# 1. Veriler
y_gercek = 1
y_tahmin_iyi = 0.95  # Model emin: Bu bir Spam (1)
y_tahmin_kotu = 0.05  # Model emin: Bu Spam değil (0) -> Ama yanılıyor!


# 2. Log Loss Fonksiyonu
def log_loss(y, p):
    # Formül: - (y * log(p) + (1-y) * log(1-p))
    # np.log() kullan
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss


# 3. Test Et
loss_iyi = log_loss(y_gercek, y_tahmin_iyi)
loss_kotu = log_loss(y_gercek, y_tahmin_kotu)

print(f"İyi Tahmin Cezası: {loss_iyi:.4f}")
print(f"Kötü Tahmin Cezası: {loss_kotu:.4f}")

"""
Artık teorik engel kalmadı. Elindeki Spam Dedektörünü eğitme vakti.

Senaryo: Elimizde kelime sayılarına göre Spam olup olmadığını bildiğimiz 4 mesaj var.

    Veri (X): Mesajdaki "Bedava" kelimesi sayısı.

        [1, 5, 10, 50] (Az geçenler normal, çok geçenler spam gibi)

    Etiket (Y): 0 (Normal) veya 1 (Spam).

        [0, 0, 1, 1]

Senden Sigmoid, Log Loss ve Gradient Descent parçalarını birleştirmeni istiyorum.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Veri Seti
X = np.array([1, 5, 10, 50])  # "Bedava" kelime sayısı
Y = np.array([0, 0, 1, 1])  # 0: Normal, 1: Spam

# 2. Parametreler
w = 0.0  # Başlangıçta ağırlık yok
b = 0.0
lr = 0.01
epochs = 2000


# Yardımcı Fonksiyon: Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


loss_history = []

print("Eğitim Başlıyor...")

# 3. Eğitim Döngüsü
for i in range(epochs):
    # --- ADIM 1: İleri Yayılım (Forward Pass) ---
    # a. Z'yi hesapla (Lineer kısım)
    z = w * X + b

    # b. p'yi hesapla (Sigmoid ile olasılığa çevir)
    p = sigmoid(z)

    # c. Loss hesapla (Log Loss - Sadece izlemek için)
    # Epsilon (1e-15) ekliyoruz ki log(0) hatası almayalım
    loss = -np.mean(Y * np.log(p + 1e-15) + (1 - Y) * np.log(1 - p + 1e-15))
    loss_history.append(loss)

    # --- ADIM 2: Geri Yayılım (Backward Pass) ---
    # Türev Formülü: X * (p - y) / N
    hata_farki = p - Y
    N = len(X)

    dw = (1 / N) * np.dot(
        X, hata_farki
    )  # Dot product kullanıyoruz çünkü X ve hata_farki vektör
    db = np.mean(hata_farki)

    # --- ADIM 3: Güncelleme ---
    # w ve b'yi güncelle (Gradient Descent)
    w = w - (lr * dw)
    b = b - (lr * db)

    if i % 500 == 0:
        print(f"Epoch {i}: Loss {loss:.4f}")

print(f"\nFinal Modeli: w={w:.4f}, b={b:.4f}")

# Test Edelim: 20 tane "Bedava" kelimesi geçen mesaj spam mı?
test_mesaj = 20
test_olasilik = sigmoid(w * test_mesaj + b)
print(f"20 'Bedava' kelimeli mesajın Spam olma ihtimali: %{test_olasilik*100:.2f}")
