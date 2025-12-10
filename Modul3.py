"""
İlk işimiz, üzerinde çalışacağımız sahte bir "emlak verisi" oluşturmak ve henüz eğitilmemiş modelimizin ne kadar kötü tahmin yaptığını görmek.

Görev:

    NumPy ile basit bir veri seti oluştur.

    Rastgele bir w ve b belirle.

    Bu rastgele değerlerle tahmin yap ve MSE hatasını hesapla.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. VERİ SETİ (X: Metrekare, Y: Fiyat)
# Gerçek kural (Bilgisayar bunu bilmiyor, bulmaya çalışacak): y = 3x + 10
X = np.array([1, 2, 3, 4, 5])  # Evlerin büyüklüğü
Y = np.array([13, 19, 25, 31, 37])  # Evlerin gerçek fiyatları

# 2. PARAMETRELER (Rastgele başlangıç)
w = 1.0  # Rastgele bir ağırlık (Yanlış tahmin)
b = 1.0  # Rastgele bir bias (Yanlış tahmin)

# 3. TAHMİN (Forward Pass)
# Lineer Formül: tahmin = w * X + b
# NumPy broadcasting sayesinde tek satırda tüm X'ler için hesaplar
tahminler = (
    X * w + b
)  # np.dot(X, w) + b = X*w + b / np.dot -> vector x vector = scalar, np.dot -> vector x scalar = vector

# 4. HATA HESAPLA (MSE - Mean Squared Error)
# Formül: Ortalama( (Gerçek - Tahmin)^2 )
# İpucu: np.mean() fonksiyonunu kullan
hata = np.sum((Y - tahminler) ** 2) / len(X)

print(f"Rastgele w={w}, b={b} ile Hata (MSE): {hata}")

# Görselleştirme
plt.scatter(X, Y, color="blue", label="Gerçek Veri")
plt.plot(X, tahminler, color="red", label="Model Tahmini (Eğitimsiz)")
plt.legend()
plt.show()

"""
Şimdi parçaları birleştirip yapay zekayı eğitelim. Aşağıdaki kodda # --- GÜNCELLEME --- kısmını doldurmanı istiyorum.

Mantık, Modül 2'deki "Kör Dağcı" ile birebir aynı: yeni_değer = eski_değer - (learning_rate * egim)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Veri
X = np.array([1, 2, 3, 4, 5])
Y = np.array([13, 19, 25, 31, 37])  # Hedef: y = 6x + 7

# 2. Parametreler (Rastgele Başlangıç)
w = 0.0
b = 0.0

lr = 0.01  # Learning Rate (Öğrenme Hızı)
epochs = 1000  # Adım sayısı

N = len(X)
history_loss = []

# 3. EĞİTİM
for i in range(epochs):
    # a. Forward Pass (Tahmin)
    tahminler = w * X + b

    # b. Loss (Sadece izlemek için)
    loss = np.mean((Y - tahminler) ** 2)  # np.sum((Gerçek - Tahmin)**2)/len(X)
    history_loss.append(loss)

    # c. Gradient Hesaplama (Türev Formülleri)
    # Hata farkı: (Gerçek - Tahmin) -> Bu bize yönü gösterir
    hata_farki = Y - tahminler

    # Matematiksel Türev Formülleri (Zincir kuralından gelir)
    dw = (-2 / N) * np.sum(X * hata_farki)
    db = (-2 / N) * np.sum(hata_farki)

    # --- GÖREV: GÜNCELLEME ---
    # w ve b değerlerini güncelle (Gradient Descent)
    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print(f"Epoch {i}: Loss {loss:.4f} | w: {w:.2f} b: {b:.2f}")

print(f"\nEğitim Bitti! Bulunan Değerler -> w: {w:.2f}, b: {b:.2f}")
print(f"Gerçek Değerler olması gereken -> w: 6.00, b: 7.00")

# Sonucu çizdir
plt.scatter(X, Y, label="Gerçek")
plt.plot(X, w * X + b, color="red", label="Yapay Zeka Tahmini")
plt.legend()
plt.show()
