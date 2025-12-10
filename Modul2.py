"""
Biz bilgisayarda "sonsuz küçük" diye bir şey yapamayız. O yüzden h'yi elle küçük bir sayı (0.0001) verip türevi "taklit edeceğiz".

Buna Nümerik Diferansiyel denir.

Kodlama Görevi:

    Bir fonksiyon tanımla: f(x)=x^2

    Türev fonksiyonu yaz:

        Fonksiyonun x'teki değeri: f(x)

        Fonksiyonun biraz ilerisindeki (x+h) değeri: f(x + 0.0001)

        Farkı bul ve h'ye böl.
"""


def f(x):
    # Fonksiyonumuz: y = x^2
    return x**2


def turev_al(x):
    h = 0.0001  # Çok küçük bir adım (limit hilesi)

    # Formül: (f(x+h) - f(x)) / h
    # Burayı sen dolduracaksın 👇
    egim = (f(x + h) - f(x)) / h

    return egim


# Test edelim (x=3 noktasında türev 6 çıkmalı)
print(f"x=3 noktasında eğim: {turev_al(3)}")

"""
Şimdi senden, yazdığın turev_al fonksiyonunu kullanarak bir döngü kurmanı istiyorum. Bilgisayar adım adım x=10 noktasından x=0 noktasına kendi kendine inecek.

Senaryo:

    Başlangıç noktası: current_x = 10

    Learning Rate: learning_rate = 0.1

    100 kere çalışacak bir for döngüsü kur.

Döngünün içinde yapılacaklar:

    Şu anki x noktasındaki türevi hesapla (turev_al fonksiyonunu kullan).

    current_x değerini güncelle: current_x = current_x - (learning_rate * egim)

    Her 10 adımda bir ekrana current_x değerini yazdır ki inişi izleyelim.

Bu kod bittiğinde, current_x değeri 0'a (veya 0.00000...1 gibi çok küçük bir sayıya) ulaşmış olmalı.
"""

current_x = 10.0
learning_rate = 0.1

for i in range(100):
    gx = turev_al(current_x)
    current_x = current_x - (learning_rate * gx)
    # print(f"Egim: {gx}")
    print(f"Current x = {current_x:.4f}")
    print("-------------------------")

"""
Artık türev almayı ve bir değeri minimize etmeyi biliyorsun. Şimdi bunu tek bir sayı (x) için değil, iki sayı (x ve y) için yapacağız.

Senaryo: 3 Boyutlu bir arazideyiz. Fonksiyonumuz bir kase (Bowl) şeklinde:
z=f(x,y)=x^2+y^2

Amacımız bu kasenin en dibini (x=0,y=0) bulmak. Ama bilgisayar başlangıçta rastgele bir yerde, mesela x=10,y=10'da.

Görev: Aşağıdaki kodda eksik olan türev alma ve güncelleme kısımlarını doldurmanı istiyorum. İpucu: x için yaptığının aynısını y için de yapacaksın. İkisi birbirinden bağımsızdır (Kısmi Türev mantığı).
"""

import numpy as np
import matplotlib.pyplot as plt


# Fonksiyonumuz (Loss Function)
def cost_function(x, y):
    return x**2 + y**2


# Türev Fonksiyonu (Hem X hem Y için eğimi hesapla)
def gradient(x, y):
    h = 0.0001

    # 1. X'e göre türev (Y sabit kalır)
    # Formül: (f(x+h, y) - f(x, y)) / h
    grad_x = (cost_function(x + h, y) - cost_function(x, y)) / h

    # 2. Y'ye göre türev (X sabit kalır)
    # Formül: (f(x, y+h) - f(x, y)) / h
    grad_y = (cost_function(x, y + h) - cost_function(x, y)) / h

    return grad_x, grad_y


# Başlangıç Noktası (Dağın tepesi)
current_x = 10.0
current_y = 10.0
learning_rate = 0.1

# Tarihçeyi tutalım (Grafik çizmek için)
history_x = []
history_y = []
history_cost = []

# --- EĞİTİM DÖNGÜSÜ ---
print("Eğitim Başlıyor...")

for i in range(50):
    # Kayıt al
    history_x.append(current_x)
    history_y.append(current_y)
    history_cost.append(cost_function(current_x, current_y))

    # 1. Eğimleri hesapla
    gx, gy = gradient(current_x, current_y)

    # 2. Güncelle (Gradient Descent Formülü)
    # x_yeni = x_eski - (lr * egim_x)
    current_x = current_x - (learning_rate * gx)
    current_y = current_y - (learning_rate * gy)

print(f"Final Konum: x={current_x:.4f}, y={current_y:.4f}")
print(f"Final Hata (Cost): {cost_function(current_x, current_y):.4f}")

# --- GÖRSELLEŞTİRME (Kodu değiştirme, sadece çalıştır) ---
plt.figure(figsize=(10, 6))
plt.plot(history_cost)
plt.title("Hata (Loss) Grafiği - Dib'e İniş")
plt.xlabel("Adım Sayısı (Epoch)")
plt.ylabel("Hata (Cost)")
plt.grid(True)
plt.show()
