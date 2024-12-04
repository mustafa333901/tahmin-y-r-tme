import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import json
import csv

class AdvancedPredictor:
    def __init__(self):
        self.data_file = "model_data.json"
        self.load_data()
        self.train_model()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
                self.X = np.array(data["X"])
                self.y = np.array(data["y"])
        else:
            self.X = np.array([[1], [2], [3], [4], [5]])
            self.y = np.array([2, 4, 6, 8, 10])
            self.save_data()

    def save_data(self):
        data = {"X": self.X.tolist(), "y": self.y.tolist()}
        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        self.score = r2_score(self.y, self.model.predict(self.X))

    def predict(self, value):
        return self.model.predict([[value]])[0]

    def update_data(self, new_x, new_y):
        self.X = np.append(self.X, [[new_x]], axis=0)
        self.y = np.append(self.y, new_y)
        self.train_model()
        self.save_data()

    def reset_model(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([2, 4, 6, 8, 10])
        self.train_model()
        self.save_data()

    def export_predictions(self, predictions, filename="predictions.csv"):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Input", "Prediction"])
            writer.writerows(predictions)
        print(f"Tahminler '{filename}' dosyasına kaydedildi.")

def plot_predictions(predictions):
    x = [item[0] for item in predictions]
    y = [item[1] for item in predictions]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', label='Tahminler')
    plt.title("Tahmin Grafiği")
    plt.xlabel("Giriş Değerleri")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    predictor = AdvancedPredictor()
    predictions = []

    print("Gelişmiş Tahmin Modeline Hoş Geldiniz!")
    print("Bu modelle tahmin yapabilir, yeni veri ekleyebilir ve modeli analiz edebilirsiniz.")

    while True:
        print("\n--- Menü ---")
        print("1. Tahmin yap")
        print("2. Modele yeni veri ekle")
        print("3. Toplu veri ekle")
        print("4. Model doğruluğunu görüntüle")
        print("5. Tahmin grafiği görüntüle")
        print("6. Tahmin verilerini dışa aktar")
        print("7. Modeli sıfırla")
        print("8. Çıkış")

        choice = input("Bir seçenek girin (1-8): ").strip()
        if choice == "8":
            print("Programdan çıkılıyor. İyi günler!")
            break

        if choice == "1":
            try:
                value = float(input("Tahmin yapmak için bir sayı girin: "))
                prediction = predictor.predict(value)
                predictions.append((value, prediction))
                print(f"{value} için tahmin edilen değer: {prediction:.2f}")
            except ValueError:
                print("Geçerli bir sayı girin!")

        elif choice == "2":
            try:
                new_x = float(input("Yeni giriş değeri (x): "))
                new_y = float(input("Yeni sonuç değeri (y): "))
                predictor.update_data(new_x, new_y)
                print("Yeni veri eklendi ve model güncellendi!")
            except ValueError:
                print("Geçerli sayılar girin!")

        elif choice == "3":
            try:
                num_points = int(input("Kaç adet veri eklemek istiyorsunuz? "))
                for _ in range(num_points):
                    new_x = float(input("Yeni giriş değeri (x): "))
                    new_y = float(input("Yeni sonuç değeri (y): "))
                    predictor.update_data(new_x, new_y)
                print("Toplu veriler eklendi ve model güncellendi!")
            except ValueError:
                print("Geçerli sayılar girin!")

        elif choice == "4":
            print(f"Modelin doğruluk (R²) skoru: {predictor.score:.2f}")

        elif choice == "5":
            if predictions:
                plot_predictions(predictions)
            else:
                print("Henüz bir tahmin yapılmadı, önce tahmin yapın.")

        elif choice == "6":
            try:
                file_name = input("Tahminlerin kaydedileceği dosya adı (örn. predictions.csv): ").strip()
                predictor.export_predictions(predictions, file_name)
            except Exception as e:
                print(f"Dosya kaydedilirken hata oluştu: {e}")

        elif choice == "7":
            confirm = input("Modeli sıfırlamak istediğinize emin misiniz? (Evet/Hayır): ").strip().lower()
            if confirm in ["evet", "e", "yes", "y"]:
                predictor.reset_model()
                predictions = []
                print("Model sıfırlandı!")
        else:
            print("Geçersiz bir seçenek girdiniz, lütfen tekrar deneyin.")

if __name__ == "__main__":
    main()
