import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Funkcja do wczytywania danych
def wczytaj_dane(plik):
    """
    Wczytuje dane z pliku CSV w formacie:
    cecha1, cecha2, ..., cechaN, klasa
    """
    dane = np.loadtxt(plik, delimiter=',')
    X = dane[:, :-1]  # Wszystkie kolumny oprócz ostatniej to cechy
    y = dane[:, -1].astype(int)  # Ostatnia kolumna to klasy
    return X, y

# Funkcja do normalizacji danych
def normalizuj_dane(train_X, test_X):
    """
    Normalizuje dane cech w zbiorze treningowym i testowym do przedziału [0, 1].
    """
    X_min = train_X.min(axis=0)
    X_max = train_X.max(axis=0)
    train_X_norm = (train_X - X_min) / (X_max - X_min)
    test_X_norm = (test_X - X_min) / (X_max - X_min)
    return train_X_norm, test_X_norm

# Funkcja obliczająca odległości euklidesowe
def oblicz_odleglosci(x, train_X):
    """
    Oblicza odległość euklidesową między punktem x a wszystkimi punktami w train_X.
    """
    return np.sqrt(np.sum((train_X - x) ** 2, axis=1))

# Algorytm k-NN
def knn(train_X, train_y, test_X, k):
    """
    Klasyfikuje punkty testowe przy użyciu algorytmu k-NN.
    """
    przewidywane = []
    for x in test_X:
        # Oblicz odległości do wszystkich punktów treningowych
        odleglosci = oblicz_odleglosci(x, train_X)
        # Znajdź k najbliższych sąsiadów
        sasiedzi_idx = np.argsort(odleglosci)[:k]
        sasiedzi_klasy = train_y[sasiedzi_idx]
        # Głosowanie
        klasy, liczby = np.unique(sasiedzi_klasy, return_counts=True)
        przewidywane.append(klasy[np.argmax(liczby)])
    return np.array(przewidywane)

# Funkcja obliczająca skuteczność
def skutecznosc(y_true, y_pred):
    """
    Oblicza procent poprawnych klasyfikacji.
    """
    return np.sum(y_true == y_pred) / len(y_true)

# Funkcja testująca skuteczność dla różnych wartości k
def testuj_k(train_X, train_y, test_X, test_y, k_range):
    """
    Testuje skuteczność k-NN dla różnych wartości k.
    """
    wyniki = []
    for k in k_range:
        przewidywane = knn(train_X, train_y, test_X, k)
        wynik = skutecznosc(test_y, przewidywane)
        wyniki.append(wynik)
    return wyniki

# Wizualizacja skuteczności dla różnych k
def rysuj_wykres_k(k_range, wyniki):
    """
    Rysuje wykres skuteczności dla różnych wartości k.
    """
    plt.plot(k_range, wyniki, marker='o')
    plt.title('Skuteczność klasyfikacji dla różnych wartości k')
    plt.xlabel('k (liczba sąsiadów)')
    plt.ylabel('Skuteczność')
    plt.grid()
    plt.show()

# Rysowanie macierzy pomyłek
def rysuj_macierz_pomylek(y_true, y_pred, klasy):
    """
    Rysuje macierz pomyłek dla prawdziwych i przewidywanych klas.
    """
    cm = confusion_matrix(y_true, y_pred, labels=klasy)
    plt.imshow(cm, cmap='Blues')
    plt.title('Macierz pomyłek')
    plt.colorbar()
    plt.xlabel('Przewidywane klasy')
    plt.ylabel('Prawdziwe klasy')
    plt.xticks(range(len(klasy)), klasy)
    plt.yticks(range(len(klasy)), klasy)
    plt.show()

# Główna funkcja programu
def main():
    # Wczytaj dane
    train_X, train_y = wczytaj_dane('data3_train.csv')  # Zbiór treningowy
    test_X, test_y = wczytaj_dane('data3_test.csv')    # Zbiór testowy

    # Normalizacja danych
    train_X, test_X = normalizuj_dane(train_X, test_X)

    # Testowanie dla k = 1, ..., 15
    k_range = range(1, 16)
    wyniki = testuj_k(train_X, train_y, test_X, test_y, k_range)

    # Rysowanie wykresu skuteczności
    rysuj_wykres_k(k_range, wyniki)

    # Najlepsze k
    najlepsze_k = k_range[np.argmax(wyniki)]
    print(f'Najlepsze k: {najlepsze_k}')

    # Macierz pomyłek dla najlepszego k
    najlepsze_przewidywania = knn(train_X, train_y, test_X, najlepsze_k)
    klasy = np.unique(np.concatenate((train_y, test_y)))
    rysuj_macierz_pomylek(test_y, najlepsze_przewidywania, klasy)

# Uruchomienie programu
if __name__ == "__main__":
    main()
