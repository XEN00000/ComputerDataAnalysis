#       Tworzymy klasę która będzie rozwiązywać całe zadanie
#       Nasza klasa będzie miała następujące rozwiązania:
#       -> W konstruktorze przyjmnie od razu zbiór treningowy
#       -> Metodę kitóra przyjmie zbiór testowy  zapełni tablicę odległości obiektów ze zbioru treningowego od obiektów ze zbioru testowego
#       -> Metodę normalizującą dane (robimy to po to aby bardzo duże liczby np. na osi x nie zakłamywały wyniku gdy na osi y znajdą się bardzo małe np. Wykres zarobków i wieku osób)
#       -> Metodę wyświetlającą wykres z rozwiązaniem
#       -> Metodę wyświetlającą macierz pomyłek

import numpy as np


class AlgorithmKNN:
    trainingSet = []  # Zbiór treingowy
    testSet = []  # Zbiór testowy
    distanceMatrix = []  # Tablica odległości obiektów training od test
    trainingSetNormalized = []  # Zbiór treingowy znormalizowany (bez klas)
    testSetNormalized = []  # Zbiór testowy znormalizowany (bez klas)

    def __init__(self, trainingData):
        self.trainingSet = trainingData
        self.trainingSetNormalized = self.normalize(self.trainingSet[:, :-1])

    @classmethod
    def loadTestData(cls, testData):
        cls.testSet = testData
        cls.testSetNormalized = cls.normalize(cls.testSet[:, :-1])

    @classmethod
    def knn(cls, k):
        # krok 1.
        cls.calculateDistances(cls.trainingSetNormalized, cls.testSetNormalized)
        print(cls.distanceMatrix)

        # krok 2.
        lowestInColumns = cls.findMin(cls.distanceMatrix, k)

        # krok 3.
        # Teraz trzeba wyłuskać kolumny i wiersze najniższych odległości
        # Zapisać je w zmiennej żeby móc wyjąć sobie z trainingSet kolumny z klasami
        # Zobaczyć których jest najwięcej i przypisać klasy dla obiektów testowych

    @classmethod
    def __calculateDistances(cls, A, B):
        #TODO: Do przejrzenia i zrozumienia oraz ewnetualnej optymalizacji
        A_suma_kwadratow = np.sum(A ** 2, axis=1).reshape(-1, 1)  # Wektor (m x 1)
        B_suma_kwadratow = np.sum(B ** 2, axis=1).reshape(1, -1)  # Wektor (1 x n)

        # Odległość euklidesowa: sqrt(|A|^2 + |B|^2 - 2*A*B.T)
        macierz_odleglosci = np.sqrt(A_suma_kwadratow + B_suma_kwadratow - 2 * np.dot(A, B.T))

        cls.distanceMatrix = macierz_odleglosci

    @classmethod
    def __normalize(cls, x):
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        return (x - x_min) / (x_max - x_min)

    @classmethod
    def __findMin(cls, macierz, n):
        # TODO: Przetłumaczyc wszystkie zmienne na ang
        wynik = []
        for j in range(macierz.shape[1]):  # Iterujemy po kolumnach
            kolumna = macierz[:, j]  # Pobieramy kolumnę
            najmniejsze = []  # Lista na n najmniejszych wartości
            odwiedzone = set()  # Indeksy już odwiedzone, by uniknąć powtórzeń

            # n oznacza nasze k obiektów do wybrania. Pętla przejdzie po kolumnie tyle razy ile mamy wybrać najmnieszych wartości z kolumny
            for _ in range(n):
                min_wartosc = float(
                    2.0)  # Inicjalizujemy wartość minimalną (2.0 dlatego że po normalizacji największą wartością będzie sqrt(2))
                min_indeks = -1

                # Przeszukujemy całą kolumnę, aby znaleźć najmniejszy element
                for i, wartosc in enumerate(kolumna):
                    if i not in odwiedzone and wartosc < min_wartosc:
                        min_wartosc = wartosc
                        min_indeks = i

                # Dodajemy znaleziony element do listy wyników
                if min_indeks != -1:
                    najmniejsze.append((float(min_wartosc), min_indeks))
                    odwiedzone.add(min_indeks)

            wynik.append(najmniejsze)
        return wynik

    @classmethod
    def visualize(cls):
        return

    @classmethod
    def missesMatrix(cls):
        return
