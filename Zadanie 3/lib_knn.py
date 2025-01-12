import numpy as np


class AlgorithmKNN:
    trainingSet = []  # Zbiór treingowy
    testSet = []  # Zbiór testowy
    distanceMatrix = []  # Tablica odległości obiektów training od test
    trainingSetNormalized = []  # Zbiór treingowy znormalizowany (bez klas)
    testSetNormalized = []  # Zbiór testowy znormalizowany (bez klas)
    expectedClasses = []    # Wyniki klasyfikacji przeprowadzonej przez program

    def __init__(self, trainingData):
        self.trainingSet = trainingData
        self.trainingSetNormalized = self.normalize(self.trainingSet[:, :-1])

    def loadTestData(self, testData):
        self.testSet = testData
        self.testSetNormalized = self.normalize(self.testSet[:, :-1])

    def knn(self, k):
        # krok 1.
        self.calculateDistances(self.trainingSetNormalized, self.testSetNormalized)

        # krok 2.
        lowestInColumns = self.findMin(self.distanceMatrix, k)

        # krok 3.
        # Teraz trzeba wyłuskać kolumny i wiersze najniższych odległości
        # Zapisać je w zmiennej żeby móc wyjąć sobie z trainingSet kolumny z klasami
        # Zobaczyć których jest najwięcej i przypisać klasy dla obiektów testowych

        classesToSelect = self.fromDistanceToIndex(lowestInColumns)
        self.expectedClasses = self.classesSelector(classesToSelect, k)
        return

    def calculateDistances(self, A, B):
        # TODO: Do przejrzenia i zrozumienia oraz ewnetualnej optymalizacji
        A_suma_kwadratow = np.sum(A ** 2, axis=1).reshape(-1, 1)  # Wektor (m x 1)
        B_suma_kwadratow = np.sum(B ** 2, axis=1).reshape(1, -1)  # Wektor (1 x n)

        # Odległość euklidesowa: sqrt(|A|^2 + |B|^2 - 2*A*B.T)
        macierz_odleglosci = np.sqrt(A_suma_kwadratow + B_suma_kwadratow - 2 * np.dot(A, B.T))

        self.distanceMatrix = macierz_odleglosci

    def fromDistanceToIndex(self, distances):
        listOfObjectIndexes = []
        for column, i in enumerate(distances):
            sublist = []
            for value, index in i:
                sublist.append([column, int(self.trainingSet[index][4]), float(value)])
                # print(f"Obiekt testowy nr: {column}   Klasa: {self.testSet[column][4]} <-> {self.trainingSet[index][4]}")
                # print(f"Obiekt testowy nr: {column} -> {self.trainingSet[index][4]}")
            listOfObjectIndexes.append(sublist)

        return listOfObjectIndexes

    def visualize(self):
        return

    def missesMatrix(self):
        return

    def accuracy(self):
        classification_accuracy = 0
        for number, category in enumerate(self.expectedClasses):
            if category == self.testSet[number][4]:
                classification_accuracy += 1

        return round((classification_accuracy / len(self.testSet))*100, 2)

    @staticmethod
    def normalize(x):
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def classesSelector(classes, n):
        result = []

        for i in range(len(classes)):
            votes = []
            for j in range(n):
                votes.append([classes[i][j][1], classes[i][j][2]])

            #print(votes)

            #   Liczymy nasze "głosy" korzystając ze słownika
            counter = {}
            weights = {}
            for vote in votes:
                if vote[0] in counter:
                    counter[vote[0]] += 1
                    weights[vote[0]] += float(1 / vote[1] + 1e-9)
                else:
                    counter[vote[0]] = 1
                    weights[vote[0]] = float(1 / vote[1] + 1e-9)

            #print(weights)
            #print(counter)

            winner = None
            numberOfVotes = 0
            for value, number in counter.items():
                if number > numberOfVotes:
                    numberOfVotes = number
                    winner = value

                if number == numberOfVotes:
                    if weights[value] > weights[winner]:
                        numberOfVotes = number
                        winner = value

            result.append(winner)

        #print(result)
        return result

    @staticmethod
    def findMin(macierz, n):
        # TODO: Przetłumaczyc wszystkie zmienne na ang
        wynik = []
        for j in range(macierz.shape[1]):  # Iterujemy po kolumnach
            kolumna = macierz[:, j]  # Pobieramy kolumnę
            najmniejsze = []  # Lista na n najmniejszych wartości
            odwiedzone = set()  # Indeksy już odwiedzone, by uniknąć powtórzeń

            # n oznacza nasze k obiektów do wybrania. Pętla przejdzie po kolumnie tyle razy ile mamy wybrać najmnieszych wartości z kolumny
            for _ in range(n):
                # Inicjalizujemy wartość minimalną (2.0 dlatego że po normalizacji największą wartością będzie sqrt(2))
                min_wartosc = float(2.0)
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
