import numpy as np
import matplotlib.pyplot as plt

import lib_knn as knn


# Import danych do programu
data_train = np.loadtxt('data3_train.csv', delimiter=',')
data_test = np.loadtxt('data3_test.csv', delimiter=',')

#+---------------------------------------------------------------------------------+
#|                            Dla wszystkich cech irysów                           |
#+---------------------------------------------------------------------------------+
print("\n\n---------------------------Dla wszystkich cech irysów---------------------------\n")


# Utworzenie obiektu AlgorithmKNN, i załadowanie do niego danych
knnSolver = knn.AlgorithmKNN(data_train, data_test)

k_values = list(range(1, 16))
globalAccuracy = []

for i in range(1, 16):
    knnSolver.knn(i)
    acc = knnSolver.accuracy()
    globalAccuracy.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver.missesMatrix()

# 2. Tworzenie wykresu
plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla wszystkich cech w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(95, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()


#+---------------------------------------------------------------------------------+
#|                           Dla sepal length i sepal width                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla sepal length i sepal width---------------------------\n")


knnSolver01 = knn.AlgorithmKNN(data_train[:, [0, 1, -1]], data_test[:, [0, 1, -1]])

globalAccuracy01 = []

for i in range(1, 16):
    knnSolver01.knn(i)
    acc = knnSolver01.accuracy()
    globalAccuracy01.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver01.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy01, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla sepal length i sepal width w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(68, 76, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()

#+---------------------------------------------------------------------------------+
#|                          Dla sepal length i petal length                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla sepal length i petal length---------------------------\n")


knnSolver02 = knn.AlgorithmKNN(data_train[:, [0, 2, -1]], data_test[:, [0, 2, -1]])

globalAccuracy02 = []

for i in range(1, 16):
    knnSolver02.knn(i)
    acc = knnSolver02.accuracy()
    globalAccuracy02.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver02.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy02, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla sepal length i petal length w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(88, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()

#+---------------------------------------------------------------------------------+
#|                           Dla sepal length i petal width                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla sepal length i petal width---------------------------\n")


knnSolver03 = knn.AlgorithmKNN(data_train[:, [0, 3, -1]], data_test[:, [0, 3, -1]])

globalAccuracy03 = []

for i in range(1, 16):
    knnSolver03.knn(i)
    acc = knnSolver03.accuracy()
    globalAccuracy03.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver03.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy03, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla sepal length i petal width w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(93, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()

#+---------------------------------------------------------------------------------+
#|                           Dla sepal width i petal length                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla sepal width i petal length---------------------------\n")


knnSolver04 = knn.AlgorithmKNN(data_train[:, [1, 2, -1]], data_test[:, [1, 2, -1]])

globalAccuracy04 = []

for i in range(1, 16):
    knnSolver04.knn(i)
    acc = knnSolver04.accuracy()
    globalAccuracy04.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver04.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy04, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla sepal width i petal length w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(90, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()

#+---------------------------------------------------------------------------------+
#|                            Dla sepal width i petal width                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla sepal width i petal width---------------------------\n")


knnSolver05 = knn.AlgorithmKNN(data_train[:, [1, 3, -1]], data_test[:, [1, 3, -1]])

globalAccuracy05 = []

for i in range(1, 16):
    knnSolver05.knn(i)
    acc = knnSolver05.accuracy()
    globalAccuracy05.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver05.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy05, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla Dla sepal width i petal width w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(90, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()

#+---------------------------------------------------------------------------------+
#|                           Dla petal length i petal width                        |
#+---------------------------------------------------------------------------------+

print("\n\n---------------------------Dla petal length i petal width---------------------------\n")


knnSolver06 = knn.AlgorithmKNN(data_train[:, [2, 3, -1]], data_test[:, [2, 3, -1]])

globalAccuracy06 = []

for i in range(1, 16):
    knnSolver06.knn(i)
    acc = knnSolver06.accuracy()
    globalAccuracy06.append(acc)
    if acc == 100.0:
        print(f"k = {i}, osiągnęło 100%")
        knnSolver06.missesMatrix()

plt.figure(figsize=(10, 6))  # Ustawienie rozmiaru wykresu
plt.plot(k_values, globalAccuracy06, marker='o', linestyle='-', color='blue', label='Dokładność (%)')  # Linia z punktami

# 3. Dodanie tytułu i opisów osi
plt.title('Dokładność klasyfikacji dla petal length i petal width w zależności od k (k-NN)', fontsize=14)  # Tytuł wykresu
plt.xlabel('Liczba sąsiadów (k)', fontsize=12)  # Opis osi X
plt.ylabel('Dokładność (%)', fontsize=12)  # Opis osi Y

# Dostosowanie osi Y (np. od 75% do 90% co 2%)
plt.yticks(range(95, 101, 1))  # Znaczniki co 2%, od 75% do 90%

# 4. Dostosowanie osi X (wyświetlenie wszystkich wartości k)
plt.xticks(k_values)

# 5. Dodanie siatki i legendy
plt.grid(True, linestyle='--', alpha=0.7)  # Siatka z przerywanymi liniami
plt.legend()  # Legenda

# 6. Wyświetlenie wykresu
plt.show()
