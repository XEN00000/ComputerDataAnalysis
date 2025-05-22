
# Zadanie: Klasyfikacja irysów algorytmem k-NN

Zbiory danych `data3_train.csv` i `data3_test.csv` zawierają odpowiednio dane treningowe (105 obserwacji) i testowe (45 obserwacji) irysów. Każda obserwacja posiada:

1. długość działki kielicha (sepal length) [cm]  
2. szerokość działki kielicha (sepal width) [cm]  
3. długość płatka (petal length) [cm]  
4. szerokość płatka (petal width) [cm]  
5. gatunek (0 - *setosa*, 1 - *versicolor*, 2 - *virginica*)

Do klasyfikacji wykorzystaj klasę `AlgorithmKNN` z biblioteki `lib_knn.py`, która zawiera wszystkie niezbędne metody:

- normalizacja cech  
- obliczanie macierzy odległości  
- wybór k najbliższych sąsiadów  
- wybór klasy większościowej (z uwzględnieniem ważenia głosów)  
- obliczanie dokładności klasyfikacji i tworzenie macierzy pomyłek  

---

## Zadania do wykonania

### 1. Klasyfikacja na podstawie wszystkich cech:

- Przeprowadź klasyfikację dla wartości `k` od 1 do 15.  
- Wyznacz dokładność klasyfikacji dla każdej wartości `k`.  
- Znajdź i podaj wartość `k`, która daje najlepszy wynik.  
- Wyświetl macierz pomyłek dla tej wartości `k`.  
- Zobrazuj zależność dokładności od wartości `k` (wykres liniowy).  

---

### 2. Powtórz klasyfikację, wykonując analogiczne kroki jak wyżej, ale tylko dla wybranych par cech:

- **sepal length i sepal width**  
- **sepal length i petal length**  
- **sepal length i petal width**  
- **sepal width i petal length**  
- **sepal width i petal width**  
- **petal length i petal width**  

Dla każdej pary cech:
- sprawdź wartości `k` od 1 do 15,  
- narysuj wykres dokładności,  
- wypisz najlepszą wartość `k` i macierz pomyłek.
