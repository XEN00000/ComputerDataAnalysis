
# Zadanie: Klasteryzacja danych irysów metodą k-średnich

Dysponujesz zbiorem danych (`data2.csv`) zawierającym 150 obserwacji irysów. Każda obserwacja zawiera cztery cechy:

1. długość działki kielicha (sepal length) [cm]  
2. szerokość działki kielicha (sepal width) [cm]  
3. długość płatka (petal length) [cm]  
4. szerokość płatka (petal width) [cm]  

Twoim zadaniem jest przeprowadzenie klasteryzacji danych metodą **k-średnich** z wykorzystaniem dołączonej biblioteki `lib_ksrednich.py`, zawierającej potrzebne funkcje:

- `normalize_min_max` – normalizacja danych  
- `kmeans` – implementacja algorytmu k-średnich  
- `calculate_wcss` – obliczanie błędu wewnątrzklastrowego (WCSS)  
- `elbow_method` – wykres metody łokcia  
- `plot_clusters` – wizualizacja klastrów  
- `denormalize_centroids` – przekształcenie centroidów do oryginalnej skali danych  

---

## Wykonaj poniższe zadania:

### 1. Wczytaj dane z pliku `data2.csv` i nazwij kolumny według podanych cech.

### 2. Znormalizuj dane za pomocą metody min-max.

### 3. Zastosuj **metodę łokcia**, aby dobrać optymalną liczbę klastrów. W zakresie `k = 2` do `k = 10`:
- uruchom algorytm k-średnich,  
- oblicz wartość WCSS (within-cluster sum of squares),  
- przedstaw wyniki graficznie.

### 4. Wybierz **k = 3** i:
- przeprowadź klasteryzację,  
- przekształć centroidy znormalizowane z powrotem do oryginalnej skali danych,  
- wygeneruj **6 wykresów** rozrzutu (scatter plot) dla wszystkich par cech, pokazujących klastry i centroidy.
