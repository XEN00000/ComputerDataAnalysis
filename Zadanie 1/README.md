
# Analiza danych o irysach

Dane do analizy znajdują się w pliku `data1.csv`, zawierającym informacje o cechach trzech gatunków irysów: *setosa*, *versicolor* i *virginica*. Plik zawiera 150 obserwacji (po 50 dla każdego gatunku) oraz następujące kolumny:

1. Długość działki kielicha (cm)  
2. Szerokość działki kielicha (cm)  
3. Długość płatka (cm)  
4. Szerokość płatka (cm)  
5. Gatunek (0 - *setosa*, 1 - *versicolor*, 2 - *virginica*)

## Wymagania
Zadanie należy wykonać w języku Python z użyciem bibliotek:  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `numpy`

---

## Zadanie 1.1  
Wczytaj dane, przypisz nazwy kolumn i przemapuj numery gatunków na ich nazwy słowne. Następnie policz:
- liczebność obserwacji dla każdego gatunku,
- procentowy udział każdego gatunku w całym zbiorze,
- sumaryczne podsumowanie („Razem”).

Wyniki zaprezentuj w postaci tabeli.

---

## Zadanie 1.2  
Dla każdej z cech liczbowych (bez kolumny „Gatunek”) wyznacz:
- wartość minimalną,
- wartość maksymalną,
- średnią,
- odchylenie standardowe,
- medianę,
- kwartyle Q1 i Q3.

Wyniki przedstaw w formie tabeli.

---

## Zadanie 2.1  
Wygeneruj histogramy dla każdej z cech liczbowych (4 wykresy), stosując odpowiednie przedziały klasowe. Zadbaj o czytelność osi, tytułów i legendy.

---

## Zadanie 2.2  
Utwórz wykresy pudełkowe (boxploty) dla każdej cechy liczbowej, z podziałem na trzy gatunki irysów. (4 wykresy)

---

## Zadanie 3.1  
Dla wszystkich możliwych par cech liczbowych (łącznie 6 wykresów), wyznacz współczynnik korelacji Pearsona i dopasuj linię regresji. Dla każdego wykresu podaj równanie prostej oraz wartość współczynnika korelacji.
