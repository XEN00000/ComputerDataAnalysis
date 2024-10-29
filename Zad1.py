import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# Wczytanie danych
# Zakładam, że plik 'data1.csv' to klasyczny zbiór irysów bez nagłówków
file_path = 'data1.csv'
columns = ['długość działki kielicha', 'szerokość działki kielicha', 'długość płatka', 'szerokość płatka', 'gatunek']
data = pd.read_csv(file_path, header=None, names=columns)

# Zamiana wartości w kolumnie 'gatunek' na odpowiednie nazwy gatunków
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
data['gatunek'] = data['gatunek'].map(species_mapping)

# 1. Liczności poszczególnych gatunków
# Obliczenie liczby wystąpień poszczególnych gatunków
species_counts = data['gatunek'].value_counts()

# Obliczenie udziałów procentowych poszczególnych gatunków
species_percentage = (species_counts / len(data)) * 100

# Zaokrąglenie udziałów procentowych do jednego miejsca po przecinku
species_percentage = species_percentage.round(1)

# 2. Miary rozkładu dla cech kwiatów
# Obliczenie miar: minimum, maksimum, średnia, mediana, dolny kwartyl (Q1), górny kwartyl (Q3), odchylenie standardowe
# W tabeli 2 będą zawarte te statystyki dla każdej cechy (oprócz gatunku)

# Lista cech, które będziemy analizować
features = data.columns[:-1]  # Zakładając, że ostatnia kolumna to 'gatunek'

# Stworzenie DataFrame do przechowywania miar
summary_stats = pd.DataFrame(index=features,
                             columns=['Minimum', 'Śr. arytm. (± odch. stand.)', 'Mediana (Q1 - Q3)',
                                      'Maksimum'])

for feature in features:
    # Minimum i maksimum
    min_value = data[feature].min()
    max_value = data[feature].max()

    # Średnia arytmetyczna i odchylenie standardowe
    mean_value = data[feature].mean()
    std_dev = data[feature].std()
    mean_std = f"{mean_value:.2f} (± {std_dev:.2f})"

    # Mediana oraz kwartyle Q1 i Q3
    median = data[feature].median()
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    median_q1_q3 = f"{median:.2f} ({q1:.2f} - {q3:.2f})"

    # Uzupełnienie tabeli z miarami
    summary_stats.loc[feature] = [f"{min_value:.2f}", mean_std, median_q1_q3, f"{max_value:.2f}"]

# Wyświetlenie wyników bez kropek
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("\nTabela 1: Liczności gatunków irysów")
species_summary = pd.DataFrame({
    'Gatunek': species_counts.index,  # Nazwy gatunków
    'Liczebność': species_counts.values,  # Liczba obserwacji dla każdego gatunku
    'Udział procentowy (%)': species_percentage.values  # Udział procentowy dla każdego gatunku
})
print(species_summary.to_string(index=False, line_width=None, col_space=15, justify='center'))  # Wyświetlamy tabelę bez indeksów, z dodanymi separatorami

# Wyświetlenie tabeli 2: Charakterystyka cech irysów
print("\nTabela 2: Charakterystyka cech irysów")
print(summary_stats.to_string(index=True, line_width=None, col_space=15, justify='center'))  # Wyświetlamy pełną tabelę statystyk z separatorami

# Tworzenie histogramów i wykresów pudełkowych
plt.figure(figsize=(15, 20))  # Ustawienie rozmiaru całego wykresu

# Iteracja po każdej cesze w celu stworzenia wykresów
for i, feature in enumerate(features):
    # Histogram dla danej cechy
    plt.subplot(4, 2, 2 * i + 1)  # Ustawienie miejsca na wykres (1, 3, 5, 7)
    sns.histplot(data[feature], bins=10, kde=False, color='blue')  # Tworzenie histogramu
    plt.xlabel(f'{feature} (cm)')
    plt.ylabel('Liczebność')
    plt.title(f'Histogram: {feature}')

    # Wykres pudełkowy dla danej cechy z rozróżnieniem na gatunki
    plt.subplot(4, 2, 2 * i + 2)  # Ustawienie miejsca na wykres (2, 4, 6, 8)
    sns.boxplot(x='gatunek', y=feature, data=data, palette='Set3')  # Tworzenie wykresu pudełkowego
    plt.xlabel('Gatunek')
    plt.ylabel(f'{feature} (cm)')
    plt.title(f'Wykres pudełkowy: {feature}')

# Zapisanie wykresów do pliku PNG i wyświetlenie ich
plt.tight_layout()  # Optymalne rozmieszczenie wykresów bez nakładania się
plt.savefig('wykresy_cech_irysow.png')  # Zapisanie wykresów do pliku
plt.show()  # Wyświetlenie wszystkich wykresów

# Tworzenie wykresów punktowych z linią regresji dla każdej pary cech
plt.figure(figsize=(20, 20))  # Ustawienie rozmiaru całego wykresu

# Iteracja po parach cech
for i, (feature_x, feature_y) in enumerate(
        [(features[0], features[1]), (features[0], features[2]), (features[0], features[3]), (features[1], features[2]),
         (features[1], features[3]), (features[2], features[3])]):
    # Obliczenie współczynnika korelacji Pearsona oraz linii regresji
    correlation, p_value = np.corrcoef(data[feature_x], data[feature_y])[0, 1], 0  # Współczynnik korelacji Pearsona
    slope, intercept, r_value, p_value, std_err = linregress(data[feature_x], data[feature_y])
    regression_line = f"y = {slope:.2f}x + {intercept:.2f}"  # Równanie regresji

    # Tworzenie wykresu punktowego z linią regresji
    plt.subplot(3, 2, i + 1)  # Ustawienie miejsca na wykres
    sns.scatterplot(x=feature_x, y=feature_y, data=data, color='blue')  # Wykres punktowy
    plt.plot(data[feature_x], slope * data[feature_x] + intercept, color='red')  # Linia regresji
    plt.xlabel(f'{feature_x} (cm)')
    plt.ylabel(f'{feature_y} (cm)')
    plt.title(f'r = {correlation:.2f}; {regression_line}')

# Zapisanie wykresów punktowych do pliku PNG i wyświetlenie ich
plt.tight_layout()  # Optymalne rozmieszczenie wykresów bez nakładania się
plt.savefig('wykresy_regresji_cech_irysow.png')  # Zapisanie wykresów do pliku
plt.show()  # Wyświetlenie wszystkich wykresów
