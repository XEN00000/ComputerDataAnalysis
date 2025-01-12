import pandas as pd
import lib_ksrednich as ksr

# 6. Główna część programu
if __name__ == "__main__":

    # Wczytanie danych do pliku
    data = pd.read_csv('data2.csv', header=None).values
    plot_labels = ['Długość działki kielicha',
              'Szerokość działki kielicha',
              'Długość płatka',
              'Szerokość płatka']

    # Normalizacja danych
    X_normalized = ksr.normalize_min_max(data)  # Normalizacja kopii danych

    # Analiza dla różnych wartości k
    k_range = range(2, 11)
    wcss_values, iteration_counts = ksr.elbow_method(X_normalized, k_range)

    # Wizualizacja wyników dla k=3
    labels, centroids, iterations = ksr.kmeans(X_normalized, k=3)
    centroids_original = ksr.denormalize_centroids(centroids, data)
    ksr.plot_clusters(data, labels, centroids_original, plot_labels)