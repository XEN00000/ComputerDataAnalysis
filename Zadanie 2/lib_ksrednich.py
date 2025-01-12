import numpy as np
import matplotlib.pyplot as plt


# 1. Funkcja normalizująca dane za pomocą metody min-max
def normalize_min_max(x):
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min)


# 2. Implementacja algorytmu k-średnich
def kmeans(x, k, max_iter=100, tolerance=1e-4):
    # Losowa inicjalizacja centroidów
    # i tu sie warto pochylić bo z moich obserwacji wynika że od tego
    # zależą nieścisłości (np. punkt obok zielonego centroidu jest niebieski)
    np.random.seed(42)
    random_indices = np.random.choice(x.shape[0], k, replace=False)
    centroids = x[random_indices]

    labels = []
    iteration = 0

    for iteration in range(max_iter):
        # Przypisanie punktów do najbliższych centroidów
        distances = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)    # distances przechowuje w przechowuje odległości poszczególnych punktów do każdego z centroidów
        labels = np.argmin(distances, axis=1)   # labels zawiera przypisane już do klastrów punkty

        # Aktualizacja centroidów
        new_centroids = np.array([x[labels == i].mean(axis=0) for i in range(k)]) # przesuwamy centroid na środek gęstości przypisanych punktów

        # Sprawdzenie warunku stopu
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < tolerance:
            print(f"Algorytm zbieżny po {iteration + 1} iteracjach.")
            break

        centroids = new_centroids

    return labels, centroids, iteration + 1


# 3. Funkcja do obliczenia WCSS
def calculate_wcss(x, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = x[labels == i] # cluster_points zawiera tylko punkty które są przpisane do konkretnego klastra
        wcss += np.sum((cluster_points - centroid) ** 2) # oblicza różnicę między każdym punktem w klastrze a jego centroidem

    return wcss


# 4. Funkcja wizualizująca metodę łokcia
def elbow_method(x, k_range):
    wcss_values = []
    iteration_counts = []

    for k in k_range:
        labels, centroids, iterations = kmeans(x, k)
        wcss = calculate_wcss(x, labels, centroids)
        wcss_values.append(wcss)
        iteration_counts.append(iterations)

    # Wykres metody łokcia
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, wcss_values, marker='o', linestyle='-')
    plt.title("Metoda łokcia")
    plt.xlabel("Liczba klastrów (k)")
    plt.ylabel("WCSS")
    plt.grid()
    plt.show()

    return wcss_values, iteration_counts


# 5. Funkcja wizualizująca klastry
def plot_clusters(x, labels, centroids, features):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # Sześć par cech
    plt.figure(figsize=(15, 10))

    for i, (x_idx, y_idx) in enumerate(pairs):
        plt.subplot(2, 3, i + 1)
        for cluster in np.unique(labels):
            cluster_points = x[labels == cluster]
            plt.scatter(cluster_points[:, x_idx], cluster_points[:, y_idx], label=f'Klaster {cluster}')
        plt.scatter(centroids[:, x_idx], centroids[:, y_idx], color='red', marker='X', s=200, label='Centroidy')
        plt.xlabel(features[x_idx])
        plt.ylabel(features[y_idx])
        plt.title(f"{features[x_idx]} vs {features[y_idx]}")
        plt.legend()
    plt.tight_layout()
    plt.show()

# 7. Funkcja denormalizująca dane
def denormalize_centroids(normalized_centroids, X_original):
    X_min = X_original.min(axis=0)
    X_max = X_original.max(axis=0)
    return normalized_centroids * (X_max - X_min) + X_min

