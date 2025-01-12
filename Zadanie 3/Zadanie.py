#       Samo użycie naszej biblioteki do rozwiązania zadania
import numpy as np

import lib_knn as knn

data_train = np.array([
    [5.1, 3.5, 1.4, 0.2, 0],
    [5.2, 3.5, 1.5, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0],
    [6.7, 3.1, 4.4, 1.4, 1],
    [5.2, 2.7, 3.9, 1.4, 1],
    [6.9, 3.1, 5.1, 2.3, 2],
    [7.7, 3.8, 6.7, 2.2, 2],
    [7.2, 3.2, 6.0, 1.8, 2]
])

data_test = np.array([
    [5.5, 4.2, 1.4, 0.2, 0],
    [5.6, 2.7, 4.2, 1.3, 1],
    [6.8, 3.0, 5.5, 2.1, 2]
])

knnSolver = knn.AlgorithmKNN(data_train)

knnSolver.loadTestData(data_test)
knnSolver.knn(3)

#knnSolver.visualize()
#knnSolver.missesMatrix()