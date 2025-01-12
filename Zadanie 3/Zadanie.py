import numpy as np
import lib_knn as knn


data_train = np.loadtxt('data3_train.csv', delimiter=',')
data_test = np.loadtxt('data3_test.csv', delimiter=',')

# print(data_train)
# print(data_train2)
# print(data_test)
# print(data_test2)

knnSolver = knn.AlgorithmKNN(data_train)

knnSolver.loadTestData(data_test)

for i in range(1, 16):
    knnSolver.knn(i)
    knnSolver.visualize()
    print(f"k = {i}, osiągnęło skuteczność: {knnSolver.accuracy()}%")
# nnSolver.knn(4)
# rint(f"{knnSolver.accuracy()}%")
# knnSolver.visualize()
# knnSolver.missesMatrix()
