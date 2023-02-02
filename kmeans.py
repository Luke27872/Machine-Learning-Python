import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt



task2_dataset = pd.read_csv('Task2 - dataset - dog_breeds.csv')
X = task2_dataset.iloc[:, [0, 1, 2, 3]].values



def initialise_centroids(dataset, k):
    m = dataset.shape[0]
    n = dataset.shape[1]
    Centroids = np.array([]).reshape(n, 0)
    for i in range(k):
        rand = rd.randint(0, m - 1)
        Centroids = np.c_[Centroids, dataset[rand]]
    return Centroids




def kmeans(dataset, K):
    n_iter = 100
    m = dataset.shape[0]
    Centroids = initialise_centroids(dataset, K)
    for i in range(n_iter):
        EuclidianDistance = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
            EuclidianDistance = np.c_[EuclidianDistance, tempDist]
        C = np.argmin(EuclidianDistance, axis=1) + 1
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(4, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
        for k in range(K):
            Y[k + 1] = Y[k + 1].T
        for k in range(K):
            Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        Output = Y



    colour = ['red', 'blue', 'green']
    labels = ['cluster1', 'cluster2', 'cluster3']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=colour[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.legend()
    plt.xlabel("Height")
    plt.ylabel("Tail Length")
    plt.title("K-Means Algorithm on Height and Tail Length")
    plt.show()

    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 2], c=colour[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[2, :], s=300, c='yellow', label='Centroids')
    plt.legend()
    plt.xlabel("Height")
    plt.ylabel("Leg Length")
    plt.title("K-Means Algorithm on Height and Leg Length")
    plt.show()
    return Centroids, Y

kmeans(X, 2)














