import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

task2_dataset = pd.read_csv('Task2 - dataset - dog_breeds.csv')
X = task2_dataset.iloc[:, [0, 1, 2, 3]].values



def initialise_centroids(dataset, k):
    x = dataset.shape[0]
    y = dataset.shape[0]
    Centroids = np.array([]).reshape(y, 0)
    for i in range(k):
        randomInitialise = rd.randint(0, x - 1)
        Centroids = np.c_[Centroids, dataset[randomInitialise]]
    return Centroids

def kmeans(dataset, K):
    # X = dataset.iloc[:, [2, 3]].values
    n_iter = 100
    m = dataset.shape[0]  # number of training examples

    Centroids = initialise_centroids(dataset, K)

    for i in range(n_iter):
        # step 2.a
        EuclidianDistance = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
            EuclidianDistance = np.c_[EuclidianDistance, tempDist]
        C = np.argmin(EuclidianDistance, axis=1) + 1
        # step 2.b
        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(4, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]

        for k in range(K):
            Y[k + 1] = Y[k + 1].T

        for k in range(K):
            Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        Output = {}
        Output = Y

    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.xlabel('Income')
    plt.ylabel('Number of transactions')
    plt.legend()
    plt.show()

    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5']
    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 2], c=color[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[2, :], s=300, c='yellow', label='Centroids')
    plt.xlabel('Income')
    plt.ylabel('Number of transactions')
    plt.legend()
    plt.show()
    return Centroids, Y

kmeans(X, 2)