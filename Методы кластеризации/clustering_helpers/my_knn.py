from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt
# from jupyterthemes import jtplot
from pylab import rcParams
import pickle
from ipywidgets import interact

# jtplot.style()

# фиксируем размер отображаемых картинок
rcParams['figure.figsize'] = (9, 7)


def show_knn(X, y_pred, clusters):
    all_clusters = []
    for cl in range(clusters.shape[0]):
        cluster = np.array([[0, 0]])
        for i in range(X.shape[0]):
            if y_pred[i] == cl:
                cluster = np.append(cluster, np.array([X[i]]), axis=0)
        all_clusters.append(cluster)

    # Plotting along with the Centroids
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i in range(clusters.shape[0]):
        plt.scatter(all_clusters[i][1:, 0], all_clusters[i][1:, 1], s=10, c=colors[i])
        plt.scatter(clusters[i][0], clusters[i][1], marker='*', s=200, c=colors[i])
    plt.xlabel('feature1', fontsize=15)
    plt.ylabel('feature2', fontsize=15)
    plt.title('Kmeans', fontsize=20)
    plt.grid()
    plt.show()


def show_start(clusters, x_1, x_2):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    plt.scatter(x_1, x_2, c='#050505', s=10)
    for i in range(clusters.shape[0]):
        plt.scatter(clusters[i][0], clusters[i][1], marker='*', s=200, c=colors[i])
    plt.xlabel('feature1', fontsize=15)
    plt.ylabel('feature2', fontsize=15)
    plt.title('Clusters', fontsize=20)
    plt.grid()
    plt.show()


def euclidean_dist(instance1, instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.sqrt(sum((instance1 - instance2) ** 2))


def update_y_pred(X, clusters, y_pred):
    for i in range(X.shape[0]):
        old_dist = 9e12
        for num, cl in zip(range(clusters.shape[0]), clusters):
            distance = euclidean_dist(X[i], cl)
            if distance < old_dist:
                y_pred[i] = num
                old_dist = distance
    return y_pred


def update_clusters(X, clusters, y_pred):
    for clust in range(clusters.shape[0]):
        list_cluster = []
        for i in range(X.shape[0]):
            if y_pred[i] == clust:
                list_cluster.append(X[i])
        if len(list_cluster) > 0:
            clusters[clust] = np.mean(list_cluster, axis=0)
    return clusters


def show_kmeans(step):
    X, y = make_blobs(n_samples=300, centers=3,
                      cluster_std=0.80, random_state=40)
    x_1 = X[:, 0]
    x_2 = X[:, 1]
    X = np.array(list(zip(x_1, x_2)))
    clusters = np.array([[3.3, -8.8],
                         [3.7, -9.2],
                         [3.46, -9.3]])

    y_pred = np.zeros_like(y)
    steps = {}
    step_ = 1
    clusters_old = np.zeros_like(clusters)
    for i in range(8):
        clusters_old = clusters.copy()
        # defied clusters for each points
        y_pred = update_y_pred(X, clusters, y_pred)
        steps[step_] = [y_pred.copy(), clusters.copy()]  # фиксируем данные для визуализации
        step_ += 1

        # updates centroid of clusters
        clusters = update_clusters(X, clusters, y_pred)
        steps[step_] = [y_pred.copy(), clusters.copy()]  # фиксируем данные для визуализации
        step_ += 1

    if step == 0:
        show_start(steps[1][1], x_1, x_2)
    else:
        show_knn(X, steps[step][0], steps[step][1])
