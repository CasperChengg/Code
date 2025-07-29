import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
def split_dataset_by_class(X, y, test_size=0.2, random_state=None):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    unique_classes = np.unique(y)
    for class_label in unique_classes:
        mask = y == class_label
        X_class, y_class = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=test_size)

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    X_train = np.vstack(X_train_list) 
    X_test = np.vstack(X_test_list)
    y_train = np.hstack(y_train_list)
    y_test = np.hstack(y_test_list)

    return X_train, X_test, y_train, y_test

def blobs_generator(n_classes = 3, n_samples = 1000):

    # np.random.seed(random_state)
    rand_values = np.random.rand(n_classes)
    rand_values = rand_values ** 2
    rand_values = rand_values / np.sum(rand_values)
    class_counts = (n_samples * rand_values).astype(int)
    print(class_counts)
    
    centers = np.random.uniform(-10, 10, size=(n_classes, 2)) 
    X_blobs, y_blobs = make_blobs(n_samples=class_counts, centers=centers, cluster_std=3)
    X_blobs_train, X_blobs_test, y_blobs_train, y_blobs_test = split_dataset_by_class(X_blobs, y_blobs + 1, test_size=0.2)
    
    np.savetxt("./blobs-5-1tra.dat", np.hstack((X_blobs_train, y_blobs_train.reshape(-1, 1))), delimiter=",", fmt="%.6f")
    np.savetxt("./blobs-5-1tst.dat", np.hstack((X_blobs_test, y_blobs_test.reshape(-1, 1))), delimiter=",", fmt="%.6f")

    fig, axes = plt.subplots(2, figsize=(12, 12))

    axes[0].scatter(X_blobs_train[:, 0], X_blobs_train[:, 1], c=y_blobs_train, cmap="viridis", edgecolors="k")
    axes[1].scatter(X_blobs_test[:, 0], X_blobs_test[:, 1], c=y_blobs_test, cmap="viridis", edgecolors="k")

    plt.show()


if __name__ == "__main__":
    blobs_generator()