import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
def split_dataset_by_class(X, y, test_size=0.2):
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

def generate_spiral(n_samples, n_classes, noise=0.6):
    X, y = [], []
    for j in range(n_classes):
        class_count = n_samples[j]
        r = np.linspace(0.0, 2, class_count)
        t = np.linspace(j * (2 * np.pi / n_classes), (j + 1) * (2 * np.pi / n_classes), class_count) 
        t += np.random.randn(class_count) * noise 
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(class_count, j)) 
    return np.vstack(X), np.hstack(y)

def spiral_generator(n_classes = 3, n_samples = 1000, random_state=42):

    # np.random.seed(random_state)
    rand_values = np.random.rand(n_classes)
    rand_values = rand_values ** 2
    rand_values = rand_values / np.sum(rand_values)
    
    class_counts = (n_samples * rand_values).astype(int)
    print(class_counts)

    X_spiral, y_spiral = generate_spiral(class_counts, n_classes=n_classes)
    X_spiral_train, X_spiral_test, y_spiral_train, y_spiral_test = train_test_split(X_spiral, y_spiral + 1, test_size=0.2)

    np.savetxt("./spiral-5-1tra.dat", np.hstack((X_spiral_train, y_spiral_train.reshape(-1, 1))), delimiter=",", fmt="%.6f")
    np.savetxt("./spiral-5-1tst.dat", np.hstack((X_spiral_test, y_spiral_test.reshape(-1, 1))), delimiter=",", fmt="%.6f")

    fig, axes = plt.subplots(2, figsize=(12, 12))

    axes[0].scatter(X_spiral_train[:, 0], X_spiral_train[:, 1], c=y_spiral_train.astype(int), cmap="viridis", edgecolors="k")
    axes[1].scatter(X_spiral_test[:, 0], X_spiral_test[:, 1], c=y_spiral_test.astype(int), cmap="viridis", edgecolors="k")

    plt.show()


if __name__ == "__main__":
    spiral_generator()