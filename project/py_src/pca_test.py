from sklearn.decomposition import PCA
import numpy as np

X = np.zeros((256, 256)).reshape(256, 256)
X[1, :] = 1
X[2, :] = 1
X[:, 1] = 1
X[5, 5] = 25
X[6, 6] = 36
X[7, 7] = 49
X[8, 8] = 64
X[9, 9] = 81
X[10, 10] = 100

pca = PCA(n_components=256)
pca.fit(X)

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
threshold = 1.0  # reconstruction threshold

# the number of components that exceed the threshold
num_components_needed = np.argmax(cumulative_variance_ratio >= threshold) + 1

print(f"Number of components needed for {threshold:.0%} variance retention: {num_components_needed}")

print("Principal components:")
for component, i in zip(pca.components_, range(1, num_components_needed + 1)):
    print(component)