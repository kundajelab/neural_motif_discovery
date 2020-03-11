import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

# Import data
base_path = "/users/amtseng/tfmodisco/data/processed/AI-TAC/"
data_path = os.path.join(base_path, "data")

# Normalized peak heights for all cell types
cell_type_array = np.load(os.path.join(data_path, "cell_type_array.npy"))

# Names of each immune cell type in the same order as cell_type_array.npy,
# along with lineage designation of each cell type
cell_type_names = np.load(
    os.path.join(data_path, "cell_type_names.npy"), allow_pickle=True
)


# Center the features
features = np.transpose(cell_type_array)  # 81 x N
features = features - np.mean(features, axis=0, keepdims=True)

# PCA
pca = sklearn.decomposition.PCA(n_components=2)
vecs = pca.fit_transform(features)

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(vecs[:, 0], vecs[:, 1])
for i, row in enumerate(vecs):
    plt.text(row[0], row[1], cell_type_names[i][1])
plt.show()

