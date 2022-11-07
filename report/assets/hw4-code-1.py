from scipy.io.arff import loadarff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, cluster
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Reading the ARFF file
data = loadarff("../data/pd_speech.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")

# Separate features from the outcome (y)
X = df.drop("class", axis=1)
y_true = df["class"]

# Normalize data
X_normalized = MinMaxScaler().fit_transform(X)

X_normalized = pd.DataFrame(X_normalized, columns=df.columns[:-1])

kmeans_models = []

for i in range(0, 3):

    # parameterize clustering
    kmeans_algo = cluster.KMeans(n_clusters=3, random_state=i)

    # learn the model
    kmeans_model = kmeans_algo.fit(X_normalized)

    # append the model to the list
    kmeans_models.append(kmeans_model)

for i in range(0, 3):

    y_pred = kmeans_models[i].labels_

    # Compute silhouette
    print(
        f"random_state = {i} | Silhouette (euclidean): {metrics.silhouette_score(X, y_pred, metric='euclidean'):6.5f}"
    )


def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


for i in range(0, 3):

    y_pred = kmeans_models[i].labels_

    # Compute purity
    print(f"random_state = {i} | Purity: {purity_score(y_true, y_pred):6.5f}")
