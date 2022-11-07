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

X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

kmeans_models = []

for i in range(3):
    # parameterize clustering
    kmeans_algo = cluster.KMeans(n_clusters=3, random_state=i)

    # learn the model
    kmeans_model = kmeans_algo.fit(X_normalized)

    # append the model to the list
    kmeans_models.append(kmeans_model)

for model in kmeans_models:
    random_state = model.random_state
    y_pred = model.labels_

    # Compute silhouette
    silhouette = metrics.silhouette_score(X_normalized, y_pred, metric="euclidean")

    print(f"random_state = {random_state} | Silhouette (euclidean): {silhouette:6.5f}")


def purity_score(y_true, y_pred):
    # compute contingency/confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


for model in kmeans_models:
    random_state = model.random_state
    y_pred = model.labels_

    # Compute purity
    purity = purity_score(y_true, y_pred)
    print(f"random_state = {random_state} | Purity: {purity:6.5f}")
