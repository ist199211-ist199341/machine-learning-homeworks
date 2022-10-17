from scipy.io.arff import loadarff
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the ARFF file
data = loadarff("../data/pd_speech.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")

# Separate features from the outcome (class)
X = df.drop("class", axis=1)
y = df["class"]

folds = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

predictor_knn = KNeighborsClassifier(
    weights="uniform", n_neighbors=5, metric="euclidean"
)

predictor_naive_bayes = GaussianNB()

knn_pred = []
naive_bayes_pred = []

knn_acc = []
naive_bayes_acc = []

y_test_values = []

for train_k, test_k in folds.split(X, y):

    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]

    scaler = StandardScaler().fit(X_train)
    # Normalize the data with a standard scaler
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit both models with the training data
    predictor_knn.fit(X_train, y_train)
    predictor_naive_bayes.fit(X_train, y_train)

    # Predict the outcome of the test values
    knn_test_pred = predictor_knn.predict(X_test)
    naive_bayes_test_pred = predictor_naive_bayes.predict(X_test)

    # Save predictions so we can create a confusion matrix later
    knn_pred += knn_test_pred.tolist()
    naive_bayes_pred += naive_bayes_test_pred.tolist()
    y_test_values += y_test.tolist()

    # Get the accuracy of each test
    knn_acc.append(accuracy_score(y_test, knn_test_pred))
    naive_bayes_acc.append(accuracy_score(y_test, naive_bayes_test_pred))

# Generate the confusion matrix data for kNN
knn_cm = np.array(confusion_matrix(y_test_values, knn_pred, labels=(["0", "1"])))
knn_confusion_df = pd.DataFrame(
    knn_cm,
    index=["Healthy", "Parkinson"],
    columns=["Predicted Healthy", "Predicted Parkinson"],
)

# Generate the confusion matrix data for Naive Bayes
cm_naive_bayes = np.array(
    confusion_matrix(y_test_values, naive_bayes_pred, labels=(["0", "1"]))
)
confusion_naive_bayes = pd.DataFrame(
    cm_naive_bayes,
    index=["Healthy", "Parkinson"],
    columns=["Predicted Healthy", "Predicted Parkinson"],
)

# Draw the confusion matrices using a heatmap
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.heatmap(knn_confusion_df, annot=True, fmt="g", cmap="Greens", ax=axes[0])
axes[0].set_title("kNN Confusion Matrix")

sns.heatmap(confusion_naive_bayes, annot=True, fmt="g", cmap="Greens", ax=axes[1])
axes[1].set_title("Naive Bayes Confusion Matrix")

plt.show()
