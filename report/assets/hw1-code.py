from operator import itemgetter
import pandas as pd
from scipy.io.arff import loadarff
from sklearn import feature_selection, model_selection, tree, metrics, preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Reading the ARFF file
data = loadarff("data/pd_speech.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")

# Separate features from the outcome (class)
X = df.drop("class", axis=1)
y = df["class"]

# Split the dataset into a training set (70%) and a testing set (30%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=1
)

NUM_FEATURES = [5, 10, 40, 100, 250, 700]

training_accuracy = []
test_accuracy = []

for num_features in NUM_FEATURES:
    # Select the features with the greatest information gain (by mutual_info_classif)
    kbest = feature_selection.SelectKBest(
        score_func=lambda a, b: feature_selection.mutual_info_classif(
            a, b, random_state=1
        ),
        k=num_features,
    )
    kbest.fit(X_train, y_train)

    # Get the observation sets with only the selected N best features
    X_train_cut = kbest.transform(X_train)
    X_test_cut = kbest.transform(X_test)

    # Fit the decision tree classifier
    predictor = tree.DecisionTreeClassifier(random_state=1)
    predictor.fit(X_train_cut, y_train)

    # Use the decision tree to predict the outcome of the given observations
    y_train_pred = predictor.predict(X_train_cut)
    y_test_pred = predictor.predict(X_test_cut)

    # Get the accuracy of each test
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)

    training_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

plt.plot(
    NUM_FEATURES,
    training_accuracy,
    label="Training Accuracy",
    marker="+",
    color="#4caf50",
)
plt.plot(
    NUM_FEATURES, test_accuracy, label="Test Accuracy", marker=".", color="#ff5722"
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")

plt.legend()
plt.show()
