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

X = df.drop("class", axis=1)
y = df["class"]

mutualInfoClassif = feature_selection.mutual_info_classif(X, y)

df_ig = pd.DataFrame(mutualInfoClassif, columns=["values"])
df_ig.index = X.columns
df_ig = df_ig.sort_values(by=["values"], ascending=False)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=1
)

NUM_FEATURES = [5, 10, 40, 100, 250, 700]

training_accurancy = []
test_accurancy = []

for num_features in NUM_FEATURES:

    df_ig_cut = df_ig[:num_features]
    X_train_cut = X_train[df_ig_cut.index]
    X_test_cut = X_test[df_ig_cut.index]

    predictor = tree.DecisionTreeClassifier()
    predictor.fit(X_train_cut, y_train)

    y_train_pred = predictor.predict(X_train_cut)
    y_test_pred = predictor.predict(X_test_cut)

    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)

    training_accurancy.append(train_acc)
    test_accurancy.append(test_acc)

plt.plot(NUM_FEATURES, training_accurancy, label="Training Accuraccy")
plt.plot(NUM_FEATURES, test_accurancy, label="Test Accuraccy")

plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")

plt.legend()
plt.show()
