from scipy.io.arff import loadarff
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

# Reading the ARFF file
data = loadarff("../data/kin8nm.arff")
df = pd.DataFrame(data[0])

# Separate features from the outcome (y)
X = df.drop("y", axis=1)
y = df["y"]

# Split the dataset into a training set (70%) and a testing set (30%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X.values, y.values, train_size=0.7, random_state=0
)

# Create a Ridge Regression model
rr = Ridge(alpha=0.01)

# Create a Multi-Layer Perceptron model with early stopping
mlp1 = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="tanh",
    max_iter=500,
    random_state=0,
    early_stopping=True,
)

# Create a Multi-Layer Perceptron model without early stopping
mlp2 = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="tanh",
    max_iter=500,
    random_state=0,
    early_stopping=False,
)

# Fit the models
rr.fit(X_train, y_train)
mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

# Predict the outcome for the test set
rr_pred = rr.predict(X_test)
mlp1_pred = mlp1.predict(X_test)
mlp2_pred = mlp2.predict(X_test)

# Print the results
print("Ridge Regularization MAE:", metrics.mean_absolute_error(y_test, rr_pred))
print("MLP1 Regularization MAE:", metrics.mean_absolute_error(y_test, mlp1_pred))
print("MLP2 Regularization MAE:", metrics.mean_absolute_error(y_test, mlp2_pred))
