# get the first 2 columns with the highest variance
selected_features = list(X_normalized.var().sort_values(ascending=False).head(2).index)
