import seaborn as sns

# Get the first 2 columns with the highest variance
selected_features = list(X_normalized.var().sort_values(ascending=False).head(2).index)

rows, columns = 1, 2
fig, axs = plt.subplots(
    rows, columns, figsize=(columns * 5, rows * 4), layout="constrained"
)

# Original diagnosis
ax = sns.scatterplot(
    data=X_normalized,
    x=selected_features[0],
    y=selected_features[1],
    hue=y_true,
    ax=axs[0],
    hue_order=("0", "1"),
    palette=["#0cac8c", "#f25f5c"],
)
ax.get_legend().set_title("Disease?")
legend_map = {"0": "No (0)", "1": "Yes (1)"}
for text in ax.get_legend().get_texts():
    text.set_text(legend_map[text.get_text()])
ax.set_title("Original Parkinson Diagnosis")

# Predicted k=3 clusters
ax = sns.scatterplot(
    data=X_normalized,
    x=selected_features[0],
    y=selected_features[1],
    hue=y_pred,
    ax=axs[1],
    palette=["#1588e0", "#0cac8c", "#f58b00"],
)
ax.get_legend().set_title("Cluster")
ax.set_title("Learned k = 3 clusters")
