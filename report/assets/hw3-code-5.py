import seaborn as sns
import matplotlib.pyplot as plt

rr_residues = []
mlp1_residues = []
mlp2_residues = []

# Calculate the residues for each model
for i in range(0, len(y_test)):
    rr_residues.append(abs(y_test[i] - rr_pred[i]))
    mlp1_residues.append(abs(y_test[i] - mlp1_pred[i]))
    mlp2_residues.append(abs(y_test[i] - mlp2_pred[i]))

# Create dataframe with the residues
df = pd.DataFrame({"Ridge": rr_residues, "MLP1": mlp1_residues, "MLP2": mlp2_residues})

# Plot the residues with a boxplot
ax = sns.boxplot(data=df)

# Specfiy axis labels
ax.set(xlabel="Models", ylabel="Residues")

plt.show()

# Plot the residues with a histplot
ax = sns.histplot(data=df, color=["red", "green", "blue"], multiple="dodge", bins=20)

# Specfiy axis labels
ax.set(xlabel="Residue Value", ylabel="Count")

plt.show()
