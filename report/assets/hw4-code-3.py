# learn the transformation (components as linear combination of features)
pca = PCA(n_components=0.8, svd_solver="full")

pca.fit(X_normalized)

print(pca.n_components_)
