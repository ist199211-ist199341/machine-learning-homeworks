from scipy import stats

# is knn better than naive bayes?
res = stats.ttest_rel(knn_acc, naive_bayes_acc, alternative="greater")
print("knn > naive_bayes? pval=", res.pvalue)
