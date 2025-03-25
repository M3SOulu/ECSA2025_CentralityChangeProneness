import pandas as pd
from collections import Counter

centrality ={
    "Taylor_JC",
    "Taylor_CC",
    "Liu_JC",
    "Liu_CC",
    "Yin_JC",
    "Yin_CC",
    "Huang_JC",
    "Huang_CC",
    "Taylor_FOM",
    "Taylor_FOM_NORM",
    "Taylor_TAC"
}
static = {"Taylor_MNC", "Yin_MNC", "Liu_MNC", "Huang_MNC"}
# Load data on normality testing
normality = pd.read_excel("Results/AndersonDarling.ods")
all_metrics = pd.read_csv("Metrics/metrics_all.csv")

# Metrics that did not have statistical significance at least once (removed from analysis)
non_significant_AD = normality[normality["pvalue"] > 0.01]
excluded_AD = set(non_significant_AD["Variable"]) - centrality

all_non_normal = set(all_metrics.columns) - excluded_AD
excluded_AD = sorted(excluded_AD)
excluded_AD = [f"{metric}\n" for metric in excluded_AD]

with open("Results/excluded_metrics_AD.txt", 'w') as f:
    f.writelines(excluded_AD)

all_non_normal = [col for col in all_metrics.columns if col in all_non_normal]

df_all_non_normal = all_metrics[all_non_normal]
df_all_non_normal.to_csv("Results/metrics_non_normal.csv", index=False, header=True)


metric_types = {}
with open("Metrics/metrics_type.csv", 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        metric, type_ = line.split(',')
        metric_types[metric] = type_


rejected = {
    "size": Counter(),
    "complexity": Counter(),
    "quality": Counter()
}
total_hypotheses = Counter()
for tup in normality.itertuples():
    pvalue = tup.pvalue
    metric = tup.Variable
    if metric not in metric_types:
        continue
    metric_type = metric_types[metric]
    total_hypotheses[metric_type] += 1
    version = tup.Version_Id
    if pvalue <= 0.01:
        rejected[metric_type][version] += 1

for version in range (1,8):
    print("Version:", version)
    print("\tSize rejected:", rejected["size"][version])
    print("\tComplexity rejected:", rejected["complexity"][version])
    print("\tQuality rejected:", rejected["quality"][version])

print("Total hypotheses:", total_hypotheses, total_hypotheses.total())
