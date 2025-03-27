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
# Load Anderson-Darling results
AD = pd.read_excel("Results/AndersonDarling.ods")
# Load all metrics
all_metrics = pd.read_csv("Metrics/metrics_all.csv")

# Metrics that did not have statistical significance at least once (removed from analysis)
non_significant_AD = AD[AD["pvalue"] > 0.01]

# Make sure to not delete temporal centrality columns
excluded_AD = set(non_significant_AD["Variable"]) - centrality

# Excluded the metrics, keep the rest
all_non_normal = set(all_metrics.columns) - excluded_AD
excluded_AD = sorted(excluded_AD)
excluded_AD = [f"{metric}\n" for metric in excluded_AD]

# Save list of excluded metrics
with open("Results/excluded_metrics_AD.txt", 'w') as f:
    f.writelines(excluded_AD)

all_non_normal = [col for col in all_metrics.columns if col in all_non_normal]

# Save only included data for further analysis
df_all_non_normal = all_metrics[all_non_normal]
df_all_non_normal.to_csv("Results/metrics_non_normal.csv", index=False, header=True)


# Load the mapping from metric to type size/complexity/quality
metric_types = {}
with open("Metrics/metrics_type.csv", 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        metric, type_ = line.split(',')
        metric_types[metric] = type_


# Count the number of rejected hypothesis by metrics type and version
rejected = {
    "size": Counter(),
    "complexity": Counter(),
    "quality": Counter()
}

# Count the number of tested hypotheses by metric type
total_hypotheses = Counter()
# Test all hypotheses
for tup in AD.itertuples():
    pvalue = tup.pvalue
    metric = tup.Variable
    if metric not in metric_types:
        continue
    metric_type = metric_types[metric]
    total_hypotheses[metric_type] += 1
    version = tup.Version_Id
    if pvalue <= 0.01:
        rejected[metric_type][version] += 1

# Print summary by version and metric type
for version in range (1,8):
    print("Version:", version)
    print("\tSize rejected:", rejected["size"][version])
    print("\tComplexity rejected:", rejected["complexity"][version])
    print("\tQuality rejected:", rejected["quality"][version])

print("Total hypotheses:", total_hypotheses, total_hypotheses.total())
