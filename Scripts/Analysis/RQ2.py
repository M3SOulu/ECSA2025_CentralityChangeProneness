from collections import Counter
import pandas as pd

CCP = pd.read_excel("Results/RQ2/Wilcoxon.ods")

rejected = {
    "size": Counter(),
    "complexity": Counter(),
    "quality": Counter()
}

metric_types = {}
with open("Metrics/metrics_type.csv", 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        metric, type_ = line.split(',')
        metric_types[metric] = type_

software_metrics = CCP["by_Variable"].unique()
included_corr = set()
metric_counter = Counter()
for metric in software_metrics:
    # Get all correlations of specific centrality and metric across time
    time_series = CCP[CCP["by_Variable"] == metric]
    # Keep a metric if it is always stat. sig. correlated
    if all(time_series["pvalue"] <= 0.01):
        included_corr.add(metric)
        # Iterate over each hypothesis
    for tup in time_series.itertuples():
        metric = tup.by_Variable
        version = tup.Version_Id
        metric_type = metric_types[metric]
        if tup.pvalue <= 0.01:
            metric_counter[metric] += 1
            rejected[metric_type][version] += 1
print("Metrics consistently affected:", included_corr)
for version in range (1,8):
    print("Version:", version)
    print("\tSize rejected:", rejected["size"][version])
    print("\tComplexity rejected:", rejected["complexity"][version])
    print("\tQuality rejected:", rejected["quality"][version])
print(metric_counter.total())

