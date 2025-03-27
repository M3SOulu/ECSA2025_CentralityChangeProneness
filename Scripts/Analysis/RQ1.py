from collections import Counter
import pandas as pd


centralities =[
    "Yin_JC",
    "Yin_CC",
    "Liu_JC",
    "Liu_CC",
    "Huang_JC",
    "Huang_CC",
    "Taylor_JC",
    "Taylor_CC",
    "Taylor_TAC",
    "Taylor_FOM",
    "Taylor_FOM_NORM",
    "CCP"
]

# Load the mapping from metric to type size/complexity/quality
metric_types = {}
with open("Metrics/metrics_type.csv", 'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        metric, type_ = line.split(',')
        metric_types[metric] = type_


# Load pairwise Spearman correlations
correlation = pd.read_excel("Results/RQ1/Correlation.ods")

# Count the number of rejected hypothesis by metrics type and version
rejected = {
    "size": Counter(),
    "complexity": Counter(),
    "quality": Counter()
}

# Count the number of tested hypotheses by metric type
total_hypotheses = Counter()
for tup in correlation.itertuples():
    metric = tup.by_Variable
    metric_type = metric_types[metric]
    total_hypotheses[metric_type] += 1
    centrality = tup.Variable
    version = tup.Version_Id
    pvalue = tup.pvalue
    if pvalue <= 0.01:
        rejected[metric_type][version] += 1

# Print summary by version and metric type
for version in range (1,8):
    print("Version:", version)
    print("\tSize rejected:", rejected["size"][version])
    print("\tComplexity rejected:", rejected["complexity"][version])
    print("\tQuality rejected:", rejected["quality"][version])

print("Total hypotheses:", total_hypotheses)



centrality_metrics = correlation["Variable"].unique()
software_metrics = correlation["by_Variable"].unique()
included_corr = set()
# Include only metrics that always have a stat.sig. correlation with at least one centrality
for centrality in centrality_metrics:
    for metric in software_metrics:
        # Get all correlations of specific centrality and metric across time
        time_series = correlation[(correlation["Variable"] == centrality) & (correlation["by_Variable"] == metric)]
        # Keep a metric if it is always stat. sig. correlated
        if all(time_series["pvalue"] <= 0.01):
            included_corr.add(metric)

# # All the rest of metrics are excluded
software_metrics = set(software_metrics)
excluded_corr = software_metrics - included_corr

# Write the lists of included and excluded metrics
excluded_corr = sorted(excluded_corr)
included_corr = sorted(included_corr)

# Split included metrics by type
included_size = [metric for metric in included_corr if metric_types[metric] == 'size']
included_complexity = [metric for metric in included_corr if metric_types[metric] == 'complexity']
included_quality = [metric for metric in included_corr if metric_types[metric] == 'quality']

# Save list of included and excluded metrics
excluded_corr = [f"{metric}\n" for metric in excluded_corr]
included_corr = [f"{metric}\n" for metric in included_corr]
with open("Results/RQ1/excluded_metrics_Spearman.txt", 'w') as f:
    f.writelines(excluded_corr)
with open("Results/RQ1/included_metrics_Spearman.txt", 'w') as f:
    f.writelines(included_corr)

# Save included size, centrality, and quality metrics separately
size_columns = ["MS_system", "Version Id", "Microservice", *included_size, *centralities]
complexity_columns = ["MS_system", "Version Id", "Microservice", *included_complexity, *centralities]
quality_columns = ["MS_system", "Version Id", "Microservice", *included_quality, *centralities]

all_metrics = pd.read_csv("Results/metrics_non_normal.csv")
size_metrics = all_metrics[size_columns]
complexity_metrics = all_metrics[complexity_columns]
quality_metrics = all_metrics[quality_columns]
size_metrics.to_csv("Results/RQ1/metrics_size.csv", index=False)
complexity_metrics.to_csv("Results/RQ1/metrics_complexity.csv", index=False)
quality_metrics.to_csv("Results/RQ1/metrics_quality.csv", index=False)
