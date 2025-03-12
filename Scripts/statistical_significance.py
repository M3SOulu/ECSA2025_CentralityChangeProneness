import csv

import pandas as pd

# Load data on normality testing
normality = pd.read_excel("Results/AndersonDarling.xlsx")

# Metrics that did not have statistical significance at least once (removed from analysis)
non_significant_AD = normality[normality["Simulated p-Value"] > 0.01]
filtered_AD = set(non_significant_AD["Y"])
filtered_AD.discard("Taylor_FOM")
filtered_AD.discard("Taylor_FOM_NORM")
filtered_AD.discard("Taylor_TAC")

# The rest of the metrics (to keep for analysis)
kept_AD = set(normality["Y"]) - set(filtered_AD)

# Load pairwise Spearman correlations
correlation = pd.read_excel("Results/Correlation.xlsx")

centrality_metrics = [
    "Taylor_JC",
    "Yin_JC",
    "Liu_JC",
    "Huang_JC",
    "Taylor_CC",
    "Yin_CC",
    "Liu_CC",
    "Huang_CC",
]
static = [
    "Taylor_MNC",
    "Huang_MNC",
    "Yin_MNC",
    "Liu_MNC",
    "Taylor_MLC",
    "Huang_MLC",
    "Yin_MLC",
    "Liu_MLC",
    "Taylor_TAC",
    "Taylor_FOM",
    "Taylor_FOM_NORM",
]

# Make sure "Variable" column is only centrality and "by Variable" are the kept metrics
kept_AD = set(kept_AD) - set(centrality_metrics)
kept_AD = set(kept_AD) - set(static)
correlation = correlation[correlation["Variable"].isin(centrality_metrics)]
correlation = correlation[correlation["by Variable"].isin(kept_AD)]

kept_corr = set()
for centrality in centrality_metrics:
    for metric in kept_AD:
        # Get all correlations of specific centrality and metric across time
        time_series = correlation[(correlation["Variable"] == centrality) & (correlation["by Variable"] == metric)]
        # Keep a metric if it is always stat. sig. correlated
        if all(time_series["p-value"] <= 0.01):
            kept_corr.add(metric)

# All the res of metrics are excluded
filtered_corr = kept_AD - kept_corr

# Write a csv with all excluded metrics and the reason for exclusion
filtered_AD = sorted(filtered_AD)
filtered_corr = sorted(filtered_corr)
with open("Results/excluded_metrics.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metric","Reason"])
    for metric in filtered_AD:
        writer.writerow([metric,"AD"])
    for metric in filtered_corr:
        writer.writerow([metric,"Rho"])

# Write a list of included metrics
kept_corr = sorted(kept_corr)
kept_corr = [f"{metric}\n" for metric in kept_corr]
with open("Results/included_metrics.txt", 'w') as f:
    f.writelines(kept_corr)

