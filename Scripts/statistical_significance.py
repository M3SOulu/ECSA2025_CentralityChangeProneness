import pandas as pd

# Load data on normality testing
normality = pd.read_excel("Results/AndersonDarling.xlsx")

# Metrics that did not have statistical significance at least once (removed from analysis)
non_significant_AD = normality[normality["Simulated p-Value"] > 0.01]
excluded_AD = set(non_significant_AD["Y"])
excluded_AD.discard("Taylor_FOM")
excluded_AD.discard("Taylor_FOM_NORM")
excluded_AD.discard("Taylor_TAC")


# Load pairwise Spearman correlations
correlation = pd.read_excel("Results/RQ1/Correlation.xlsx")

centrality_metrics = correlation["Variable"].unique()
software_metrics = correlation["by Variable"].unique()

included_corr = set()
for centrality in centrality_metrics:
    for metric in software_metrics:
        # Get all correlations of specific centrality and metric across time
        time_series = correlation[(correlation["Variable"] == centrality) & (correlation["by Variable"] == metric)]
        # Keep a metric if it is always stat. sig. correlated
        if all(time_series["p-value"] <= 0.01):
            included_corr.add(metric)

# All the res of metrics are excluded
software_metrics = set(software_metrics)
excluded_corr = software_metrics - included_corr

# Write the lists of included and excluded metrics
excluded_AD = sorted(excluded_AD)
excluded_corr = sorted(excluded_corr)
included_corr = sorted(included_corr)
excluded_AD = [f"{metric}\n" for metric in excluded_AD]
excluded_corr = [f"{metric}\n" for metric in excluded_corr]
included_corr = [f"{metric}\n" for metric in included_corr]

with open("Results/excluded_metrics_AD.txt", 'w') as f:
    f.writelines(excluded_AD)
with open("Results/RQ1/excluded_metrics_Spearman.txt", 'w') as f:
    f.writelines(excluded_corr)
with open("Results/RQ1/included_metrics_Spearman.txt", 'w') as f:
    f.writelines(included_corr)

