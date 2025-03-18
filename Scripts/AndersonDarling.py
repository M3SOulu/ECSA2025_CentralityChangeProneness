import pandas as pd

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
normality = pd.read_excel("Results/AndersonDarling.xlsx")
all_metrics = pd.read_csv("Metrics/metrics_all.csv")

# Metrics that did not have statistical significance at least once (removed from analysis)
non_significant_AD = normality[normality["Simulated p-Value"] > 0.01]
excluded_AD = set(non_significant_AD["Y"]) - centrality

all_non_normal = set(all_metrics.columns) - excluded_AD
excluded_AD = sorted(excluded_AD)
excluded_AD = [f"{metric}\n" for metric in excluded_AD]

with open("Results/excluded_metrics_AD.txt", 'w') as f:
    f.writelines(excluded_AD)

all_non_normal = [col for col in all_metrics.columns if col in all_non_normal]

df_all_non_normal = all_metrics[all_non_normal]
df_all_non_normal.to_csv("Results/metrics_non_normal.csv", index=False, header=True)
