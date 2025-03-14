import pandas as pd

df_metrics = pd.read_csv("Results/included_metrics_type.csv")

all_metrics = pd.read_csv("Metrics/metrics_all.csv")

centrality_cols = [col for col in all_metrics.columns if "_JC" in col or "_CC" in col]
centrality_cols.extend(["Taylor_TAC", "Taylor_FOM", "Taylor_FOM_NORM"])
complexity_metrics = (["MS_system", "Version Id", "Microservice"]
                      + list(df_metrics[df_metrics["Type"] == "complexity"]["Metric"])
                      + centrality_cols )
df_complexity = all_metrics[complexity_metrics]
df_complexity.to_csv("Results/RQ1/metrics_complexity.csv", index=False, header=True)

size_metrics = (["MS_system", "Version Id", "Microservice"]
                      + list(df_metrics[df_metrics["Type"] == "size"]["Metric"])
                      + centrality_cols )
df_size = all_metrics[size_metrics]
df_size.to_csv("Results/RQ1/metrics_size.csv", index=False, header=True)

quality_metrics = (["MS_system", "Version Id", "Microservice"]
                + list(df_metrics[df_metrics["Type"] == "quality"]["Metric"])
                + centrality_cols )
df_quality = all_metrics[quality_metrics]
df_quality.to_csv("Results/RQ1/metrics_quality.csv", index=False, header=True)


with open("Results/excluded_metrics_AD.txt", 'r') as f:
    metrics = f.readlines()
excluded_ad = {metric.strip("\n") for metric in metrics}
all_non_normal = set(all_metrics.columns) - excluded_ad
all_non_normal = [col for col in all_metrics.columns if col in all_non_normal]

df_all_non_normal = all_metrics[all_non_normal]
df_all_non_normal.to_csv("Results/metrics_non_normal.csv", index=False, header=True)