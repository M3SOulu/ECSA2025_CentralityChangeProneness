import json

import pandas as pd

with open("raw_data/package_map.json", 'r') as f:
    PACKAGE_MAP = json.load(f)

# Mappping of packages to services
def map_packages(df: pd.DataFrame) -> pd.DataFrame:
    new_col = []
    for row in df.itertuples():
        version = row.MS_system
        package = row.Package
        for base_package, service in PACKAGE_MAP[version].items():
            if package.startswith(base_package):
                new_col.append(service)
                break
        else:
            new_col.append(None)
    df["Microservice"] = new_col
    # Remove rows that are not mapped to a service
    df = df.dropna(subset=["Microservice"])
    df = df.drop(columns=["Package"])
    return df


# --- Understand metrics
understand = pd.read_csv("Metrics/metrics_understand.csv")

# Remove NaN columns
understand = understand.dropna(axis=1, how='all')

# Map Package to Microservice
understand = understand.rename(columns={"Name": "Package"})
understand = map_packages(understand)

# Remove Average and Ratio columns
avg_cols = [col for col in understand.columns if "Avg" in col or "Ratio" in col or "Max" in col]
understand = understand.drop(columns=avg_cols)

# Group by microservice and sum all the metric columns
understand = understand.groupby(["MS_system", "Microservice"], as_index=False).sum()

# Calculate Private to Public method ratio
understand["RatioPrivateToPublicMethod"] = understand["CountDeclMethodPrivate"]/understand["CountDeclMethodPublic"]
understand["RatioProtectedToPublicMethod"] = understand["CountDeclMethodProtected"]/understand["CountDeclMethodPublic"]

# with open("Metrics/metrics_type.csv", "a") as f:
#     for col in understand.columns:
#         f.write(f"{col},size\n")



# --- Jasome Package
jasome_package = pd.read_csv("Metrics/metrics_jasome_package.csv")
# Map Package to Microservice
jasome_package = map_packages(jasome_package)

# Drop unnecessary metrics
jasome_package = jasome_package.drop(columns=["A", "I", "DMS", "CCRC"], errors="ignore")

# Group by microservice and sum all the metric columns
jasome_package = jasome_package.groupby(["MS_system", "Microservice"], as_index=False).sum()
jasome_package.columns = jasome_package.columns.map(lambda x: f"Sum({x})" if x not in ["MS_system", "Microservice"] else x)

# with open("Metrics/metrics_type.csv", "a") as f:
#     for col in jasome_package.columns:
#         f.write(f"{col},\n")

# --- Jasome Class
jasome_class = pd.read_csv("Metrics/metrics_jasome_class.csv")
# Map Package to Microservice
jasome_class =  map_packages(jasome_class)

# Drop unnecessary metrics
jasome_class = jasome_class.drop(columns=["ClRCi", "ClTCi", "TLOC"], errors="ignore")

SUM_COLS = ["Aa", "Ad", "Ait", "Ao", "Av", "HMd", "HMi", "Ma", "Md", "Mi", "Mit", "Mo", "NF", "NM", "NMA",
            "NMI", "NOA", "NOCh", "NOD", "NOL", "NOPa", "NORM", "NPF", "NPM", "NSF", "NSM", "PMd",
            "PMi", "RTLOC", "SIX", "WMC", "LCOM*", "PF"]
AVG_COLS = ["Aa", "Ad", "Ait", "Ao", "Av", "DIT", "HMd", "HMi", "MHF", "MIF", "Ma", "Md", "Mi", "Mit", "Mo",
            "NF", "NM", "NMA", "NMI", "NOA", "NOCh", "NOD", "NOL", "NOPa", "NORM", "NPF", "NPM",
            "NSF", "NSM", "PMR", "PMd", "PMi", "SIX", "AHF", "AIF", "LCOM*", "NMIR", "PF"]
MAX_COLS = ["DIT", "WMC", "LCOM*", "PF"]

sum_df = jasome_class[["MS_system", "Microservice", *SUM_COLS]]
sum_metrics = sum_df.groupby(by=["MS_system", "Microservice"], as_index=False).sum()
sum_metrics.columns = sum_metrics.columns.map(lambda x: f"Sum({x})" if x not in ["MS_system", "Microservice"] else x)

avg_df = jasome_class[["MS_system", "Microservice", *AVG_COLS]]
avg_metrics = avg_df.groupby(by=["MS_system", "Microservice"], as_index=False).mean()
avg_metrics.columns = avg_metrics.columns.map(lambda x: f"Avg({x})" if x not in ["MS_system", "Microservice"] else x)

max_df = jasome_class[["MS_system", "Microservice", *AVG_COLS]]
max_metrics = max_df.groupby(by=["MS_system", "Microservice"], as_index=False).max()
max_metrics.columns = max_metrics.columns.map(lambda x: f"Max({x})" if x not in ["MS_system", "Microservice"] else x)

jasome_class_merged = sum_metrics.merge(avg_metrics, on=["MS_system", "Microservice"])  # Insert back the MS_system column
jasome_class = jasome_class_merged.merge(max_metrics, on=["MS_system", "Microservice"])


# with open("Metrics/metrics_type.csv", "a") as f:
#     for col in jasome_class.columns:
#         f.write(f"{col},\n")

# --- Jasome Method
jasome_method = pd.read_csv("Metrics/metrics_jasome_method.csv")
# Map Package to Microservice
jasome_method =  map_packages(jasome_method)

# Drop unnecessary metrics
jasome_method = jasome_method.drop(columns=["ClRCi", "ClTCi", "TLOC"], errors="ignore")

SUM_COLS = ["Di", "Fin", "Fout", "IOVars", "MCLC", "NBD", "Si", "VG"]
AVG_COLS = ["Di", "Fin", "Fout", "IOVars", "MCLC", "NBD", "Si", "VG"]
MAX_COLS = ["Di", "Fin", "Fout", "IOVars", "MCLC", "NBD", "Si", "VG"]

sum_df = jasome_method[["MS_system", "Microservice", *SUM_COLS]]
sum_metrics = sum_df.groupby(by=["MS_system", "Microservice"], as_index=False).sum()
sum_metrics.columns = sum_metrics.columns.map(lambda x: f"Sum({x})" if x not in ["MS_system", "Microservice"] else x)

avg_df = jasome_method[["MS_system", "Microservice", *AVG_COLS]]
avg_metrics = avg_df.groupby(by=["MS_system", "Microservice"], as_index=False).mean()
avg_metrics.columns = avg_metrics.columns.map(lambda x: f"Avg({x})" if x not in ["MS_system", "Microservice"] else x)

max_df = jasome_method[["MS_system", "Microservice", *AVG_COLS]]
max_metrics = max_df.groupby(by=["MS_system", "Microservice"], as_index=False).max()
max_metrics.columns = max_metrics.columns.map(lambda x: f"Max({x})" if x not in ["MS_system", "Microservice"] else x)

jasome_method_merged = sum_metrics.merge(avg_metrics, on=["MS_system", "Microservice"])  # Insert back the MS_system column
jasome_method = jasome_method_merged.merge(max_metrics, on=["MS_system", "Microservice"])

# with open("Metrics/metrics_type.csv", "a") as f:
#     for col in jasome_method.columns:
#         f.write(f"{col},\n")

# --- SonarQube
sonarqube = pd.read_csv("Metrics/metrics_sonarqube.csv")
sonarqube = map_packages(sonarqube)

rating_cols = ["Sqale rating", "Reliability rating", "Security rating"]
count_cols = [col for col in sonarqube.columns if col not in ["MS_system", "Microservice"] + rating_cols]
sum_df = sonarqube[["MS_system", "Microservice", *count_cols]]
sum_metrics = sum_df.groupby(by=["MS_system", "Microservice"], as_index=False).sum()


avg_df = sonarqube[["MS_system", "Microservice", *rating_cols]]
avg_metrics = sum_df.groupby(by=["MS_system", "Microservice"], as_index=False).mean()
avg_metrics.columns = avg_metrics.columns.map(lambda x: f"Avg({x})" if x not in ["MS_system", "Microservice"] else x)


max_df = sonarqube[["MS_system", "Microservice", *rating_cols]]
max_metrics = max_df.groupby(by=["MS_system", "Microservice"], as_index=False).max()
max_metrics.columns = max_metrics.columns.map(lambda x: f"Max({x})" if x not in ["MS_system", "Microservice"] else x)

sonarqube_merged = sum_metrics.merge(avg_metrics, on=["MS_system", "Microservice"])  # Insert back the MS_system column
sonarqube = sonarqube_merged.merge(max_metrics, on=["MS_system", "Microservice"])
#
# with open("Metrics/metrics_type.csv", "a") as f:
#     for col in sonarqube.columns:
#         f.write(f"{col},quality\n")


# --- Centrality
centrality = pd.read_csv("Metrics/metrics_temporal_centrality.csv")


# --- Total merge

total = understand
total = total.merge(jasome_package, on=["MS_system", "Microservice"], how="left")
total = total.merge(jasome_class, on=["MS_system", "Microservice"], how="left")
total = total.merge(jasome_method, on=["MS_system", "Microservice"], how="left")
total = total.merge(sonarqube, on=["MS_system", "Microservice"], how="left")
total = total.merge(centrality, on=["MS_system", "Microservice"], how="left")
total = total.drop(columns=["Avg(NMIR)", "Avg(PF)", "Max(NMIR)", "Max(PF)"])  # These cause values to be NaN
total = total.dropna()

versions = {
    "train-ticket-0.0.1": 1,
    "train-ticket-0.0.2": 2,
    "train-ticket-0.0.3": 3,
    "train-ticket-0.0.4": 4,
    "train-ticket-0.1.0": 5,
    "train-ticket-0.2.0": 6,
    "train-ticket-1.0.0": 7,
}

total["Version Id"] = total["MS_system"].map(versions)

# Reorder columns to start with system, service
cols = ["MS_system", "Version Id", "Microservice"] + [col for col in total.columns
                                        if col not in ["MS_system","Version Id", "Microservice"]]
total = total[cols]
total = total.sort_values(by=["MS_system", "Microservice"])
total.to_csv("Metrics/metrics_all.csv", index=False, header=True)
