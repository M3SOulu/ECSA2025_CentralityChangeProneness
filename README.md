# Replication package and Online Appendix

This is a replication package and online appendix for the ECSA2025 paper "Centrality Change Proneness: an Early Indicator of Microservice Architectural Degradation".

# Contents
This repository contains the following:
- [INSTALL](INSTALL.md): Detailed installation instructions for each used tool
- [Appendix](Appendix):
  - [Background](Appendix/Background.pdf): Extended background section that was not present in the manuscript due to size constraints
- [Projects](Projects): a folder containing source code of all seven studied releases of "train-ticket"
- [Figures](Figures): a folder containing all figures used in the paper
  - [Centrality trajectories](Figures/Figure%201.%20Centrality%20Trajectories.pdf): Figure 1 from the manuscript (temporal centrality trajectories)
  - [Marginal Layer Centrality](Figures/Figure%202.%20MLC.pdf): Figure 2 from the manuscript (MLC)
  - [Correlation with size metrics](Figures/Figure%203.%20Correlation%20with%20size%20metrics.pdf): Figure 3 from the manuscript (Correlation with size metrics heatmap)
  - [Correlation with complexity metrics](Figures/Figure%204.%20Correlation%20with%20complexity%20metrics.pdf): Figure 4 from the manuscript (Correlation with complexity metrics heatmap)
  - [CCP trajectories](Figures/Figure%205.%20CCP%20trajectories.pdf): Figure 5 from the manuscript (CCP trajectories)
- [Raw data](raw_data): a folder containing all raw data extracted from different tools
  - [Code2DFD](raw_data/code2DFD): Raw output of Code2DFD
  - [graph](raw_data/graph): Graphs extracted from Code2DFD output
  - [temp_net](raw_data/temp_net): Temporal networks for each release accumulated since the first one
  - [understand](raw_data/understand): Raw data from Understand
  - [jasome](raw_data/jasome): Raw data from Jasome
  - [package_map.json](raw_data/package_map.json): Mapping of Java packages to microservices
- [Metrics](Metrics): metrics extracted from raw data
  - [metrics_temporal_centrality.csv](Metrics/metrics_temporal_centrality.csv): All the temporal centrality metrics for all microservice
  - [metrics_mnc.csv](Metrics/metrics_mnc.csv): All the Marginal Node Centralities for all Microservices
  - [metrics_mlc.csv](Metrics/metrics_mlc.csv): All the Marginal Layer Centralities for all Microservices
  - [metrics_understand.csv](Metrics/metrics_understand.csv): All the `Understand` metrics for all microservices
  - [metrics_jasome_package.csv](Metrics/metrics_jasome_package.csv): All the `Jasome` metrics for all microservices on package level
  - [metrics_jasome_class.csv](Metrics/metrics_jasome_class.csv): All the `Jasome` metrics for all microservices on class level
  - [metrics_jasome_method.csv](Metrics/metrics_jasome_method.csv): All the `Jasome` metrics for all microservices on method level
  - [metrics_sonarqube.csv](Metrics/metrics_sonarqube.csv): All the `SonarQube` metrics for all microservices
  - [metrics_all.csv](Metrics/metrics_all.csv): All the metrics for all microservices
  - [metrics_type.csv](Metrics/metrics_type.csv): Mapping of all metrics to size/complexity/quality category
- [Results](Results): Data files containing the analyzed results to answer the Research Question
  - [AndersonDarling](Results/AndersonDarling.ods): Results of testing normality of each metric distribution with Anderson-Darling
  - [metrics_non_normal.csv](Results/metrics_non_normal.csv): A subset of all metrics that are not normally distributed
  - [excluded_metrics_AD.txt](Results/excluded_metrics_AD.txt): List of metrics excluded due to Anderson-Darling test (normally distributed)
  - [RQ1](Results/RQ1): Does  temporal centrality correlate with size, complexity, or quality metrics?
    - [Correlation.ods](Results/RQ1/Correlation.ods): Spearman Rho correlation between temporal centrality and software metrics
    - [metrics_size.csv](Results/RQ1/metrics_size.csv): All the size metrics that have a statistically significant correlation with centrality
    - [metrics_complexity.csv](Results/RQ1/metrics_complexity.csv): All the complexity metrics that have a statistically significant correlation with centrality
    - [metrics_quality.csv](Results/RQ1/metrics_quality.csv): All the quality metrics that have a statistically significant correlation with centrality
    - [included_metrics_Spearman.txt](Results/RQ1/included_metrics_Spearman.txt): List of software metrics that are consistently correlated with at least one temporal centrality
    - [excluded_metrics_Spearman.txt](Results/RQ1/excluded_metrics_Spearman.txt): List of software metrics that are not consistently correlated with at least one temporal centrality
  - [RQ2](Results/RQ2): Does centrality correlate with complexity metrics?
    - [FOM_NORM_quartiles.csv](Results/RQ2/FOM_NORM_quartiles.csv): Quartile value for normalized FOM score (to convert to CCP)
    - [Wilcoxon.ods](Results/RQ2/Wilcoxon.ods): Results of Wilcoxon test for CCP and software metrics
    - [FOM_correlation.ods](Results/RQ2/FOM_correlation.ods): Results of Spearman Rho correlation between FOM and software metrics
- [Scripts](Scripts): a folder containing all the scripts
  - [Data-Centrality](Scripts/Data-Centrality): Scripts to construct networks and compute temporal centrality
    - [extract_graphs.py](Scripts/Data-Centrality/extract_graphs.py): processes the `Code2DFD` output into standard graph `json`
    - [make_temp_net.py](Scripts/Data-Centrality/make_temp_net.py): merge the graphs into temporal networks
    - [metrics_centrality.py](Scripts/Data-Centrality/metrics_centrality.py): computes temporal centrality metrics 
  - [Data-JaSoMe](Scripts/Data-JaSoMe): Scripts to run JaSoMe tool
    - [run_jasome.py](Scripts/Data-JaSoMe/run_jasome.py): executes `Jasome` tool and saves raw data
    - [merge_jasome.py](Scripts/Data-JaSoMe/merge_jasome.py): merge the raw `Jasome` data into `csv` files
  - [Data-Understand](Scripts/Data-Understand): Scripts to run Understand tool
    - [run_understand.py](Scripts/Data-Understand/run_understand.py): executes `Understand` tool and saves raw data
    - [merge_understand.py](Scripts/Data-Understand/merge_understand.py): merge the raw `Understand` data into a `csv` file
  - [Data-SonarQube](Scripts/Data-SonarQube): Scripts to run SonarQube tool
    - [run_sonarqube.py](Scripts/Data-SonarQube/run_sonarqube.py): executes `SonarQube` analysis
    - [merge_sonarqube.py](Scripts/Data-SonarQube/merge_sonarqube.py): put the `SonarQube` data into a `csv` file
  - [Analysis](Scripts/Analysis): prepare the data for analysis and answer RQ1 and RQ2
    - [merge_all.py](Scripts/Analysis/merge_all.py): merge all metrics into a single `csv` file
    - [AndersonDarling.py](Scripts/Analysis/AndersonDarling.py): check results of Anderson-Darling test and excluded normal metrics
    - [RQ1.py](Scripts/Analysis/RQ1.py): Count the rejected null hypotheses for Spearman Rho test of RQ1
    - [RQ2.py](Scripts/Analysis/RQ2.py): Count the rejected null hypotheses for Wilcoxon and Spearman Rho tests of RQ2
    - [figure_subset_services.py](Scripts/Analysis/figure_subset_services.py): Select only the data necessary to produce Figures 1 and 5

# License

All generated data is provided under [Creative Commons 4.0 Attribution License](DATA_LICENSE).

All scripts are provided under the [MIT License](SCRIPT_LICENSE).

Analysed train-ticket project [was provided](https://github.com/FudanSELab/train-ticket/blob/master/LICENSE) by the authors under the [Apache-2.0 License](PROJECT_LICENSE).

# Using the replication package

## Preparation and installation

Follow the instructions in [INSTALL](INSTALL.md) to install and configure all used tools.

##  `Code2DFD` and centrality metrics
All DFDs are reconstructed with [c65b4a](https://github.com/tuhh-softsec/code2DFD/tree/c65b4a081ed2ca1618319e5dabf9ecf590988059) version of `Code2DFD` tool.

The raw json output for project version is saved here to [raw_data/code2DFD](raw_data/code2DFD).

### Converting `Code2DFD` to graph and temporal network
The script [extract_graphs.py](Scripts/Data-Centrality/extract_graphs.py) converts the `json` files of the `Code2DFD` output into
standard network json files.

For each `PROJECT`, it creates 2 files in the [raw_data/graph](raw_data/graph) folder:
- `PROJECT-gwcc.json`: The Greatest Weakly Connected Component (GWCC) of the reconstructed architecture graph
- `PROJECT-gwcc_noDB.json`: The GWCC with all databases that are only connected to one service removed

The script [make_temp_net.py](Scripts/Data-Centrality/make_temp_net.py) converts the `json` network files to the `csv` temporal networks, accumulating the releases, in folder [raw_data/temp_net](raw_data/temp_net)

### Computing centrality metrics

The script [metrics_centrality.py](Scripts/Data-Centrality/metrics_centrality.py) loads the temporal network files and computes the following temporal centralities:
- Taylor algorithm:
  - Joint Centrality
  - Conditional Centrality
  - Marginal Node Centrality
  - Marginal Layer Centrality
  - Time-Averaged Centrality (TAC)
  - First-Order Mover score (FOM)
  - L2-normalized FOM
- Yin algorithm:
    - Joint Centrality
    - Conditional Centrality
    - Marginal Node Centrality
    - Marginal Layer Centrality
- Liu algorithm:
    - Joint Centrality
    - Conditional Centrality
    - Marginal Node Centrality
    - Marginal Layer Centrality
- Huang algorithm:
    - Joint Centrality
    - Conditional Centrality
    - Marginal Node Centrality
    - Marginal Layer Centrality
- Closeness centrality
- Betweenness centrality
- Load centrality
- Harmonic centrality
- Information Centrality
- Current flow centrality
- Subgraph centrality

## `Jasome` metrics

`Jasome` tool can be downloaded from its GitHub [page](https://github.com/rodhilton/jasome).

### Raw `Jasome` data
The script [run_jasome.py](Scripts/Data-JaSoMe/run_jasome.py) executes the `Jasome` tool for each `PROJECT`.

Change the variable `JASOME_PATH` in the script to point to the `Jasome` binary on your system.

For each `PROJECT`, the scripts saves to the folder `PROJECT-jasome` the raw `xml` output from `Jasome` for each `src` folder in the project. 

### Merging `Jasome` metrics

The script [merge_jasome.py](Scripts/Data-JaSoMe/merge_jasome.py) takes the data from all the raw `xml`s into the following `csv` files:
- [metrics_jasome_package.csv](Metrics/metrics_jasome_package.csv): metrics calculated for each package
- [metrics_jasome_class.csv](Metrics/metrics_jasome_class.csv): metrics calculated for each class
- [metrics_jasome_method.csv](Metrics/metrics_jasome_method.csv): metrics calculated for each method

## `Understand` metrics

Download the `Understand` tool and acquire its license on the official [website](https://scitools.com/)

### Raw `Understand` data

The script [run_understand.py](Scripts/Data-Understand/run_understand.py) executes the `Understand` tool for each `PROJECT`.

Change the variable `UND_PATH` in the script to point to the `und` [cli tool](https://support.scitools.com/support/solutions/articles/70000582798-using-understand-from-the-command-line-with-und) on your system.

For each `PROJECT`, the scripts saves to the folder `PROJECT-und` the raw `csv` output from `Understand`.

### Merging `Understand` metrics

The script [merge_understand.py](Scripts/Data-Understand/merge_understand.py) takes only the metrics calculated on `Package` level for all
`PROJECTS` and saves them to [metrics_understand.csv](Metrics/metrics_understand.csv) `csv` file.


## `SonarQube` metrics

Deploy a `SonarQube` instance using the instructions from the [official website](https://docs.sonarsource.com/sonarqube/latest/setup-and-upgrade/install-the-server/introduction/).

Generate a `Global Analysis Token` and a `User token`.

Download the `SonarScanner` application from the [official website](https://docs.sonarsource.com/sonarqube/9.9/analyzing-source-code/scanners/sonarscanner/).

### Raw `SonarQube` data

The script [run_sonarqube.py](Scripts/Data-SonarQube/run_sonarqube.py) sets up a `SonarQube` project for each of the `PROJECT`s
in the repository and executes the analysis with `SonarScanner`.

Change the `SONAR_PATH` variable to the location of the `sonar-scanner` binary.

Change the `TOKEN` variable to the `Global Analysis Token` generated in `SonarQube`.

Additionally, if `SonarQube` is not deployed on `localhost:9000`, change the `-Dsonar.host.url` parameter in the run command.

After executing the script, you should see all projects analyzed in the `SonarQube` dashboard.

### Merging `SonarQube` data

The script [merge_sonarqube.py](Scripts/Data-SonarQube/merge_sonarqube.py) queries data for each `PROJECT`.

Change the variable `USER_TOKEN` to the `User token` generated in `SonarQube`.

The script will query the `SonarQube` metrics on directory level, infer the package name from the directory path,
and save the metrics for each `PROJECT` and each package in the [metrics_sonarqube.csv](Metrics/metrics_sonarqube.csv).

## Merging all data

The file [package_map.json](raw_data/package_map.json) contains the mapping of Java packages to the microservices.

The script [merge_all.py](Scripts/Analysis/merge_all.py) takes the metric files of individual files, maps the packages to microservices
and creates a unified `csv` file [metrics_all.csv](Metrics/metrics_all.csv) with microservices that have all possible metrics.

Metrics are aggregated from packages using `sum`, `mean` and `max` wherever suitable.

## Performing analysis for the RQs

The file [AndersonDarling.py](Scripts/Analysis/AndersonDarling.py) checks the hypotheses of the Anderson-Darling
normality tests and keeps only the metrics are are **not** normally distributed to the file
[metrics_non_normal.csv](Results/metrics_non_normal.csv).

The file [RQ1.py](Scripts/Analysis/RQ1.py) checks the hypotheses for the Spearman Rho test
of RQ1 and counts the amount of rejected hypotheses as well as keeps only the metrics
that have consistently a statistically significant correlation with at least one
temporal centrality for all releases.

The file [RQ2.py](Scripts/Analysis/RQ2.py) checks the hypotheses for Wilcoxon and Spearman Rho tests
for RQ2 and count the amount of rejected hypotheses.