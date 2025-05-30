# Installation guide

This files contains instructions on how to install and run each tool used in the data gathering process.

## Code2DFD

Download `Code2DFD` from its official [repository](https://github.com/tuhh-softsec/code2DFD).

Data presented in this package/paper were generated with the [c65b4a](https://github.com/tuhh-softsec/code2DFD/tree/c65b4a081ed2ca1618319e5dabf9ecf590988059) version of the tool.

Create a separate virtual environment where you will install and run the tool.

Install its dependencies from [requirements.txt](https://github.com/tuhh-softsec/code2DFD/blob/c65b4a081ed2ca1618319e5dabf9ecf590988059/requirements.txt) as

```
python -m pip install -r requirements.txt
```

Additionally, install the following dependencies:

```
python -m pip instal RUAMEL.yaml
python -m pip install six
```

The tool can be executed as 

```
python -m code2DFD.py --repo_url PATH_TO_REPO
```

in the cloned directory, where `PATH_TO_REPO` is a path to the repository that should be analysed (one of the directories in [Projects](Projects)).

The tool will generate a folder `code2DFD_output/REPO_NAME/COMMIT` with the results for a specific commit from a specific repo.

This folder is provided in [raw_data/code2DFD_output](raw_data/code2DFD_output).

Move your own generated folder there if you want to test the rest of the package with the newly generated data.


## Understand by SciTools

Download and install Understand from the [official website](https://licensing.scitools.com/download).

The data in this package are generated using version `6.5.1201`.

Request an [Educational/Academic license](https://scitools.com/student) at the appropriate step.

Change the variable `UND_PATH` in the [Understand script](Scripts/Data-Understand/run_understand.py) to point to the `und` [cli tool](https://support.scitools.com/support/solutions/articles/70000582798-using-understand-from-the-command-line-with-und) on your system.

Use the [Understand script](Scripts/Data-Understand/run_understand.py) to generate the data. See instructions in [README](README.md#understand-metrics).

## Jasome

Download the latest release of `Jasome` from official [GitHub page](https://github.com/rodhilton/jasome/releases).

The data in these package are generated using `v0.6.8-alpha`.

Unzip the downloaded release and locate the `bin` subdirectory.

Change the variable `JASOME_PATH` in the [Jasome script](Scripts/Data-JaSoMe/run_jasome.py) to point to the `Jasome` binary on your system (inside `bin` subfolder). 

Use the [Jasome script](Scripts/Data-JaSoMe/run_jasome.py) to generate the data. See instructions in [README](README.md#jasome-metrics).

## SonarQube

NOTE: There are many ways to deploy `SonarQube`, we provide the most simple way to achieve necessary results.

Install [Docker](https://www.docker.com/).

Deploy `SonarQube` using the example [docker-compose.yml](https://github.com/SonarSource/docker-sonarqube/blob/master/example-compose-files/sq-with-postgres/docker-compose.yml) as:


```
docker compose -f docker-compose.yml up -d
```

Navigate to [http://localhost:9000](http://localhost:9000). For the first login, credentials are `admin/admin`, then set your own.

Navigate to `Profile` -> `My Account` -> `Security` -> `Token Generation`.

Generate a `Global Analysis Token` and a `User token`.

Download the `SonarScanner` application from the [official website](https://docs.sonarsource.com/sonarqube/9.9/analyzing-source-code/scanners/sonarscanner/).

In the [SonarQube script](Scripts/Data-SonarQube/run_sonarqube.py):
- Change the `SONAR_PATH` variable to the location of the `sonar-scanner` binary.
- Change the `TOKEN` variable to the `Global Analysis Token` generated in `SonarQube`.

In the [SonarQube merge script](Scripts/Data-SonarQube/merge_sonarqube.py):
- Change the variable `USER_TOKEN` to the `User token` generated in `SonarQube`.

Use the [SonarQube script](Scripts/Data-SonarQube/run_sonarqube.py) to generate the data. See instructions in [README](README.md#sonarqube-metrics).

## Working with the package scripts

The code has been developed for Python 3.10.

Install the necessary Python packages using the [requirements.txt](requirements.txt) file as:
```
python -m pip install -r requirements.txt
```

Follow the instructions in [README](README.md) to replicate each step of the data collection.