# Adapted from https://zenodo.org/records/14748464

import os

import requests
import pandas as pd

USER_TOKEN = "squ_716bdb66681839207f45848b2884859ab60540d5"

METRICS = {"sqale_rating": "Sqale rating",
           "reliability_remediation_effort": "Reliability remediation effort",
           "security_remediation_effort": "Security remediation effort",
           "software_quality_maintainability_remediation_effort": "SQ maintainability remediation effort",
           "software_quality_reliability_remediation_effort": "SQ reliability remediation effort",
           "software_quality_security_remediation_effort": "SQ security remediation effort",
           "bugs": "Bugs",
           "code_smells": "Code smells",
           "cognitive_complexity": "Cognitive complexity",
           "violations": "Violations",
           "effort_to_reach_software_quality_maintainability_rating_a": "Effort to reach rating A",
           "duplicated_blocks": "Duplicated blocks",
           "reliability_rating": "Reliability rating",
           "security_hotspots": "Security hotspots",
           "security_rating": "Security rating",
           "critical_violations": "Critical violations",
           "info_violations": "Info violations",
           "major_violations": "Major violations",
           "minor_violations": "Minor violations"}

COMPONENTS = {}
for folder in os.listdir("Projects"):
    response = requests.get("http://localhost:9000/api/measures/component_tree", params={
        "component": folder,
        "metricKeys": ",".join(METRICS.keys()),
        "ps": 500,
        "qualifiers": "DIR"
    }, auth=(USER_TOKEN, ''))
    r = response.json()
    COMPONENTS[r["baseComponent"]["name"]] = r["components"]

DATA = []
for ms_system, components in COMPONENTS.items():
    for component in components:
        if "src/main/java" not in component["path"]:
            continue
        package = component["path"].split("src/main/java/")[1].replace("/", ".")
        datapoint = {"MS_system": ms_system, "Package": package}
        for metric in component["measures"]:
            datapoint[METRICS[metric["metric"]]] = metric["value"]
        DATA.append(datapoint)
df = pd.DataFrame(DATA)
df = df.sort_values(by="MS_system")
df.to_csv("Metrics/metrics_sonarqube.csv", header=True, index=False)

