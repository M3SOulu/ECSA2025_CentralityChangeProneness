# Adapted from https://zenodo.org/records/14748464

import os
import xml.etree.ElementTree as ET

import pandas as pd

DATA_PACKAGE = []
DATA_CLASS = []
DATA_METHOD = []
for folder in os.scandir(os.path.join(os.getcwd(), "raw_data", "jasome")):
    for f in os.scandir(folder):
        tree = ET.parse(f)
        root = tree.getroot()
        for child_r in root:
            if child_r.tag == "Packages":
                for package in child_r:
                    package_name = package.attrib["name"]
                    for child_p in package:
                        if child_p.tag == "Metrics":
                            data_point = {"MS_system": folder.name,
                                          "Package": package_name}
                            for metric in child_p:
                                data_point[metric.attrib["name"]] = float(metric.attrib["value"].replace(",", "."))
                            DATA_PACKAGE.append(data_point)
                        elif child_p.tag == "Classes":
                            for clas in child_p:
                                class_name = clas.attrib["name"]
                                for child_c in clas:
                                    if child_c.tag == "Metrics":
                                        data_point = {"MS_system": folder.name,
                                                      "Package": package_name,
                                                      "Class": class_name}
                                        for metric in child_c:
                                            data_point[metric.attrib["name"]] = float(metric.attrib["value"].replace(",", "."))
                                        DATA_CLASS.append(data_point)
                                    elif child_c.tag == "Methods":
                                        for method in child_c:
                                            method_name = method.attrib["name"]
                                            for child_m in method:
                                                if child_m.tag == "Metrics":
                                                    data_point = {"MS_system": folder.name,
                                                                  "Package": package_name,
                                                                  "Class": class_name,
                                                                  "Method": method_name}
                                                    for metric in child_m:
                                                        data_point[metric.attrib["name"]] = float(metric.attrib["value"].replace(",", "."))
                                                    DATA_METHOD.append(data_point)

df = pd.DataFrame(DATA_PACKAGE)
df = df.sort_values(by="MS_system")
df.to_csv("Metrics/metrics_jasome_package.csv", index=False, header=True)
df = pd.DataFrame(DATA_CLASS)
df = df.sort_values(by="MS_system")
df.to_csv("Metrics/metrics_jasome_class.csv", index=False, header=True)
df = pd.DataFrame(DATA_METHOD)
df = df.sort_values(by="MS_system")
df.to_csv("Metrics/metrics_jasome_method.csv", index=False, header=True)
