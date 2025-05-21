import json
import csv
import os

v = ["v0.0.1", "v0.0.2", "v0.0.3", "v0.0.4", "v0.1.0", "v0.2.0", "v1.0.0"]

to_csv = [["source", "target", "version", "weight"]]
# Accumulate TNs by version
for version in v:
    with open(os.path.join("raw_data", "graph", f"train-ticket-{version}.json"), 'r') as f:
        d = json.load(f)
    for edge_dict in d["edges"]:
        source = edge_dict["sender"]
        target = edge_dict["receiver"]
        # services can only start with 'ts'
        if not source.startswith("ts") or not target.startswith("ts"):
            continue
        to_csv.append([source, target, version, 1])

    # Save currently accumulated TN
    with open(os.path.join("raw_data","temp_net", f"train-ticket-temporal-{version}.csv"), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(to_csv)
