import json
import csv
import os

import networkx as nx

v = ["v0.0.1", "v0.0.2", "v0.0.3", "v0.0.4", "v0.1.0", "v0.2.0", "v1.0.0"]

to_csv = [["source", "target", "version", "weight"]]
for version in v:
    with open(os.path.join("raw_data", "graph", f"train-ticket-{version}_gwcc_noDB.json"), 'r') as f:
        d = json.load(f)
    G = nx.node_link_graph(d, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                           multigraph=False, directed=True)  # Load the graph
    for source, target in G.edges():
        if not source.startswith("ts") or not target.startswith("ts"):
            continue
        to_csv.append([source, target, version, 1])

with open(os.path.join("raw_data", "train-ticket-temporal.csv"), 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(to_csv)
