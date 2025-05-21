import json
import os

import networkx as nx

for version in os.scandir(os.path.join(os.getcwd(), "raw_data", "code2DFD_output")):
    for commit in os.scandir(version):
        for file in os.scandir(commit):
            if not file.path.endswith("architecture.json"):  # Get the json format
                continue
            file_path = file.path
            with open(file_path, 'r') as f:
                g = json.load(f)
            g = {
                "nodes": g["microservices"] + g["external_components"],  # Merge all nodes
                "edges": g["information_flows"]
            }
            G = nx.node_link_graph(g, edges="edges", nodes="nodes", name="name", source="sender", target="receiver",
                   multigraph=False, directed=True)  # Load the graph

            # Filter Greatest Weakly connected component
            gcc = max(nx.weakly_connected_components(G), key=len)  # Get the nodes in the weakly g.c.c.
            not_gcc = set(G.nodes()) - set(gcc)  # Get the set of nodes that are not in the g.c.c
            G.remove_nodes_from(not_gcc)  # Keep only the g.c.c.
            g = nx.node_link_data(G, edges="edges", nodes="nodes", name="name", source="sender", target="receiver")
            del g["multigraph"]
            del g["directed"]
            del g["graph"]
            name = version.name
            output_folder = os.path.join(os.getcwd(), "raw_data", "graph")
            os.makedirs(output_folder, exist_ok=True)
            gwcc_path = os.path.join(output_folder, f"{name}_gwcc.json")
            with open(gwcc_path, 'w') as f:
                json.dump(g, f, indent=4, sort_keys=True)

            # Remove DB only connected to own service
            nodes = set(G.nodes)
            for node in nodes:
                if (("mongo" in node.casefold() or "mysql" in node.casefold() or "database" in node.casefold())
                        and G.in_degree(node)+G.out_degree(node) == 1):
                    G.remove_node(node)
            g = nx.node_link_data(G, edges="edges", nodes="nodes", name="name", source="sender", target="receiver")
            del g["multigraph"]
            del g["directed"]
            del g["graph"]
            gwcc_nodb_path = os.path.join(output_folder, f"{name}_gwcc_noDB.json")
            with open(gwcc_nodb_path, 'w') as f:
                json.dump(g, f, indent=4, sort_keys=True)
