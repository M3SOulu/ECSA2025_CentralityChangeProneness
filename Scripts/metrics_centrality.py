import os
import tenetan
import pandas
import numpy as np

tempnet = tenetan.networks.SnapshotGraph()
tempnet.load_csv(os.path.join("raw_data", "temp_net", "train-ticket-temporal-v1.0.0.csv"),
                 source_col="source", target_col="target", time_col="version",
                 weight_col="weight", sort_timestamps=True, sort_vertices=True)

temporal_rows = [["MS_system", "Microservice",
         "Taylor_JC", "Yin_JC", "Liu_JC", "Huang_JC",
         "Taylor_CC", "Yin_CC", "Liu_CC", "Huang_CC",
         "Taylor_TAC", "Taylor_FOM", "Taylor_FOM_NORM"
                  ]]

static_rows = [["Microservice",
                "Taylor_MNC", "Yin_MNC", "Liu_MNC", "Huang_MNC",
                ]]

time_rows = [["Version", "Version Id",
              "Taylor_MLC", "Yin_MLC", "Liu_MLC", "Huang_MLC",
              ]]

versions = tempnet.timestamps

taylor = tenetan.centrality.eigenvector.TaylorSupraMatrix(tempnet)
taylor.compute_centrality()
taylor_joint = taylor.joint_centrality
taylor_cc = taylor.cc
taylor_mnc = taylor.mnc
taylor_mlc = taylor.mlc

liu = tenetan.centrality.eigenvector.LiuSupraMatrix(tempnet)
liu.compute_centrality()
liu_joint = liu.joint_centrality
liu_cc = liu.cc
liu_mnc = liu.mnc
liu_mlc = liu.mlc

huang = tenetan.centrality.eigenvector.HuangSupraMatrix(tempnet)
huang.compute_centrality()
huang_joint = huang.joint_centrality
huang_cc = huang.cc
huang_mnc = huang.mnc
huang_mlc = huang.mlc

yin = tenetan.centrality.eigenvector.YinSupraMatrix(tempnet)
yin.compute_centrality()
yin_joint = yin.joint_centrality
yin_cc = yin.cc
yin_mnc = yin.mnc
yin_mlc = yin.mlc

service_mapping_latest = tempnet._vertex_index_mapping


for version_id, version in enumerate(versions):
    tempnet = tenetan.networks.SnapshotGraph()
    tempnet.load_csv(os.path.join("raw_data", "temp_net", f"train-ticket-temporal-{version}.csv"),
                     source_col="source", target_col="target", time_col="version",
                     weight_col="weight", sort_timestamps=True, sort_vertices=True)
    taylor = tenetan.centrality.eigenvector.TaylorSupraMatrix(tempnet)
    taylor.compute_centrality()
    taylor.zero_first_order_expansion()
    taylor_tac = taylor.tac
    taylor_fom = taylor.fom
    fom_norm = np.linalg.norm(taylor_fom)
    taylor_fom_norm = taylor_fom / fom_norm
    service_mapping_current = tempnet._vertex_index_mapping

    time_rows.append([version, version_id+1,
                      # Marginal Layer Centralities
                      abs(float(taylor_mlc[version_id])),
                      abs(float(yin_mlc[version_id])),
                      abs(float(liu_mlc[version_id])),
                      abs(float(huang_mlc[version_id])),
                      ])
    for service, service_id in service_mapping_latest.items():

        new_row = [f"train-ticket-{version[1:]}", service,
               # Joint Centralities
               abs(float(taylor_joint[service_id, version_id])),
               abs(float(yin_joint[service_id, version_id])),
               abs(float(liu_joint[service_id, version_id])),
               abs(float(huang_joint[service_id, version_id])),
               # Conditional Centralities
               abs(float(taylor_cc[service_id, version_id])),
               abs(float(yin_cc[service_id, version_id])),
               abs(float(liu_cc[service_id, version_id])),
               abs(float(huang_cc[service_id, version_id])),
               ]

        if service in service_mapping_current:
            service_id_current = service_mapping_current[service]
            new_row.extend([
                # Time-averaged centralities
                taylor_tac[service_id_current],
                # First-order-mover scores
                taylor_fom[service_id_current],
                taylor_fom_norm[service_id_current],
            ])
        else:
            new_row.extend([0.0, 0.0, 0.0])
        temporal_rows.append(new_row)

        if version_id == 6:
            static_rows.append([service,
                              # Marginal Node Centralities
                              abs(float(taylor_mnc[service_id])),
                              abs(float(yin_mnc[service_id])),
                              abs(float(liu_mnc[service_id])),
                              abs(float(huang_mnc[service_id])),
                              ])

df = pandas.DataFrame(temporal_rows)
df.to_csv("Metrics/metrics_temporal_centrality.csv", index=False, header=False)
df = pandas.DataFrame(static_rows)
df.to_csv("Metrics/metrics_mnc.csv", index=False, header=False)
df = pandas.DataFrame(time_rows)
df.to_csv("Metrics/metrics_mlc.csv", index=False, header=False)
