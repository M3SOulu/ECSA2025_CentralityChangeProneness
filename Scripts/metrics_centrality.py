import os
import tenetan
import pandas

tempnet = tenetan.networks.SnapshotGraph()
tempnet.load_csv(os.path.join("raw_data", "train-ticket-temporal.csv"),
                 source_col="source", target_col="target", time_col="version",
                 weight_col="weight", sort_timestamps=True, sort_vertices=True)

temporal_rows = [["MS_system", "Microservice",
         "Taylor_JC", "Yin_JC", "Liu_JC", "Huang_JC",
         "Taylor_CC", "Yin_CC", "Liu_CC", "Huang_CC",
         "Taylor_MNC", "Yin_MNC", "Liu_MNC", "Huang_MNC",
         "Taylor_MLC", "Yin_MLC", "Liu_MLC", "Huang_MLC",
         "Taylor_TAC", "Taylor_FOM"
                  ]]

static_rows = [["Microservice",
                "Taylor_MNC", "Yin_MNC", "Liu_MNC", "Huang_MNC",
                "Taylor_TAC", "Taylor_FOM"
                ]]

time_rows = [["Version", "Version Id",
              "Taylor_MLC", "Yin_MLC", "Liu_MLC", "Huang_MLC",
              ]]

versions = tempnet.timestamps
services = tempnet.vertices

taylor = tenetan.centrality.eigenvector.TaylorSupraMatrix(tempnet)
taylor.compute_centrality()
taylor_joint = taylor.joint_centrality
taylor_cc = taylor.cc
taylor_mnc = taylor.mnc
taylor_mlc = taylor.mlc
taylor.zero_first_order_expansion()
taylor_tac = taylor.tac
taylor_fom = taylor.fom

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

for version_id, version in enumerate(versions):
    time_rows.append([version, version_id+1,
                      # Marginal Layer Centralities
                      abs(float(taylor_mlc[version_id])),
                      abs(float(yin_mlc[version_id])),
                      abs(float(liu_mlc[version_id])),
                      abs(float(huang_mlc[version_id])),
                      ])
    for service_id, service in enumerate(services):
        temporal_rows.append([f"train-ticket-{version[1:]}", service,
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
                              # Marginal Node Centralities
                              abs(float(taylor_mnc[service_id])),
                              abs(float(yin_mnc[service_id])),
                              abs(float(liu_mnc[service_id])),
                              abs(float(huang_mnc[service_id])),
                              # Marginal Layer Centralities
                              abs(float(taylor_mlc[version_id])),
                              abs(float(yin_mlc[version_id])),
                              abs(float(liu_mlc[version_id])),
                              abs(float(huang_mlc[version_id])),
                              # Time-averaged centralities
                              taylor_tac[service_id],
                              # First-order-mover scores
                              taylor_fom[service_id],
                              ])
        if version_id == 0:
            static_rows.append([service,
                              # Marginal Node Centralities
                              abs(float(taylor_mnc[service_id])),
                              abs(float(yin_mnc[service_id])),
                              abs(float(liu_mnc[service_id])),
                              abs(float(huang_mnc[service_id])),
                              # Time-averaged centralities
                              taylor_tac[service_id],
                              # First-order-mover scores
                              taylor_fom[service_id],
                              ])

df = pandas.DataFrame(temporal_rows)
df.to_csv("Metrics/metrics_temporal_centrality.csv", index=False, header=False)
df = pandas.DataFrame(static_rows)
df.to_csv("Metrics/metrics_static_centrality.csv", index=False, header=False)
df = pandas.DataFrame(time_rows)
df.to_csv("Metrics/metrics_layer_centrality.csv", index=False, header=False)
