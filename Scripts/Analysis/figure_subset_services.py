import pandas as pd

centrality = pd.read_csv("Metrics/metrics_all.csv")

centrality = centrality[["MS_system", "Version Id", "Microservice", "Taylor_JC", "Taylor_CC", "Liu_JC", "Liu_CC", "Yin_JC", "Yin_CC", "Taylor_FOM", "Taylor_FOM_NORM", "CCP"]]

services = ["ts-admin-travel-service",
            "ts-food-service"
            "ts-order-service",
            "ts-order-other-service",
            "ts-preserve-service",
            "ts-preserve-other-service",
            "ts-rebook-service",
            "ts-route-plan-service",
            "ts-seat-service",
            "ts-travel-service",
            "ts-travel2-service",
            "ts-travel-plan-service"]

centrality = centrality[centrality["Microservice"].isin(services)]
centrality.to_csv("Figures/figure_centrality_subset_services.csv", index=False, header=True)
