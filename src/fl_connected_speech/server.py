from typing import Tuple, List

import flwr as fl
from flwr.common import Metrics

# Parameters
ROUNDS = 5


# Add a function that aggregates all metrics
def get_weighted_av_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate all the metrics from all clients.

    :param metrics: List of tuples (num_examples, metrics) for each client
    :type metrics: List[Tuple[int, Metrics]]
    :return: A dictionary with the weighted average of each metric
    :rtype: Metrics
    """
    all_metrics = list(metrics[0][1].keys())
    final_metrics = {}
    # Create a weighted average of each metric by number of examples used in each client
    for metric in all_metrics:
        final_metrics[metric] = sum([m[1][metric] * m[0] for m in metrics]) / sum([m[0] for m in metrics])

    return final_metrics


# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=get_weighted_av_metrics),
    grpc_max_message_length=int(2e9),
)
# TODO potentially add some mlflow logging
# TODO save the model
