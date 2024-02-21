import os
from typing import Tuple, List, Union, Dict, Optional

import flwr as fl
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from flwr.common import Metrics, FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from transformers import AutoModelForSequenceClassification

# Parameters
load_dotenv(dotenv_path="../../server_details.env")
ROUNDS = 5
MODEL_BASE = "Unbabel/xlm-roberta-comet-small"
N_CLIENTS = int(os.getenv("N_CLIENTS"))  # Number of clients that need to be available to start the round
LABELS = sorted(os.listdir("../../data/input"))

# Initialize mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name=f"federated_learning_connected_speech")

# Load a model (only used to get the parameter state dict)
cls_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=len(LABELS),
)
model_output_dir = os.getenv("OUTPUT_DIR")


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            # Get the current model parameters
            params_dict = zip(cls_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            cls_model.load_state_dict(state_dict, strict=True)

            # Save the model with huggingface
            cls_model.save_pretrained(os.path.join(model_output_dir, "picture_description", f"round_{server_round}"))

        return aggregated_parameters, aggregated_metrics


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

    for metric, value in final_metrics.items():
        mlflow.log_metric(metric, value)

    return final_metrics


# Start server
with mlflow.start_run(run_name="picture_description"):
    fl.server.start_server(
        server_address="0.0.0.0:25565",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=SaveModelStrategy(
            evaluate_metrics_aggregation_fn=get_weighted_av_metrics,
            fraction_evaluate=1.0,
            min_available_clients=N_CLIENTS,
        ),
        grpc_max_message_length=int(2e9),
    )
