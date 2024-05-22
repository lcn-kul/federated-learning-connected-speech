import logging
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

from constants import (
    LABELS,
    MODEL_BASE,
    N_CLIENTS,
    ROUNDS,
    SERVER_DETAILS_PATH,
)

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

# Initialize mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name=f"federated_learning_connected_speech")

# Load a model (only used to get the parameter state dict)
cls_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=len(LABELS),
)
num_layers = len(cls_model.roberta.encoder.layer)
# Freeze all model parameters except for the classification head
# and the last layer of the transformer
for name, param in cls_model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    elif str(num_layers - 1) in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
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
            params_dict_classifier = zip(
                cls_model.classifier.state_dict().keys(), aggregated_ndarrays[: len(cls_model.classifier.state_dict())]
            )
            state_dict_classifier = {k: torch.tensor(v) for k, v in params_dict_classifier}
            cls_model.classifier.load_state_dict(state_dict_classifier, strict=True)

            params_dict_transformer = zip(
                cls_model.roberta.encoder.layer[-1].state_dict().keys(), aggregated_ndarrays[len(cls_model.classifier.state_dict()):]
            )
            state_dict_transformer = {k: torch.tensor(v) for k, v in params_dict_transformer}
            cls_model.roberta.encoder.layer[-1].load_state_dict(state_dict_transformer, strict=True)

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

    fl.common.logger.log(msg=final_metrics, level=logging.DEBUG)

    return final_metrics


# Initialize a logger 
fl.common.logger.configure(identifier="fl-cs", filename="../../data/output/server.log")

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
