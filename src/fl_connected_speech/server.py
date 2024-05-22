import os
from typing import Tuple, List, Union, Dict, Optional

import flwr as fl
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from flwr.common import FitRes, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy

from constants import (
    N_CLIENTS,
    ROUNDS,
    SERVER_DETAILS_PATH,
    OUTPUT_DIR,
)
from utils import initialize_cls_model, get_weighted_av_metrics

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

# Initialize mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name=f"federated_learning_connected_speech")

# Load a model (only used to get the parameter state dict)
cls_model = initialize_cls_model()
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


# Initialize a logger 
fl.common.logger.configure(identifier="fl-cs", filename=os.path.join(OUTPUT_DIR, "server.log"))

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
