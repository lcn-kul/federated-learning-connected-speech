import os

import flwr as fl
import mlflow
from dotenv import load_dotenv

from constants import (
    N_CLIENTS,
    ROUNDS,
    SERVER_DETAILS_PATH,
    OUTPUT_DIR,
)
from utils import get_weighted_av_metrics, SaveModelStrategy

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

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
