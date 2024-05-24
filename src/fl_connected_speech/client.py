import os

import flwr as fl
from dotenv import load_dotenv

from constants import (
    OUTPUT_DIR,
    SERVER_DETAILS_PATH,
)
from utils import ClassificationClient

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

# Initialize a logger 
fl.common.logger.configure(identifier="fl-cs", filename=os.path.join(OUTPUT_DIR, "client.log"))

# Start client (training and evaluation)
# Get the server address from the server_details.env file
fl.client.start_client(
    server_address=os.getenv("SERVER_ADDRESS"),
    client=ClassificationClient().to_client(),
    grpc_max_message_length=int(2e9),
)
