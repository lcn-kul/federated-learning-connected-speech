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
cl_client = ClassificationClient()
cid = f"fl-cs-{len(cl_client.tr_loader)}-{len(cl_client.te_loader)}"
fl.common.logger.configure(
    identifier=cid,
    filename=os.path.join(OUTPUT_DIR, "client.log"),
    host=os.getenv("SERVER_ADDRESS"),
)

# Start client (training and evaluation)
# Get the server address from the server_details.env file
fl.client.start_client(
    server_address=os.getenv("SERVER_ADDRESS"),
    client=cl_client.to_client(),
    grpc_max_message_length=int(2e9),
)
