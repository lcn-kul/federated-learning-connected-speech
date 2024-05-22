import logging
import os
from collections import OrderedDict

import flwr as fl
import torch
from dotenv import load_dotenv

from constants import (
    EPOCHS,
    OUTPUT_DIR,
    SERVER_DETAILS_PATH,
)
from utils import initialize_cls_model, load_data, train, test

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

# Initialize model and data loaders
cls_model = initialize_cls_model()
# Load the data
tr_loader, te_loader = load_data()


class ClassificationClient(fl.client.NumPyClient):
    """Flower client for the neurodegenerative disease classification task."""
    def get_parameters(self, config):
        """Return the current parameters of the model, used by the server to obtain the global parameters."""
        return [
                val.cpu().numpy() for _, val in cls_model.classifier.state_dict().items()
            ] + [
                val.cpu().numpy() for _, val in cls_model.roberta.encoder.layer[-1].state_dict().items()
            ]

    def set_parameters(self, parameters):  # noqa
        """Set the parameters of the model through the parameters obtained from the server."""
        parameters_classifier = parameters[: len(cls_model.classifier.state_dict())]
        parameters_transformer = parameters[len(cls_model.classifier.state_dict()):]

        params_dict_cls = zip(cls_model.classifier.state_dict().keys(), parameters_classifier)
        state_dict_cls = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_cls})
        cls_model.classifier.load_state_dict(state_dict_cls, strict=True)

        params_dict_tr = zip(cls_model.roberta.encoder.layer[-1].state_dict().keys(), parameters_transformer)
        state_dict_tr = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_tr})
        cls_model.roberta.encoder.layer[-1].load_state_dict(state_dict_tr, strict=True)

    def fit(self, parameters, config):
        """Train the model on the training set."""
        self.set_parameters(parameters)
        print("Training Started...")
        train(cls_model, tr_loader, epochs=EPOCHS)
        print("Training Finished.")
        return self.get_parameters(config={}), len(tr_loader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the test set."""
        self.set_parameters(parameters)
        loss, all_metrics = test(cls_model, te_loader)
        fl.common.logger.log(msg=all_metrics, level=logging.DEBUG)
        fl.common.logger.log(msg={"loss": loss}, level=logging.DEBUG)
        return float(loss), len(te_loader), all_metrics


# Initialize a logger 
fl.common.logger.configure(identifier="fl-cs", filename=os.path.join(OUTPUT_DIR, "client.log"))

# Start client (training and evaluation)
# Get the server address from the server_details.env file
fl.client.start_client(
    server_address=os.getenv("SERVER_ADDRESS"),
    client=ClassificationClient().to_client(),
    grpc_max_message_length=int(2e9),
)
