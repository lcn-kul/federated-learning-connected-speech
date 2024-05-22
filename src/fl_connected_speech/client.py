import logging
import glob
import os
from collections import OrderedDict

import flwr as fl
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from constants import (
    BATCH_SIZE,
    DEFAULT_ENCODING,
    DEVICE,
    EPOCHS,
    INPUT_DIR,
    LABELS,
    MODEL_BASE,
    OUTPUT_DIR,
    SERVER_DETAILS_PATH,
)
from utils import train, test

# Parameters
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)


def load_data():
    """Load the data for each diagnostic group."""
    raw_datasets = []
    for label in LABELS:
        # Check if the folder is empty
        if len(glob.glob(os.path.join(INPUT_DIR, label, "*.txt"))) == 0:
            continue
        # Load from text files in each label-specific sub-folder
        label_specific_dataset = load_dataset(
            "text",
            data_dir=os.path.join(INPUT_DIR, label),
            download_mode="force_redownload",
            encoding=DEFAULT_ENCODING,
        )["train"]
        # Add 'label' column to each split
        label_specific_dataset = label_specific_dataset.add_column(
            "labels", [label] * len(label_specific_dataset)
        )
        # Add to list of datasets
        raw_datasets.append(label_specific_dataset)

    # Concatenate all datasets into one
    raw_dataset = concatenate_datasets(raw_datasets)
    # Shuffle the entries
    raw_dataset = raw_dataset.shuffle(seed=42)

    # Add label column
    raw_dataset = raw_dataset.cast_column("labels", ClassLabel(num_classes=len(LABELS), names=LABELS))

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    tokenized_datasets = raw_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True), batched=True,
    )
    # Split into train and test (stratified)
    tokenized_datasets = tokenized_datasets.train_test_split(
        test_size=0.3, shuffle=True, seed=42, stratify_by_column="labels",
    )

    tokenized_datasets = tokenized_datasets.remove_columns("text")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    test_loader = DataLoader(
        tokenized_datasets["test"], batch_size=BATCH_SIZE, collate_fn=data_collator,
    )

    return train_loader, test_loader

# Initialize model and data loaders
cls_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=len(LABELS),
).to(DEVICE)
NUM_LAYERS = len(cls_model.roberta.encoder.layer)
# Freeze all model parameters except for the classification head
# and the last layer of the transformer
for name, param in cls_model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    elif str(NUM_LAYERS - 1) in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
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
