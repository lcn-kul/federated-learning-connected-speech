import glob
import logging
import os
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import evaluate
import flwr as fl
import mlflow
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from flwr.common import FitRes, Metrics, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
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
    METRICS,
    MODEL_BASE,
)

# Initialize mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(experiment_name=f"federated_learning_connected_speech")


def train(model, train_loader, epochs):
    """Train the model for a given number of epochs."""
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    print("Local training started...")
    model.train()

    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("Local training finished.")


def test(model, test_loader):
    """Evaluate the trained model on the test set."""
    # Initialize the evaluation variables
    evaluations = []
    all_metrics = {}
    loss = 0
    model.eval()

    # Get scores for each batch
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        probabilities_pos_class = torch.softmax(logits, axis=-1)[:, 1]
        predictions = torch.argmax(logits, dim=-1)
        evaluations.append({"probs": probabilities_pos_class, "predictions": predictions, "references": batch["labels"]})

    loss /= len(test_loader.dataset)
    # Compute each metric
    for metric in METRICS:
        if metric == "youden":
            metric_func = evaluate.load("helena-balabin/youden_index")
            for ev in evaluations:
                metric_func.add_batch(prediction_scores=ev["probs"], references=ev["references"])
            youden_metrics = {f"{metric}_{key}": value for key, value in metric_func.compute().items()}
            all_metrics.update(youden_metrics)
        else:
            metric_func = evaluate.load(metric)
            for ev in evaluations:
                metric_func.add_batch(predictions=ev["predictions"], references=ev["references"])
            all_metrics[metric] = metric_func.compute()[metric]
    
    print(all_metrics)
    return loss, all_metrics


def load_data():
    """Load the data for each diagnostic group."""
    # Initialize the tokenizer beforehand
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    
    loaders = []
    for partition in ["train", "test"]:
        raw_datasets = []
        for label in LABELS:
            # Check if the folder is empty
            if len(glob.glob(os.path.join(INPUT_DIR, partition, label, "*.txt"))) == 0:
                continue
            # Load from text files in each label-specific sub-folder
            label_specific_dataset = load_dataset(
                "text",
                data_dir=os.path.join(INPUT_DIR, partition, label),
                download_mode="force_redownload",
                encoding=DEFAULT_ENCODING,
            )["train"]
            # Add 'label' column to each split
            label_specific_dataset = label_specific_dataset.add_column(
                "labels", [label] * len(label_specific_dataset)
            )
            # Add to list of datasets
            raw_datasets.append(label_specific_dataset)

        if len(raw_datasets) > 0:
            # Concatenate all datasets into one
            raw_dataset = concatenate_datasets(raw_datasets)
            # Shuffle the entries
            raw_dataset = raw_dataset.shuffle(seed=42)

            # Add label column
            raw_dataset = raw_dataset.cast_column("labels", ClassLabel(num_classes=len(LABELS), names=LABELS))

            # Tokenize the dataset
            tokenized_dataset = raw_dataset.map(
                lambda examples: tokenizer(examples["text"], truncation=True), batched=True,
            )
            tokenized_dataset = tokenized_dataset.remove_columns("text")

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            loaders.append(
                DataLoader(
                    tokenized_dataset,
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    collate_fn=data_collator,
                )
            )
        else:
            loaders.append(None)

    return loaders[0], loaders[1]


def initialize_cls_model(model_base=MODEL_BASE):
    """Initialize the classification model with partially frozen layers."""
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        model_base,
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
    return cls_model


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


class ClassificationClient(fl.client.NumPyClient):
    """Flower client for the neurodegenerative disease classification task."""
    # Create an that adds the model_output_dir + cls_model + data
    def __init__(self, *args, **kwargs):
        """Initialize the client with the model and the model output directory."""
        super().__init__(*args, **kwargs)
        # Load a model (only used to get the parameter state dict)
        self.cls_model = initialize_cls_model()
        self.model_output_dir = os.getenv("OUTPUT_DIR")
        self.tr_loader, self.te_loader = load_data()


    def get_parameters(self, config):
        """Return the current parameters of the model, used by the server to obtain the global parameters."""
        return [
                val.cpu().numpy() for _, val in self.cls_model.classifier.state_dict().items()
            ] + [
                val.cpu().numpy() for _, val in self.cls_model.roberta.encoder.layer[-1].state_dict().items()
            ]

    def set_parameters(self, parameters):  # noqa
        """Set the parameters of the model through the parameters obtained from the server."""
        parameters_classifier = parameters[: len(self.cls_model.classifier.state_dict())]
        parameters_transformer = parameters[len(self.cls_model.classifier.state_dict()):]

        params_dict_cls = zip(self.cls_model.classifier.state_dict().keys(), parameters_classifier)
        state_dict_cls = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_cls})
        self.cls_model.classifier.load_state_dict(state_dict_cls, strict=True)

        params_dict_tr = zip(self.cls_model.roberta.encoder.layer[-1].state_dict().keys(), parameters_transformer)
        state_dict_tr = OrderedDict({k: torch.Tensor(v) for k, v in params_dict_tr})
        self.cls_model.roberta.encoder.layer[-1].load_state_dict(state_dict_tr, strict=True)

    def fit(self, parameters, config):
        """Train the model on the training set."""
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.cls_model, self.tr_loader, epochs=EPOCHS)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.tr_loader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the test set."""
        self.set_parameters(parameters)
        loss, all_metrics = test(self.cls_model, self.te_loader)
        fl.common.logger.log(msg=all_metrics, level=logging.DEBUG)
        fl.common.logger.log(msg={"loss": loss}, level=logging.DEBUG)
        return float(loss), len(self.te_loader), all_metrics
    

class SaveModelStrategy(fl.server.strategy.FedAvg):
    # Create an init that passes everything to the parent class and adds the model_output_dir + cls_model
    def __init__(self, *args, **kwargs):
        """Initialize the client with the model and the model output directory."""
        super().__init__(*args, **kwargs)
        # Load a model (only used to get the parameter state dict)
        self.cls_model = initialize_cls_model()
        self.model_output_dir = os.getenv("OUTPUT_DIR")

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
                self.cls_model.classifier.state_dict().keys(), aggregated_ndarrays[: len(self.cls_model.classifier.state_dict())]
            )
            state_dict_classifier = {k: torch.tensor(v) for k, v in params_dict_classifier}
            self.cls_model.classifier.load_state_dict(state_dict_classifier, strict=True)

            params_dict_transformer = zip(
                self.cls_model.roberta.encoder.layer[-1].state_dict().keys(), aggregated_ndarrays[len(self.cls_model.classifier.state_dict()):]
            )
            state_dict_transformer = {k: torch.tensor(v) for k, v in params_dict_transformer}
            self.cls_model.roberta.encoder.layer[-1].load_state_dict(state_dict_transformer, strict=True)

            # Save the model with huggingface
            self.cls_model.save_pretrained(os.path.join(self.model_output_dir, "picture_description", f"round_{server_round}"))

        return aggregated_parameters, aggregated_metrics
