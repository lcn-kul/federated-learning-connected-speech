import glob
import os
from collections import OrderedDict

import evaluate
import flwr as fl
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# Parameters
MODEL_BASE = "xlm-roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5
METRICS = ["accuracy", "precision", "recall", "f1"]
LABELS = sorted(os.listdir("../../data/input"))
label_column = ClassLabel(names=LABELS)


def load_data():
    """Load the data for each diagnostic group."""
    raw_datasets = []
    for label in LABELS:
        # Check if the folder is empty
        if len(glob.glob(f"../../data/input/{label}/*.txt")) == 0:
            continue
        # Load from text files in each label-specific sub-folder
        label_specific_dataset = load_dataset(
            "text",
            data_dir=f"../../data/input/{label}",
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
        test_size=0.2, shuffle=True, seed=42, stratify_by_column="labels",
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


def train(model, train_loader, epochs):
    """Train the model for a given number of epochs."""
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
        predictions = torch.argmax(logits, dim=-1)
        evaluations.append({"predictions": predictions, "references": batch["labels"]})

    loss /= len(test_loader.dataset)
    # Compute each metric
    for metric in METRICS:
        metric_func = evaluate.load(metric)
        for ev in evaluations:
            metric_func.add_batch(predictions=ev["predictions"], references=ev["references"])
        if metric == "accuracy":
            all_metrics[metric] = metric_func.compute()[metric]
        else:
            all_metrics[metric] = metric_func.compute(labels=range(len(LABELS)), average="micro")[metric]
    print(all_metrics)
    return loss, all_metrics


# Initialize model and data loaders
cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE, num_labels=len(LABELS)).to(DEVICE)
tr_loader, te_loader = load_data()


class ClassificationClient(fl.client.NumPyClient):
    """Flower client for the neurodegenerative disease classification task."""
    def get_parameters(self, config):
        """Return the current parameters of the model, used by the server to obtain the global parameters."""
        return [val.cpu().numpy() for _, val in cls_model.state_dict().items()]

    def set_parameters(self, parameters):  # noqa
        """Set the parameters of the model through the parameters obtained from the server."""
        params_dict = zip(cls_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        cls_model.load_state_dict(state_dict, strict=True)

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
        return float(loss), len(te_loader), all_metrics


# Start client (training and evaluation)
# Get the server address from the .env file
load_dotenv()
fl.client.start_client(
    server_address=os.getenv("SERVER_ADDRESS"),
    client=ClassificationClient().to_client(),
    grpc_max_message_length=int(2e9),
)
