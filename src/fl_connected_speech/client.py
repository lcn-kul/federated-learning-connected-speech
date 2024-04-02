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
load_dotenv(dotenv_path="../../server_details.env")
MODEL_BASE = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
# recall = sensitivity, precision = positive predictive value
# precision of the negative class = negative predictive value
# recall of the negative class = specificity
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "npv", "specificity"]
# Invert labels so that healthy = 0
LABELS = sorted(os.listdir("../../data/input"))[::-1]
EPOCHS = 10
# Find out the OS that the client is running on
DEFAULT_ENCODING = "ISO-8859-1" if os.name == "nt" else "utf-8"


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
        if metric == "npv":
            metric_func = evaluate.load("precision")
        elif metric == "specificity":
            metric_func = evaluate.load("recall")
        else:
            metric_func = evaluate.load(metric)
        
        if metric == "roc_auc":
            for ev in evaluations:
                metric_func.add_batch(prediction_scores=ev["probs"], references=ev["references"])
        else:
            for ev in evaluations:
                metric_func.add_batch(predictions=ev["predictions"], references=ev["references"])
        
        if metric == "npv":
            all_metrics[metric] = metric_func.compute(pos_label=0)["precision"]
        elif metric == "specificity":
            all_metrics[metric] = metric_func.compute(pos_label=0)["recall"]
        else:
            all_metrics[metric] = metric_func.compute()[metric]
    
    print(all_metrics)
    return loss, all_metrics


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
        return float(loss), len(te_loader), all_metrics


# Initialize a logger 
fl.common.logger.configure(identifier="fl-cs", filename="../../data/output/client.log")

# Start client (training and evaluation)
# Get the server address from the server_details.env file
fl.client.start_client(
    server_address=os.getenv("SERVER_ADDRESS"),
    client=ClassificationClient().to_client(),
    grpc_max_message_length=int(2e9),
)
