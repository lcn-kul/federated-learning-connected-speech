import glob
import os

import evaluate
import torch
from datasets import load_dataset, concatenate_datasets, ClassLabel
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
    INPUT_DIR,
    LABELS,
    METRICS,
    MODEL_BASE,
)

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


def initialize_cls_model():
    """Initialize the classification model with partially frozen layers."""
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
    return cls_model