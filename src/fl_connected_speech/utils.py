import evaluate
import torch
from transformers import AdamW

from constants import DEVICE, METRICS

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