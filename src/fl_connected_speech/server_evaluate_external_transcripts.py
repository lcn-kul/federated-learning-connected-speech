# Evaluate the model trained with FL in each iteration on external data
import os

from datasets import ClassLabel, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from utils import test
from constants import (
    BATCH_SIZE,
    EXTERNAL_PREPROCESSING_SCRIPT,
    LABELS,
    MODEL_BASE,
    OUTPUT_DIR,
    SERVER_DETAILS_PATH,
)

# Load the server details
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)
# Get the model output dir
saved_model_dir = os.path.join(os.getenv("OUTPUT_DIR"), "picture_description")

# Create a log file for evaluation results
log_file = os.path.join(OUTPUT_DIR, "external_evaluation_results.log")


def load_test_data():
    # Load the dataset
    external_dataset = load_dataset(EXTERNAL_PREPROCESSING_SCRIPT)["train"]
    external_dataset = external_dataset.cast_column("label", ClassLabel(num_classes=len(LABELS), names=LABELS))
    # Remove subject_id column
    external_dataset = external_dataset.remove_columns("subject_id")

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    tokenized_dataset = external_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True), batched=True,
    )
    tokenized_dataset = tokenized_dataset.remove_columns("text")

    # Create a dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    return data_loader

def evaluate(model_dir, test_loader):
    # Get all the subdirectories in the model directory
    subdirs = sorted([os.path.join(model_dir, rnd) for rnd in os.listdir(model_dir)])

    # First write some empty lines to separate the logs
    with open(log_file, "a") as f:
        f.write("\n\n")

    # For each round, evaluate the model on the external dataset
    for subdir in subdirs:
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(subdir)

        # Evaluate the model
        _, results = test(model, test_loader)

        # Then log the results
        with open(log_file, "a") as f:
            f.write(f"Round {subdir.split('_')[-1]} evaluation results on external data: {results}\n")

if __name__ == "__main__":
    # Load the data
    test_loader = load_test_data()

    # Evaluate the model
    evaluate(saved_model_dir, test_loader)
    