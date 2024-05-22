# Evaluate the model trained with FL in each iteration on external data
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

from constants import (
    BATCH_SIZE,
    EXTERNAL_PREPROCESSING_SCRIPT,
    LABELS,
    MODEL_BASE,
)


def load_data():
    # Load the dataset
    external_dataset = load_dataset(EXTERNAL_PREPROCESSING_SCRIPT)["train"]
    external_dataset = external_dataset.cast_column("labels", ClassLabel(num_classes=len(LABELS), names=LABELS))

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    tokenized_dataset = external_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True), batched=True,
    )

    tokenized_dataset = tokenized_dataset.remove_columns("text")


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    return test_loader