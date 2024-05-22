from datasets import load_dataset

from constants import EXTERNAL_PREPROCESSING_SCRIPT

# Load the dataset
external_dataset = load_dataset(EXTERNAL_PREPROCESSING_SCRIPT)["train"]