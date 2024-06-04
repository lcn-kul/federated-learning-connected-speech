# Set up a few constants here
from dotenv import load_dotenv
import os

import torch

# Set up the base directory for the project
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Src and data directories
SRC_DIR = os.path.join(PROJECT_DIR, "fl_connected_speech")
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

# Server details
SERVER_DETAILS_PATH = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "server_details.env")
load_dotenv(dotenv_path=SERVER_DETAILS_PATH)

# Path to a preprocessing script
EXTERNAL_PREPROCESSING_SCRIPT = os.path.join(SRC_DIR, "server_preprocess_external_transcripts.py")

# Model, device and data details
MODEL_BASE = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
# recall = sensitivity, precision = positive predictive value
# precision of the negative class = negative predictive value
# recall of the negative class = specificity
METRICS = ["accuracy", "precision", "recall", "f1", "youden"]
# Invert labels so that healthy = 0
LABELS = sorted(os.listdir(os.path.join(INPUT_DIR, "train")))[::-1]
# Number of epochs per round
EPOCHS = 10
# Find out the OS that the client is running on
DEFAULT_ENCODING = "ISO-8859-1" if os.name == "nt" else "utf-8"
# NUmber of FL rounds
ROUNDS = 10
# Number of clients that need to be available to start the round
N_CLIENTS = int(os.getenv("N_CLIENTS"))