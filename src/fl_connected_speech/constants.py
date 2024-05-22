# Set up a few constants here
import os 

# Set up the base directory for the project
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Src and data directories
SRC_DIR = os.path.join(PROJECT_DIR, "fl_connected_speech")
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

# Server details
SERVER_DETAILS_PATH = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "server_details.env")

# Path to a preprocessing script
EXTERNAL_PREPROCESSING_SCRIPT = os.path.join(SRC_DIR, "preprocess_external_transcripts.py")
