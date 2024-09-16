import os

from constants import (
    FL_MODEL_BASE,
    OUTPUT_DIR,
)
from utils import initialize_cls_model, load_data, test

log_file = os.path.join(OUTPUT_DIR, "external_results.log")

if __name__ == "__main__":
    # Load the data
    _, test_loader = load_data()

    # Initialize the model that was trained in the FL setup before
    cls_model = initialize_cls_model(model_base=FL_MODEL_BASE)

    # Test on the external and previously unseen test set
    loss, metrics = test(cls_model, test_loader)
    # Log the results
    with open(log_file, "a") as f:
        f.write(f"External evaluation: Loss: {loss}, Metrics: {metrics}\n")
