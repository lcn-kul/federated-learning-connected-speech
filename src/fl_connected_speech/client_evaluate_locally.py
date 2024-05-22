import os

from constants import (
    EPOCHS,
    ROUNDS,
    OUTPUT_DIR,
)
from utils import initialize_cls_model, load_data, train, test

log_file = os.path.join(OUTPUT_DIR, "client_local_results.log")

if __name__ == "__main__":
    # Load the data
    train_loader, test_loader = load_data()

    # Initialize the model 
    cls_model = initialize_cls_model()

    # Train the model for EPOCHS x ROUNDS (same as in FL but local model only)
    for i in range(ROUNDS):
        # Train
        train(cls_model, train_loader, EPOCHS)
        # Test on the test set (same test set as in FL)
        loss, metrics = test(cls_model, test_loader)
        # Log the results
        with open(log_file, "a") as f:
            f.write(f"Round {i+1}: Loss: {loss}, Metrics: {metrics}\n")