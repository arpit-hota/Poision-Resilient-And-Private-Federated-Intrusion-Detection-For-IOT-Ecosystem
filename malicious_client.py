import flwr as fl
import torch
import numpy as np
import joblib
import os
from model import Net
from data_setup import ENCODER_FILE, OUTPUT_FILE
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dynamic Shape Detection — uses shared encoder (no more inconsistency)
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(OUTPUT_FILE):
    raise FileNotFoundError(
        f"'{OUTPUT_FILE}' not found. Run data_setup.py first."
    )

df = pd.read_csv(OUTPUT_FILE)

# FIX: Load shared encoder so class count matches all other files exactly.
# Previously used len(df['label'].unique()) which could differ from LabelEncoder
# if any class appeared in only one partition.
if os.path.exists(ENCODER_FILE):
    encoder     = joblib.load(ENCODER_FILE)
    NUM_CLASSES = len(encoder.classes_)
else:
    from sklearn.preprocessing import LabelEncoder
    encoder     = LabelEncoder()
    encoder.fit(df["label"].values)
    NUM_CLASSES = len(encoder.classes_)

INPUT_DIM = df.shape[1] - 1   # all columns except 'label'
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[PoisonClient] Detected {NUM_CLASSES} classes, {INPUT_DIM} features.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Poison Client
# ─────────────────────────────────────────────────────────────────────────────

class PoisonClient(fl.client.NumPyClient):
    """
    Adversarial client that performs model poisoning by injecting
    high-variance Gaussian noise instead of legitimate trained weights.

    This serves as the attack baseline that Multi-Krum + norm filtering
    in simulation.py is designed to detect and exclude.

    Attack strategy:
    - Noise std=10.0 is deliberately far outside the honest weight
      distribution (~0.001–0.1), making this update easy for Krum to
      flag via pairwise distance scoring.
    - Reports fake sample count of 10,000 to try to inflate influence
      under naive FedAvg (Krum ignores sample counts during selection).

    For a more sophisticated attack, replace the noise with a targeted
    backdoor or scaling attack — but that is beyond this project's scope.
    """

    def __init__(self):
        self.model = Net(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("server_round", "?")

        print(
            f"  😈 [PoisonClient] Round {current_round}: "
            f"injecting high-variance noise (std=10.0)..."
        )

        # Poisoned update: pure Gaussian noise, far outside honest distribution
        poisoned_weights = [
            np.random.normal(0, 10.0, p.shape).astype(np.float32)
            for p in parameters
        ]

        # Fake large sample count — attempts to inflate influence under FedAvg
        # (has no effect under Krum which selects by distance, not sample count)
        return poisoned_weights, 10000, {}

    def evaluate(self, parameters, config):
        # Return 0 examples so server's weighted_average gives zero weight here
        return 0.0, 0, {"accuracy": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚨 Starting adversarial PoisonClient — for research/testing only.")
    # Modern Flower API — no deprecation warnings
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=PoisonClient().to_client(),
    )
