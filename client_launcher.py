import sys
import flwr as fl
import torch
import joblib
import os
from model import Net
from data_setup import load_data_noniid, load_data, ENCODER_FILE
from simulation import IDSClient
from gemini_copilot import GeminiCoPilot


# ─────────────────────────────────────────────────────────────────────────────
# Federated Client Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Initialises the model and data, then starts a Flower federated client.

    Usage:
        python client_launcher.py <client_id>

    Example (3 terminals):
        python client_launcher.py 0
        python client_launcher.py 1
        python client_launcher.py 2

    Each client gets a different non-IID data partition (Dirichlet alpha=0.9).
    Gemini Co-Pilot is initialised here and passed into IDSClient.
    If Gemini is unavailable the client runs exactly as before — no impact.
    """

    # 1. Parse client ID from command line
    client_id   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_clients = 3
    print(f"Starting Client {client_id} of {num_clients}")

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load non-IID data partition for this client
    #    alpha=0.9 — mildly heterogeneous, prevents extreme class starvation
    trainloader, testloader, input_dim, num_classes = load_data_noniid(
        client_id=client_id,
        num_clients=num_clients,
        batch_size=32,
        alpha=0.9,
        use_smote=True,
    )
    print(f"input_dim={input_dim}, num_classes={num_classes}")

    # 4. Load shared class names for per-class F1 reporting
    class_names = None
    if os.path.exists(ENCODER_FILE):
        enc         = joblib.load(ENCODER_FILE)
        class_names = list(enc.classes_)

    # 5. Initialise Gemini Co-Pilot
    #    Reads GEMINI_API_KEY from environment variable.
    #    If missing or SDK not installed → copilot.enabled=False → no-op each round.
    #    Training is NEVER affected regardless of Gemini status.
    copilot = GeminiCoPilot()

    # 6. Model
    model = Net(input_dim=input_dim, num_classes=num_classes).to(device)

    # 7. Start Flower client — Gemini passed in but never touches training loop
    try:
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=IDSClient(
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                class_names=class_names,
                copilot=copilot,
            ).to_client(),
        )
    except Exception as e:
        print(
            f"\n❌ Could not connect to server at 127.0.0.1:8080.\n"
            f"   Make sure simulation.py is running first.\n"
            f"   Error: {e}"
        )


if __name__ == "__main__":
    main()
