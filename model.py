import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network — MLP for tabular IDS data
# ─────────────────────────────────────────────────────────────────────────────

class Net(nn.Module):
    """
    Three-layer MLP for the CICIDS tabular feature space.

    Architecture:   input → 256 → 128 → num_classes
    Regularisation: BatchNorm1d + Dropout(0.2) after each hidden layer.
    Init:           Xavier Uniform on Linear weights, zeros on biases.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super(Net, self).__init__()

        self.fc1     = nn.Linear(input_dim, 256)
        self.bn1     = nn.BatchNorm1d(256)
        self.fc2     = nn.Linear(256, 128)
        self.bn2     = nn.BatchNorm1d(128)
        self.fc3     = nn.Linear(128, num_classes)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier Uniform to all Linear layers, zeros to biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> None:
    """
    Supervised training loop with gradient clipping.

    max_norm=1.0 must stay in sync with DP sensitivity in simulation.py.
    Changing one without the other breaks the formal privacy guarantee.
    """
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in trainloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss   = criterion(output, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"  Epoch [{epoch + 1}/{epochs}] loss: {avg_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper — with per-class F1 metrics
# ─────────────────────────────────────────────────────────────────────────────

def test(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = None,
    verbose: bool = True,
):
    """
    Evaluates the model on testloader.

    FIX: Now reports per-class precision, recall, and F1 score via
    sklearn's classification_report. Overall accuracy alone was hiding
    poor performance on minority attack classes (e.g. BrowserHijacking,
    CommandInjection) — the most dangerous classes to miss in a real IDS.

    Args:
        model:       trained PyTorch model
        testloader:  DataLoader for test set
        device:      torch device
        class_names: list of class name strings for readable F1 report.
                     Pass encoder.classes_ from data_setup for best results.
        verbose:     if True, prints the full per-class F1 report.

    Returns:
        (avg_loss, accuracy) — both plain Python floats.
    """
    criterion   = nn.CrossEntropyLoss()
    all_preds   = []
    all_targets = []
    total_loss  = 0.0

    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs      = model(data)
            loss         = criterion(outputs, target)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    avg_loss = float(total_loss / len(testloader))
    accuracy = float(np.mean(all_preds == all_targets))

    if verbose:
        print(f"\n  Overall Accuracy: {accuracy:.4f} | Avg Loss: {avg_loss:.4f}")
        print("\n  Per-Class Report:")
        print(classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            zero_division=0,
            digits=4,
        ))

    return avg_loss, accuracy
