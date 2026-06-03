import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import glob
import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "data2023/*.csv"
OUTPUT_FILE    = "processed_data.csv"
ENCODER_FILE   = "label_encoder.pkl"   # shared encoder — all files use this
SCALER_FILE    = "scaler.pkl"          # shared scaler — fit on train only
BATCH_SIZE     = 32
RANDOM_STATE   = 42
NUM_CLIENTS    = 3                     # total federated clients

SAMPLE_CONFIG = {
    "Benign": 0.03,    # raised from 0.005 — was causing extreme class imbalance
    "default": 0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data():
    """
    Reads all CSVs, samples per class, cleans infinities/NaNs, and saves
    the raw (unscaled) processed dataset.

    FIX: Scaler is no longer fit here on the full dataset — that caused data
    leakage (test set statistics were visible to the scaler). Scaling is now
    done inside load_data() after the train/test split, fit on train only.
    """
    all_files = glob.glob(DATA_PATH)
    if not all_files:
        print(f"Error: No CSV files found matching '{DATA_PATH}'")
        return

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        label_name = os.path.basename(filename).split(".")[0]
        df["label"] = label_name

        sample_rate = SAMPLE_CONFIG.get(label_name, SAMPLE_CONFIG["default"])

        if len(df) < 2:
            sampled_df = df
        else:
            n_samples  = max(1, int(len(df) * sample_rate))
            sampled_df = df.sample(n=n_samples, random_state=RANDOM_STATE)

        li.append(sampled_df)
        print(f" - {label_name}: {len(sampled_df)} rows (rate={sample_rate})")

    full_df = pd.concat(li, axis=0, ignore_index=True)

    # Clean infinities and NaNs
    X = full_df.drop("label", axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.astype(np.float64)
    X = X.fillna(X.median())
    y = full_df["label"]

    # Save raw (unscaled) — scaler will be fit on train split inside load_data()
    processed_df = pd.DataFrame(X.values, columns=X.columns)
    processed_df["label"] = y.values
    processed_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved {len(processed_df)} rows → '{OUTPUT_FILE}'")
    print(f"   Class distribution:\n{processed_df['label'].value_counts()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Standard Data Loading (used by all clients when non-IID is not needed)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(batch_size=BATCH_SIZE, use_smote=True):
    """
    Loads processed_data.csv, encodes labels using a shared LabelEncoder
    (saved to disk so every file uses identical class-integer mappings),
    scales features using MinMaxScaler fit on TRAIN only (no leakage),
    and returns PyTorch DataLoaders.

    FIX: LabelEncoder is now saved to label_encoder.pkl so simulation.py
    and malicious_client.py load the exact same mapping — previously each
    file encoded independently which could produce different integer classes.

    FIX: MinMaxScaler is now fit on X_train only, then applied to X_test.
    Previously it was fit on the full dataset before splitting (data leakage).

    Args:
        batch_size: DataLoader batch size.
        use_smote:  Apply SMOTE to training split for minority class balance.
                    Requires: pip install imbalanced-learn

    Returns:
        trainloader, testloader, input_dim, num_classes
    """
    if not os.path.exists(OUTPUT_FILE):
        print(f"'{OUTPUT_FILE}' not found — running prepare_data() first...")
        prepare_data()

    df    = pd.read_csv(OUTPUT_FILE)
    X     = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # FIX: shared encoder — saved to disk so all files use identical mapping
    encoder = LabelEncoder()
    y       = encoder.fit_transform(y_raw).astype(np.int64)
    joblib.dump(encoder, ENCODER_FILE)

    num_classes = len(encoder.classes_)
    input_dim   = X.shape[1]

    print(f"Dataset: {len(X)} samples | {input_dim} features | {num_classes} classes")
    print(f"Classes: {list(encoder.classes_)}\n")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # FIX: scaler fit on train only — prevents test set leaking into scaling
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_FILE)

    # Optional SMOTE — on train only, never on test
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"SMOTE applied → training set now {len(X_train)} samples")
        except ImportError:
            print("Warning: imbalanced-learn not installed. pip install imbalanced-learn")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, input_dim, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# 3. Non-IID Data Loading (simulates real federated setting per client)
# ─────────────────────────────────────────────────────────────────────────────

def load_data_noniid(client_id, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE,
                     alpha=0.5, use_smote=True):
    """
    Partitions the dataset using a Dirichlet distribution to simulate
    non-IID data across clients — each client sees a different class mixture.

    This is critical for genuine FL evaluation. Without it, all clients share
    the same data distribution which is just distributed training, not FL.

    Args:
        client_id:   integer index of this client (0 to num_clients-1)
        num_clients: total number of federated clients
        batch_size:  DataLoader batch size
        alpha:       Dirichlet concentration parameter.
                     Lower = more heterogeneous (non-IID).
                     alpha=0.5 is the standard FL benchmark setting.
        use_smote:   Apply SMOTE to this client's training split.

    Returns:
        trainloader, testloader, input_dim, num_classes
    """
    if not os.path.exists(OUTPUT_FILE):
        prepare_data()

    df    = pd.read_csv(OUTPUT_FILE)
    X     = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # Load shared encoder so class mappings are identical across all clients
    if os.path.exists(ENCODER_FILE):
        encoder = joblib.load(ENCODER_FILE)
        y       = encoder.transform(y_raw).astype(np.int64)
    else:
        encoder = LabelEncoder()
        y       = encoder.fit_transform(y_raw).astype(np.int64)
        joblib.dump(encoder, ENCODER_FILE)

    num_classes = len(encoder.classes_)
    input_dim   = X.shape[1]

    # Dirichlet partition — each class is split across clients non-uniformly
    np.random.seed(RANDOM_STATE)
    class_indices  = [np.where(y == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c_indices in class_indices:
        np.random.shuffle(c_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        splits      = (proportions * len(c_indices)).astype(int)
        splits[-1]  = len(c_indices) - splits[:-1].sum()  # fix rounding error
        start = 0
        for i, size in enumerate(splits):
            client_indices[i].extend(c_indices[start: start + size].tolist())
            start += size

    idx      = client_indices[client_id]
    X_client = X[idx]
    y_client = y[idx]

    print(f"[Client {client_id}] Non-IID partition: {len(X_client)} samples")
    unique, counts = np.unique(y_client, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  class {encoder.classes_[cls]}: {cnt} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_client
    )

    # Scaler fit on this client's train split only
    if os.path.exists(SCALER_FILE):
        scaler  = joblib.load(SCALER_FILE)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
    else:
        scaler  = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(3, min(counts) - 1))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"  SMOTE applied → {len(X_train)} training samples")
        except (ImportError, ValueError) as e:
            print(f"  SMOTE skipped: {e}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, input_dim, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare_data()
