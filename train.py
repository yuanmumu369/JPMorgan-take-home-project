import argparse, os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from joblib import dump

from src.data_utils import load_training_data, split_train_valid, build_preprocessor, fit_preprocessor_save
from src.model import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

def train_model(X_train_t, y_train_t, X_valid_t, y_valid_t, hidden_dim=64, lr=1e-3, batch_size=256, epochs=25, weight_decay=0.0):
    model = MLP(in_features=X_train_t.shape[1], hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid_t, y_valid_t), batch_size=batch_size)

    best_auc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(yb.numpy())
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Epoch {epoch}: loss={epoch_loss:.4f} | AUC={auc:.4f} | ACC={acc:.4f} | F1={f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc

def main(args):
    # Load data
    df = load_training_data(args.train_csv)
    # Split
    X_train, X_valid, y_train, y_valid = split_train_valid(df)
    # Preprocess
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    os.makedirs(args.artifacts_dir, exist_ok=True)
    preproc_path = os.path.join(args.artifacts_dir, "preprocessor.joblib")
    fit_preprocessor_save(preprocessor, X_train, preproc_path)

    # Transform to numpy
    X_train_np = preprocessor.transform(X_train)
    X_valid_np = preprocessor.transform(X_valid)
    y_train_np = y_train.values.astype(np.float32)
    y_valid_np = y_valid.values.astype(np.float32)

    # To tensors
    X_train_t = to_tensor(X_train_np)
    X_valid_t = to_tensor(X_valid_np)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid_np, dtype=torch.float32)

    # Train
    model, best_auc = train_model(
        X_train_t, y_train_t, X_valid_t, y_valid_t,
        hidden_dim=args.hidden_dim, lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, weight_decay=args.weight_decay
    )

    # Save
    model_path = os.path.join(args.artifacts_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save simple metrics
    with open(os.path.join(args.artifacts_dir, "metrics.txt"), "w") as f:
        f.write(f"best_valid_auc: {best_auc:.6f}\n")

    print(f"Saved preprocessor to {preproc_path}")
    print(f"Saved model to {model_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/mnt/data/training_loan_data.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()
    main(args)