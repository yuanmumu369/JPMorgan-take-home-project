import argparse, os
import numpy as np
import pandas as pd
import torch
from src.data_utils import load_test_data, load_preprocessor
from src.model import MLP

def main(args):
    test_df = load_test_data(args.test_csv)
    preprocessor = load_preprocessor(os.path.join(args.artifacts_dir, "preprocessor.joblib"))
    X_np = preprocessor.transform(test_df)
    X = torch.tensor(X_np, dtype=torch.float32)

    model = MLP(in_features=X.shape[1], hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(os.path.join(args.artifacts_dir, "model.pt"), map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({
        "id": test_df.get("id", pd.Series(range(len(test_df)))),
        "bad_flag": preds.ravel().astype(int)
    })
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "test_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="/mnt/data/testing_loan_data.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()
    main(args)