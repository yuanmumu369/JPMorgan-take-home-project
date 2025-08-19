# Loan Default Prediction (OA)

Beginner-friendly scaffold to complete the EDA + PyTorch tasks.

## Project Layout
```
loan_default_oa/
  ├─ notebooks/
  │   └─ EDA_and_Model_Starter.ipynb
  ├─ src/
  │   ├─ data_utils.py
  │   └─ model.py
  ├─ artifacts/            # saved preprocessor, model, predictions (gitignored)
  ├─ train.py              # train neural network
  ├─ infer.py              # run inference on test set
  ├─ requirements.txt
  ├─ environment.yml
  └─ .gitignore
```

## 1) Environment
Option A: pip
```
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

Option B: conda (CPU)
```
conda env create -f environment.yml
conda activate loan-oa
```

## 2) (Recommended) Explore data
Open the starter notebook at `notebooks/EDA_and_Model_Starter.ipynb` and run top to bottom.
Add your own plots & notes.

## 3) Train the PyTorch model
```
python train.py --train_csv /mnt/data/training_loan_data.csv --artifacts_dir artifacts --epochs 25 --hidden_dim 64
```
Artifacts:
- `artifacts/preprocessor.joblib`
- `artifacts/model.pt`
- `artifacts/metrics.txt`

## 4) Inference on the test set
```
python infer.py --test_csv /mnt/data/testing_loan_data.csv --artifacts_dir artifacts
```
This writes: `artifacts/test_predictions.csv` (columns: `id`, `bad_flag`).

## 5) What to commit
Commit code, notebooks, and README **but not datasets** (they are gitignored). Upload `artifacts/test_predictions.csv` separately to the recruiter as requested.

## Notes
- Target column: `bad_flag` (binary). See the data dictionary for feature meanings.
- Keep commits small: EDA data checks → cleaning → plots → insights; model class → training loop → metrics → inference.