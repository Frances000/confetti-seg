#!/usr/bin/env python3
# eval_from_csv_single_model.py
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning, FitFailedWarning

# Quiet noisy warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
np.seterr(all="ignore")

def load_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = None
    if "class" in df.columns:
        y = df["class"].astype(str).to_numpy()
        X = df.drop(columns=["class"])
    else:
        X = df
    return X, y

def align_columns_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensure X has the exact columns the model was trained on.
    If model.feature_names_in_ exists, use it. Otherwise, keep current X as-is.
    Missing columns are added and filled with zeros; extra columns are dropped.
    """
    if hasattr(model, "feature_names_in_"):
        want = list(model.feature_names_in_)
        for c in want:
            if c not in X.columns:
                X[c] = 0
        X = X[want]
    return X

def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained model on a single CSV stack.")
    ap.add_argument("--model_path", required=True, help="Path to .joblib model (e.g., model_base_rf.joblib)")
    ap.add_argument("--csv_path",   required=True, help="Path to CSV with columns (incl. optional 'class')")
    ap.add_argument("--out",        default=None,  help="TXT report path (default: <csv_stem>_eval.txt next to CSV)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    csv_path   = Path(args.csv_path)
    out_path   = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_eval.txt")

    # load
    model = joblib.load(model_path)
    X, y = load_csv(csv_path)

    # align features
    X_in = align_columns_to_model(X.copy(), model)

    # predict
    y_hat = model.predict(X_in)
    n = min(len(y_hat), len(y_hat) if y is None else len(y))
    y_hat = y_hat[:n]
    if y is not None:
        y_eval = y[:n]
        acc = float((y_hat == y_eval).mean())
        labels_sorted = sorted(pd.unique(np.concatenate([y_eval, y_hat]).astype(str)))
        cm = confusion_matrix(y_eval, y_hat, labels=labels_sorted)
        cr = classification_report(y_eval, y_hat, zero_division=0)
    else:
        acc = None
        labels_sorted = sorted(pd.unique(y_hat.astype(str)))
        cm = None
        cr = None

    # write report
    with open(out_path, "w", encoding="utf-8") as f:
        print(f"Model: {model_path}", file=f)
        print(f"CSV:   {csv_path}", file=f)
        print(f"Rows:  {len(X_in)}", file=f)
        if hasattr(model, 'feature_names_in_'):
            print(f"Features used: {len(model.feature_names_in_)}", file=f)
        else:
            print(f"Features used: {X_in.shape[1]}", file=f)

        if y is not None:
            print("\n=== Evaluation (with ground truth) ===", file=f)
            print(f"Pixels: {n}", file=f)
            print(f"Accuracy: {acc:.6f}", file=f)
            print(f"Labels: {labels_sorted}", file=f)
            print("Confusion matrix:", file=f)
            print(cm, file=f)
            print("\nClassification report:", file=f)
            print(cr, file=f)
        else:
            # no GT
            print("\n=== Predictions (no ground truth found) ===", file=f)
            vc = pd.Series(y_hat).value_counts().sort_index()
            print("Class counts:", file=f)
            for cls, cnt in vc.items():
                print(f"  {cls}: {cnt}", file=f)

    print(f"Report written to: {out_path}")

if __name__ == "__main__":
    main()
