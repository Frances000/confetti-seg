#!/usr/bin/env python3
"""
Train multiple models for Confetti segmentation features using BASE CSVs only (NO feature selection):
  - Random Forest
  - Decision Tree
  - SVM (RBF kernel)
  - XGBoost
  - ZeroR (majority class baseline)

Base CSVs are filenames WITHOUT the 'distance_' prefix.
"""

from __future__ import annotations
import argparse, sys, time, os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier

# Optional memory logging
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    import resource

# ---------------------------
# Helpers
# ---------------------------

LABEL_FROM_DIR = re.compile(r"^(filiform|foliate_L|foliate_R)", re.IGNORECASE)

def parse_label_from_dir(dirpath: Path) -> str:
    """Infer class label from parent directory name like 'filiform_001_csv'."""
    m = LABEL_FROM_DIR.match(dirpath.name)
    if not m:
        raise ValueError(f"Cannot infer label from directory name: {dirpath}")
    return m.group(1)

def list_csvs(roots: List[Path]) -> List[Path]:
    """Find all CSVs under any '*_csv' subfolder of each root."""
    csvs: List[Path] = []
    for root in roots:
        for sub in root.glob("*_csv"):
            if sub.is_dir():
                csvs.extend(sorted(sub.glob("*.csv")))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs found. Checked roots: {', '.join(str(r) for r in roots)}"
        )
    return csvs

def load_table(fp: Path, class_column: str = "class") -> Tuple[pd.DataFrame, pd.Series]:
    """Load one CSV and return numeric features + class labels."""
    df = pd.read_csv(fp)
    if class_column in df.columns:
        y = df[class_column]
        X = df.drop(columns=[class_column])
    else:
        y = pd.Series(parse_label_from_dir(fp.parent), index=df.index, name="class")
        X = df
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError(f"No numeric features in {fp}")
    return X, y

def concat_tables(file_list: List[Path]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and concatenate many CSVs."""
    Xs, ys = [], []
    for i, fp in enumerate(file_list, 1):
        if i % 25 == 0 or i == len(file_list):
            print(f"  - Loading {i}/{len(file_list)}: {fp.name}")
        Xi, yi = load_table(fp)
        Xs.append(Xi)
        ys.append(yi)
    X = pd.concat(Xs, axis=0, ignore_index=True)
    y = pd.concat(ys, axis=0, ignore_index=True)
    return X, y, list(X.columns)

def mem_usage_mb() -> float:
    """Return current memory usage in MB."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux, bytes on macOS
        return usage / 1024 if sys.platform != "darwin" else usage / (1024 * 1024)

# ---------------------------
# Training
# ---------------------------

def train_and_save_all(
    X: pd.DataFrame,
    y: pd.Series,
    outdir: Path,
    tag: str = "base",
    random_state: int = 42,
):
    print(f"\n=== Training set '{tag}' (BASE only) ===")
    print(f"Raw shape: X={X.shape}, classes={sorted(y.unique())}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = {
        "rf": RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample"
        ),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "svm": SVC(kernel="rbf", gamma="scale", C=1.0, random_state=random_state),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist"
        ),
        "zeror": DummyClassifier(strategy="most_frequent")
    }

    for name, clf in models.items():
        print(f"\n--- Training {name.upper()} ---")
        mem_before = mem_usage_mb()
        t0 = time.time()
        clf.fit(Xtr, ytr)
        train_time = time.time() - t0
        mem_after = mem_usage_mb()

        yhat = clf.predict(Xte)
        print(f"{name.upper()} trained in {train_time:.2f}s "
              f"(memory Δ {mem_after - mem_before:.1f} MB, peak {mem_after:.1f} MB)")
        print(classification_report(yte, yhat, digits=4))
        joblib.dump(clf, outdir / f"model_{tag}_{name}.joblib")
        print(f"Saved: model_{tag}_{name}.joblib")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train multiple Confetti models on BASE CSVs only (no feature selection).")
    ap.add_argument("--roots", nargs="+", required=True, help="One or more date folders (…/Cd04M-…-merge-BF etc.)")
    ap.add_argument("--outdir", required=True, help="Output directory for models")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.roots]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Scanning for CSVs…")
    all_csvs = list_csvs(roots)

    # Keep only BASE CSVs (exclude any that start with 'distance_')
    base_files = [fp for fp in all_csvs if not fp.name.startswith("distance_")]

    print(f"Found BASE CSVs: {len(base_files)}")
    if len(base_files) == 0:
        print("ERROR: No base CSVs found (filenames without 'distance_' prefix).", file=sys.stderr)
        sys.exit(2)

    print("\nLoading base tables…")
    Xb, yb, _ = concat_tables(base_files)
    train_and_save_all(Xb, yb, outdir, tag="base")

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
