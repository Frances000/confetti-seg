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
from typing import Tuple, List, Dict
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Optional xgboost (kept light + fast)
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

# Optional memory logging
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    import resource

# ---------------------------
# Helpers: file parsing
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
# NEW: label mapping utilities
# ---------------------------

def build_or_load_classes(outdir: Path, all_labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build a canonical alphabetical class order from all_labels (union of every split),
    or load an existing one from outdir/classes.json. Fail if unseen labels appear
    against an existing mapping (protects reproducibility).
    """
    classes_path = outdir / "classes.json"
    if classes_path.exists():
        with open(classes_path, "r") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
            raise ValueError(f"{classes_path} exists but is not a list[str].")
        # Any labels outside the frozen set?
        extra = sorted(set(all_labels) - set(classes))
        if extra:
            raise ValueError(
                "Unseen class labels encountered that are not present in the frozen mapping.\n"
                f"Unseen: {extra}\nExisting classes.json: {classes_path}\n"
                "Either update classes.json intentionally or filter the offending rows."
            )
        print(f"Using existing class mapping ({len(classes)} classes) from {classes_path}")
    else:
        classes = sorted(set(all_labels))
        with open(classes_path, "w") as f:
            json.dump(classes, f, indent=2)
        print(f"Saved canonical class order to {classes_path} ({len(classes)} classes).")
    cls2id = {c: i for i, c in enumerate(classes)}
    return classes, cls2id

def encode_labels(y_str: pd.Series, cls2id: Dict[str, int]) -> np.ndarray:
    """Map string labels to contiguous ints 0..K-1 based on cls2id. Error on unseen."""
    try:
        return np.asarray([cls2id[s] for s in y_str.astype(str)], dtype=int)
    except KeyError as e:
        raise ValueError(f"Unseen label {e.args[0]} not in classes.json") from None

def decode_labels(y_int: np.ndarray, classes: List[str]) -> np.ndarray:
    """Map 0..K-1 back to string labels using classes list."""
    y_int = np.asarray(y_int, dtype=int)
    if y_int.min() < 0 or y_int.max() >= len(classes):
        raise ValueError("Predicted class index out of range for classes.json")
    return np.asarray([classes[i] for i in y_int], dtype=object)

# ---------------------------
# Training
# ---------------------------

def make_models(random_state: int, xgb_tree_method: str | None):
    models = {
        "rf": RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "dt": DecisionTreeClassifier(random_state=random_state),
        "svm": SVC(kernel="rbf", gamma="scale", C=1.0, random_state=random_state),
        "zeror": DummyClassifier(strategy="most_frequent"),
    }
    if _HAVE_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=60,          # fast
            max_depth=3,              # shallow, generalises OK
            learning_rate=0.25,       # fewer rounds
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            tree_method=xgb_tree_method or "hist",  # CPU hist by default
            objective="multi:softmax",              # returns class ids (0..K-1)
            # num_class is set at train time because we need 'classes' length
        )
    else:
        print("NOTE: xgboost not installed; skipping XGB model.")
    return models

def train_and_save_all(
    X: pd.DataFrame,
    y: pd.Series,
    outdir: Path,
    tag: str,
    xgb_tree_method: str | None,
    classes: List[str],
    cls2id: Dict[str, int],
    random_state: int = 42,
):
    print(f"\n=== Training set '{tag}' ===")
    print(f"Raw shape: X={X.shape}, classes={sorted(y.unique())}")

    # Stratified 80/20 split in the ORIGINAL string label space for consistency
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = make_models(random_state=random_state, xgb_tree_method=xgb_tree_method)

    for name, clf in models.items():
        print(f"\n--- Training {name.upper()} ---")
        mem_before = mem_usage_mb()
        t0 = time.time()

        if name == "xgb":
            # Encode y -> ints for XGBoost
            ytr_enc = encode_labels(ytr, cls2id)
            # Ensure K matches our canonical class count
            clf.set_params(num_class=len(classes))
            clf.fit(Xtr, ytr_enc)
            # Predict -> ints -> decode back to strings for reporting
            yhat_int = clf.predict(Xte)
            # xgboost may return float dtype; cast to int safely
            yhat = decode_labels(np.asarray(yhat_int, dtype=int), classes)
        else:
            # scikit-learn models accept string labels directly
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)

        train_time = time.time() - t0
        mem_after = mem_usage_mb()

        print(f"{name.upper()} trained in {train_time:.2f}s "
              f"(memory Δ {mem_after - mem_before:.1f} MB, current {mem_after:.1f} MB)")
        print(classification_report(yte, yhat, digits=4))

        # Save model; classes.json already sits in outdir for downstream decoding
        joblib.dump(clf, outdir / f"model_{tag}_{name}.joblib")
        print(f"Saved: model_{tag}_{name}.joblib")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train multiple Confetti models on BASE CSVs only (no distance features)."
    )
    ap.add_argument("--roots", nargs="+", required=True,
                    help="One or more date folders (…/Cd04M-…-merge-BF etc.)")
    ap.add_argument("--outdir", required=True, help="Output directory for models")
    ap.add_argument("--xgb-gpu", action="store_true",
                    help="Use XGBoost gpu_hist (defaults to CPU 'hist').")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.roots]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    xgb_tree_method = "gpu_hist" if args.xgb_gpu else "hist"

    print("Scanning for CSVs…")
    all_csvs = list_csvs(roots)

    # Partition into base vs base_patch (ignore any distance_*)
    base_patch_prefixes = ("patch_", "base_patch_")
    base_patch_files = [
        fp for fp in all_csvs
        if fp.name.startswith(base_patch_prefixes)
    ]
    base_plain_files = [
        fp for fp in all_csvs
        if not fp.name.startswith(("distance_",) + base_patch_prefixes)
    ]

    print(f"Found CSVs total: {len(all_csvs)}")
    print(f"  - base (plain):     {len(base_plain_files)}")
    print(f"  - base_patch:       {len(base_patch_files)}")

    if len(base_plain_files) == 0 and len(base_patch_files) == 0:
        print("ERROR: No eligible BASE CSVs found (need plain base_* or base_patch_* / patch_*).",
              file=sys.stderr)
        sys.exit(2)

    # -------------------------------
    # Load datasets BEFORE training so we can:
    # (1) build a canonical class list from the union of labels,
    # (2) persist it in classes.json for reproducibility.
    # -------------------------------
    Xb = yb = Xp = yp = None

    if base_plain_files:
        print("\nLoading BASE (plain) tables…")
        Xb, yb, _ = concat_tables(base_plain_files)

    if base_patch_files:
        print("\nLoading BASE_PATCH tables…")
        Xp, yp, _ = concat_tables(base_patch_files)

    # Union of labels across whichever splits exist
    label_series = []
    if yb is not None: label_series.append(yb.astype(str))
    if yp is not None: label_series.append(yp.astype(str))
    all_labels = pd.concat(label_series, ignore_index=True).tolist()

    # Build or load canonical alphabetical classes and mapping
    classes, cls2id = build_or_load_classes(outdir, all_labels)

    # -------------------------------
    # Train on each split with the frozen mapping
    # -------------------------------
    if Xb is not None:
        train_and_save_all(
            Xb, yb, outdir, tag="base",
            xgb_tree_method=xgb_tree_method,
            classes=classes, cls2id=cls2id
        )

    if Xp is not None:
        train_and_save_all(
            Xp, yp, outdir, tag="base_patch",
            xgb_tree_method=xgb_tree_method,
            classes=classes, cls2id=cls2id
        )

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
