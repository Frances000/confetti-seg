#!/usr/bin/env python3
"""
Train multiple models for Confetti segmentation features (NO feature selection):
  - Random Forest
  - Decision Tree
  - SVM (RBF kernel)
  - XGBoost
  - ZeroR (majority class baseline)

Distance-mapped CSVs are detected by 'distance_' prefix.
Base CSVs are filenames without the prefix.
"""
from __future__ import annotations
import argparse, sys, time, os
from pathlib import Path
import re, json
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

# ---------------------------
# Helpers
# ---------------------------

LABEL_FROM_DIR = re.compile(r"^(filiform|foliate_L|foliate_R)", re.IGNORECASE)

def parse_label_from_dir(dirpath: Path) -> str:
    m = LABEL_FROM_DIR.match(dirpath.name)
    if not m:
        raise ValueError(f"Cannot infer label from directory name: {dirpath}")
    return m.group(1)

def list_csvs(roots: List[Path]) -> List[Path]:
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
    Xs, ys = [], []
    for i, fp in enumerate(file_list, 1):
        if i % 25 == 0 or i == len(file_list):
            print(f"  - Loading {i}/{len(file_list)}: {fp.name}")
        Xi, yi = load_table(fp)
        Xs.append(Xi); ys.append(yi)
    X = pd.concat(Xs, axis=0, ignore_index=True)
    y = pd.concat(ys, axis=0, ignore_index=True)
    return X, y, list(X.columns)


def build_or_load_classes(outdir: Path, all_labels: List[str], freeze: bool=True) -> tuple[list[str], dict[str,int]]:
    """
    Persist a canonical alphabetical class order at outdirjson
    If it exists and freeze=True, error on unseen labels to protect reproducibility.
    """
    classes_path = outdir / "classes.json"
    if classes_path.exists():
        with open(classes_path, "r") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
            raise ValueError(f"{classes_path} exists but is not a list[str].")
        if freeze:
            extra = sorted(set(all_labels) - set(classes))
            if extra:
                raise ValueError(
                    "Unseen class labels encountered that are not present in the frozen mapping.\n"
                    f"Unseen: {extra}\nExisting mapping: {classes_path}\n"
                    "Either rebuild classes.json intentionally or filter these rows."
                )
        print(f"Using existing class mapping ({len(classes)} classes): {classes_path}")
    else:
        classes = sorted(set(all_labels))
        with open(classes_path, "w") as f:
            json.dump(classes, f, indent=2)
        print(f"Saved canonical class order to {classes_path} ({len(classes)} classes).")
    cls2id = {c: i for i, c in enumerate(classes)}
    return classes, cls2id

def encode_labels(y_str: pd.Series, cls2id: Dict[str,int]) -> np.ndarray:
    try:
        return np.asarray([cls2id[s] for s in y_str.astype(str)], dtype=int)
    except KeyError as e:
        raise ValueError(f"Unseen label {e.args[0]} not in classes.json") from None

def decode_labels(y_int: np.ndarray, classes: List[str]) -> np.ndarray:
    y_int = np.asarray(y_int, dtype=int)
    if y_int.min() < 0 or y_int.max() >= len(classes):
        raise ValueError("Predicted class index out of range for classes.json")
    return np.asarray([classes[i] for i in y_int], dtype=object)

# ---------------------------
# Training
# ---------------------------

def make_models(random_state: int, xgb_tree_method: str | None):
    models = {
        # "rf": RandomForestClassifier(
        #     n_estimators=400, n_jobs=-1, random_state=random_state,
        #     class_weight="balanced_subsample"
        # ),
        # "dt": DecisionTreeClassifier(random_state=random_state),
        # "svm": SVC(kernel="rbf", gamma="scale", C=1.0, random_state=random_state),
        # "zeror": DummyClassifier(strategy="most_frequent"),
    }

    if _HAVE_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.25,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method=xgb_tree_method or "hist",   # "hist" (CPU) by default
            objective="multi:softmax",               # returns class ids (0..K-1)
            # num_class is set at train-time once classes.json is known
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
    cls2id: Dict[str,int],
    random_state: int = 42,
):
    print(f"\n=== Training set '{tag}' ===")
    print(f"Raw shape: X={X.shape}, classes={sorted(y.unique())}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = make_models(random_state=random_state, xgb_tree_method=xgb_tree_method)

    for name, clf in models.items():
        print(f"\n--- Training {name.upper()} ---")
        t0 = time.time()

        if name == "xgb":
            ytr_enc = encode_labels(ytr, cls2id)
            clf.set_params(num_class=len(classes))
            clf.fit(Xtr, ytr_enc)
            yhat_int = clf.predict(Xte)
            yhat = decode_labels(np.asarray(yhat_int, dtype=int), classes)
        else:
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)

        train_time = time.time() - t0
        print(f"{name.upper()} trained in {train_time:.2f}s")
        print(classification_report(yte, yhat, digits=4))
        joblib.dump(clf, outdir / f"model_{tag}_{name}.joblib")
        print(f"Saved: model_{tag}_{name}.joblib")

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train multiple Confetti models without feature selection.")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="One or more date folders (…/Cd04M-…-merge-BF etc.)")
    ap.add_argument("--outdir", required=True, help="Output directory for models")
    # --- NEW: simple switch for GPU ---
    ap.add_argument("--xgb-gpu", action="store_true",
                    help="Use XGBoost gpu_hist (defaults to fast CPU 'hist').")
    ap.add_argument("--freeze-classes", action="store_true", default=True,
                    help="If classes.json exists, error on unseen labels (default: on).")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.roots]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    xgb_tree_method = "gpu_hist" if args.xgb_gpu else "hist"

    print("Scanning for CSVs…")
    all_csvs = list_csvs(roots)

    distance_files = [fp for fp in all_csvs if fp.name.startswith("distance_")]
    base_files      = [fp for fp in all_csvs if not fp.name.startswith("distance_")]

    print(f"Found total CSVs: {len(all_csvs)}")
    print(f"  - distance-mapped: {len(distance_files)}")
    print(f"  - base:            {len(base_files)}")

    # Load datasets first so we can build/load ONE canonical mapping
    Xd = yd = Xb = yb = None
    if distance_files:
        print("\nLoading distance-mapped tables…")
        Xd, yd, _ = concat_tables(distance_files)
    if base_files:
        print("\nLoading base tables…")
        Xb, yb, _ = concat_tables(base_files)

    if Xd is None and Xb is None:
        print("ERROR: No eligible CSVs found.", file=sys.stderr)
        sys.exit(2)

    # Build or load alphabetical class order from the union of labels present this run
    label_series = []
    if yd is not None: label_series.append(yd.astype(str))
    if yb is not None: label_series.append(yb.astype(str))
    all_labels = pd.concat(label_series, ignore_index=True).tolist()

    classes, cls2id = build_or_load_classes(outdir, all_labels, freeze=args.freeze_classes)

    # Train each split with the frozen mapping
    if Xd is not None:
        train_and_save_all(Xd, yd, outdir, tag="distance",
                           xgb_tree_method=xgb_tree_method,
                           classes=classes, cls2id=cls2id)

    if Xb is not None:
        train_and_save_all(Xb, yb, outdir, tag="base",
                           xgb_tree_method=xgb_tree_method,
                           classes=classes, cls2id=cls2id)

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
