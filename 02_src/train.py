#!/usr/bin/env python3
"""
Train two models for Confetti segmentation features:
  1) distance-mapped CSVs (filenames starting with 'distance_')
  2) base CSVs (same folders, filenames without the 'distance_' prefix)

- Recurses date folders like:
    /.../Cd04M-155706-20170601-merge-BF/
      ├─ filiform_001_csv/
      │    ├─ distance_filiform_patch_00.csv
      │    └─ filiform_patch_00.csv
      ├─ foliate_L_002_csv/
      └─ foliate_R_003_csv/

Outputs:
  outdir/
    model_base.joblib
    selector_base.joblib
    model_distance.joblib
    selector_distance.joblib
    feature_names_*.txt
"""

from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


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
    """
    Load one CSV. If a 'class' column exists, use it.
    Otherwise, derive the class from the '*_csv' parent folder name.
    Drops any non-numeric feature columns automatically (except class).
    """
    df = pd.read_csv(fp)
    # Try to locate class column
    if class_column in df.columns:
        y = df[class_column]
        X = df.drop(columns=[class_column])
    else:
        # derive label from parent of the file's folder
        y = pd.Series(parse_label_from_dir(fp.parent), index=df.index, name="class")
        X = df
    # keep numeric features only
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError(f"No numeric features in {fp}")
    return X, y

def concat_tables(file_list: List[Path]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and concatenate many CSVs; return X, y, feature_names."""
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

def build_selector(k_features: int) -> Pipeline:
    """
    Feature selection pipeline:
      1) Remove constant features (VarianceThreshold)
      2) SelectKBest(f_classif) to top k
    """
    return Pipeline(steps=[
        ("var", VarianceThreshold(threshold=0.0)),
        ("kbest", SelectKBest(score_func=f_classif, k=k_features)),
    ])

def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    outdir: Path,
    tag: str,
    k_features: int,
    n_estimators: int = 400,
    random_state: int = 42,
):
    print(f"\n=== Training '{tag}' ===")
    print(f"Raw shape: X={X.shape}, classes={sorted(y.unique())}")

    # Split for a quick sanity report
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    selector = build_selector(k_features=k_features)
    t0 = time.time()
    Xtr_sel = selector.fit_transform(Xtr, ytr)
    Xte_sel = selector.transform(Xte)
    sel_time = time.time() - t0
    kept_mask = selector.named_steps["kbest"].get_support()
    # After variance filter, feature names shrink; reconstruct kept names:
    # Grab names after the variance step, then mask with kbest
    var_step = selector.named_steps["var"]
    kept_after_var = Xtr.columns[var_step.get_support(indices=True)]
    kept_names = list(kept_after_var[kept_mask])

    print(f"Feature selection: kept {Xtr_sel.shape[1]} features in {sel_time:.2f}s")
    (outdir / f"feature_names_{tag}.txt").write_text("\n".join(kept_names))

    # Train RF
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
        max_depth=None,
    )
    t1 = time.time()
    clf.fit(Xtr_sel, ytr)
    train_time = time.time() - t1
    print(f"Model: RandomForest(n_estimators={n_estimators}) trained in {train_time:.2f}s")

    # Quick validation report
    yhat = clf.predict(Xte_sel)
    print("\nValidation (hold-out 20%)")
    print(classification_report(yte, yhat, digits=4))

    # Persist
    joblib.dump(clf, outdir / f"model_{tag}.joblib")
    joblib.dump(selector, outdir / f"selector_{tag}.joblib")
    print(f"Saved: model_{tag}.joblib, selector_{tag}.joblib, feature_names_{tag}.txt\n")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train base vs distance Confetti models.")
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more date folders (…/Cd04M-…-merge-BF etc.)",
    )
    ap.add_argument(
        "--outdir", required=True, help="Output directory for models/selectors"
    )
    ap.add_argument(
        "--k_features", type=int, default=50, help="Number of features to keep (≤k)"
    )
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in args.roots]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Scanning for CSVs…")
    all_csvs = list_csvs(roots)

    distance_files = [fp for fp in all_csvs if fp.name.startswith("distance_")]
    base_files = [fp for fp in all_csvs if not fp.name.startswith("distance_")]

    print(f"Found total CSVs: {len(all_csvs)}")
    print(f"  - distance-mapped: {len(distance_files)}")
    print(f"  - base:            {len(base_files)}")

    if len(distance_files) == 0:
        print("WARNING: no distance-mapped CSVs found (prefix 'distance_').")
    if len(base_files) == 0:
        print("WARNING: no base CSVs found (without 'distance_' prefix).")

    # Distance model
    if distance_files:
        print("\nLoading distance-mapped tables…")
        Xd, yd, names_d = concat_tables(distance_files)
        train_and_save(Xd, yd, outdir, tag="distance", k_features=args.k_features)

    # Base model
    if base_files:
        print("\nLoading base tables…")
        Xb, yb, names_b = concat_tables(base_files)
        train_and_save(Xb, yb, outdir, tag="base", k_features=args.k_features)

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
