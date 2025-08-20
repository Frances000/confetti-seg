# train_with_progress.py  (updated)
import tkinter as tk
import threading
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import joblib

from ui_progress import TrainingProgressDialog


# ---------------------------
# Loading helpers
# ---------------------------

def load_flat_stack(csv_path: Path, class_column: str = "class"):
    """Back-compat: load a single CSV (pixel holdout)."""
    df = pd.read_csv(csv_path)
    if class_column not in df.columns:
        raise ValueError(f"Expected '{class_column}' in {csv_path.name}")
    X = df.drop(columns=[class_column])
    y = df[class_column].astype(str)  # keep letter codes
    return X, y


def load_csv_folder(csv_dir: Path, class_column: str = "class"):
    """
    Load all CSVs in a directory. Each CSV = one patch.
    Returns:
      X : DataFrame (rows = pixels across all patches)
      y : Series[str]  (two-letter class codes)
      groups : Series[str]  (patch_id for each row)
      patch_files : list[Path]  (for reference)
    Notes:
      - Columns are aligned across patches (outer join); missing features filled with 0.
      - We add a 'patch_id' taken from CSV filename stem.
    """
    csv_dir = Path(csv_dir)
    files = sorted([p for p in csv_dir.glob("*.csv") if p.is_file()])
    if not files:
        raise ValueError(f"No CSVs found in {csv_dir}")

    frames = []
    for p in files:
        df = pd.read_csv(p)
        if class_column not in df.columns:
            raise ValueError(f"Expected '{class_column}' in {p.name}")
        df["patch_id"] = p.stem  # tag rows with patch id
        frames.append(df)

    # Align columns (outer join) and fill missing features with 0
    big = pd.concat(frames, axis=0, join="outer", ignore_index=True)
    # move class + patch to front
    cols = list(big.columns)
    cols.remove(class_column)
    cols.remove("patch_id")
    big = big[[class_column, "patch_id"] + cols].copy()
    # fill NaNs for features only (not class/patch)
    feat_cols = [c for c in big.columns if c not in (class_column, "patch_id")]
    big[feat_cols] = big[feat_cols].fillna(0)

    y = big[class_column].astype(str)
    groups = big["patch_id"].astype(str)
    X = big.drop(columns=[class_column, "patch_id"])

    return X, y, groups, files


# ---------------------------
# Training (grouped by patch)
# ---------------------------

def train_with_progress_grouped(X, y, groups, out_dir: Path, reporter, cancel_flag, n_splits=5, test_size=0.2):
    """
    Patch-level holdout: GroupShuffleSplit for test; GroupKFold for CV.
    y is multi-class (letter codes). XGBoost uses label-encoded y internally.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Split by patch (no pixel leakage)
    reporter("Splitting train/test by PATCH…", step_delta=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    (train_idx, test_idx), = gss.split(X, y, groups=groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    if cancel_flag(): return {"status": "cancelled"}

    # ---- Prepare models
    # RF/DT/SVM accept string labels directly. XGB needs numeric classes.
    le = LabelEncoder().fit(y_train)
    n_classes = len(le.classes_)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", C=4.0, gamma="scale", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            n_jobs=-1, random_state=42
        ),
    }

    # ---- Grouped CV setup
    # Ensure we have at least n_splits patches in the TRAIN set
    n_train_groups = groups_train.nunique()
    if n_train_groups < n_splits:
        reporter(f"⚠️ Only {n_train_groups} train patches; reducing CV folds to {n_train_groups}.", step_delta=0)
        n_splits = int(n_train_groups)
    gkf = GroupKFold(n_splits=n_splits)

    results = {}
    for name, model in models.items():
        if cancel_flag(): return {"status": "cancelled"}
        reporter(f"Fitting {name} on train (patch-held-out)…", step_delta=1)

        # Fit (handle XGB label encoding)
        if name == "XGBoost":
            model.fit(X_train, le.transform(y_train))
        else:
            model.fit(X_train, y_train)

        # Persist model (+ label encoder for XGB)
        joblib.dump(model, out_dir / f"{name}_model.joblib")
        if name == "XGBoost":
            joblib.dump({"classes_": le.classes_}, out_dir / f"{name}_label_encoder.joblib")

        # ---- Manual Grouped CV
        fold_scores = []
        fold_num = 0
        for tr_idx, te_idx in gkf.split(X_train, y_train, groups=groups_train):
            if cancel_flag(): return {"status": "cancelled"}
            fold_num += 1
            reporter(f"{name}: CV fold {fold_num}/{n_splits} (patch groups)…", step_delta=1)

            X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
            y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

            mdl = model.__class__(**getattr(model, "get_params")())
            if name == "XGBoost":
                mdl.fit(X_tr, le.transform(y_tr))
                y_hat = le.inverse_transform(mdl.predict(X_te).astype(int)) \
                        if n_classes == 2 else le.inverse_transform(mdl.predict(X_te).astype(int))
                # Prefer predict over predict_proba for acc; XGB's predict gives class ids in multi:softprob
                if mdl.get_xgb_params().get("objective", "").startswith("multi"):
                    y_hat = le.inverse_transform(mdl.predict(X_te).astype(int))
                else:
                    y_hat = (mdl.predict_proba(X_te)[:,1] >= 0.5).astype(int)
                    y_hat = le.inverse_transform(y_hat)
            else:
                mdl.fit(X_tr, y_tr)
                y_hat = mdl.predict(X_te)

            fold_acc = (y_hat == y_te).mean()
            fold_scores.append(fold_acc)

        # ---- Test evaluation (on held-out patches)
        if cancel_flag(): return {"status": "cancelled"}
        reporter(f"Evaluating {name} on held-out PATCH test…", step_delta=1)

        if name == "XGBoost":
            if n_classes > 2:
                y_pred = le.inverse_transform(models["XGBoost"].predict(X_test).astype(int))
            else:
                y_pred_bin = (models["XGBoost"].predict_proba(X_test)[:,1] >= 0.5).astype(int)
                y_pred = le.inverse_transform(y_pred_bin)
        else:
            y_pred = model.predict(X_test)

        acc = float((y_pred == y_test).mean())
        cr = classification_report(y_test, y_pred, zero_division=0, output_dict=False)
        labels_sorted = sorted(np.unique(np.concatenate([y_test.values, y_pred])))
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        results[name] = {
            "test_accuracy": acc,
            "cv_mean_accuracy": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "labels": list(labels_sorted),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
        }
        reporter(f"{name} ✓ Test acc={acc:.3f}, CV mean={np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    # ---- Unsupervised sanity check (KMeans) on all data
    if cancel_flag(): return {"status": "cancelled"}
    reporter("Running KMeans (unsupervised, all pixels)…", step_delta=1)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X)
    ari = adjusted_rand_score(y, kmeans.labels_)
    results["KMeans"] = {"adjusted_rand_index": float(ari)}
    reporter(f"KMeans ✓ ARI (vs ground truth) = {ari:.3f}", step_delta=1)

    return {"status": "ok", "results": results}


def estimate_total_steps(n_models=4, n_splits=5):
    # 1 split + per model: 1 fit + n_splits CV folds + 1 eval + 2 for KMeans
    return 1 + n_models * (1 + n_splits + 1) + 2


# ---------------------------
# GUI entrypoints
# ---------------------------

def launch_training_gui(input_path: Path, out_dir: Path, class_column="class", n_splits=5, test_size=0.2):
    """
    If input_path is a file -> single CSV (pixel-holdout, back-compat).
    If input_path is a directory -> MULTI CSVs (patch-holdout; recommended).
    """
    root = tk.Tk()
    root.withdraw()

    total = estimate_total_steps(n_models=4, n_splits=n_splits)
    dlg = TrainingProgressDialog(root, total_steps=total, title="Training progress")
    reporter = dlg.make_reporter()

    def cancel_flag():
        return dlg.cancelled

    def worker():
        try:
            ip = Path(input_path)
            if ip.is_dir():
                reporter(f"Loading CSV patches from folder '{ip.name}'…", step_delta=1)
                X, y, groups, files = load_csv_folder(ip, class_column=class_column)
                res = train_with_progress_grouped(
                    X, y, groups, out_dir, reporter, cancel_flag, n_splits=n_splits, test_size=test_size
                )
            else:
                reporter(f"Loading data from {ip.name}…", step_delta=1)
                X, y = load_flat_stack(ip, class_column=class_column)
                # Make a dummy groups vector (pixel-holdout mode)
                groups = pd.Series(np.zeros(len(y), dtype=int))
                res = train_with_progress_grouped(
                    X, y, groups, out_dir, reporter, cancel_flag, n_splits=n_splits, test_size=test_size
                )

            if res.get("status") == "ok":
                reporter("✅ Training complete.", step_delta=0)
            else:
                reporter("🛑 Training cancelled.", step_delta=0)
        except Exception as e:
            reporter(f"❌ Error: {e}", step_delta=0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    root.mainloop()


# --- Example run ---
if __name__ == "__main__":
    # Directory mode (PATCH holdout): point to a folder full of per-patch CSVs
    csv_dir = Path("/Users/franceskan/Documents/confetti-seg/output/filiform_001/tester/csv")  # <-- each *.csv is a patch
    out_dir = Path("trained_models")
    launch_training_gui(csv_dir, out_dir, class_column="class", n_splits=5, test_size=0.2)
