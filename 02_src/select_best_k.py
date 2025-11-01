#!/usr/bin/env python3
"""
Tongue-level holdout:
- Group by tongue image ID (folder right under 'mouse_tongue_subimages').
- Leave-One-Group-Out CV (each fold holds out a whole tongue).
- VarianceThreshold -> SelectKBest (k in {10,20,30,40,50}) -> RandomForest.
- Pick smallest k achieving ≥95% of peak accuracy; enforce the same k for base & distance.
- Save final models for base and distance at that k.

Run example at bottom.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys, time, random
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CLASS_COL_CANDIDATES = ["class", "Class", "label", "Label"]

def find_class_col(df: pd.DataFrame) -> str:
    for c in CLASS_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]

def is_distance_file(p: Path) -> bool:
    return p.name.startswith("distance_")

def tongue_id_from_path(p: Path) -> str:
    """
    Extract the tongue image ID as the directory immediately under 'mouse_tongue_subimages'.
    Example:
      .../mouse_tongue_subimages/Cd04M-155706-20170601-merge-BF/filiform_001_csv/...
      -> tongue_id = 'Cd04M-155706-20170601-merge-BF'
    Fallback: use p.parents[2] if the anchor isn't found.
    """
    parts = p.parts
    try:
        idx = parts.index("mouse_tongue_subimages")
        return parts[idx + 1]
    except ValueError:
        # anchor not found; best-effort fallback
        return p.parents[2].name if len(p.parents) > 2 else p.parents[0].name

def scan_csvs(root_dirs: list[Path]) -> tuple[list[Path], list[Path]]:
    print("Scanning for CSVs…")
    all_csvs = []
    for d in root_dirs:
        if not d.exists():
            print(f"  ! Skipping missing directory: {d}", file=sys.stderr)
            continue
        all_csvs.extend(sorted(d.rglob("*.csv")))
    dist = [p for p in all_csvs if is_distance_file(p)]
    base = [p for p in all_csvs if not is_distance_file(p)]
    print(f"Found total CSVs: {len(all_csvs)}")
    print(f"  - distance-mapped: {len(dist)}")
    print(f"  - base:            {len(base)}")
    return base, dist

def one_patch_per_tongue(files: list[Path], seed: int = 42) -> list[Path]:
    """
    Reduce to one CSV per tongue (for a quick reduced dataset), chosen at random.
    """
    rnd = random.Random(seed)
    by_tongue = {}
    for p in files:
        tid = tongue_id_from_path(p)
        by_tongue.setdefault(tid, []).append(p)
    chosen = [rnd.choice(v) for v in by_tongue.values()]
    print(f"  Reduced to one patch per tongue: {len(chosen)} files from {len(by_tongue)} tongues")
    return sorted(chosen)

def load_table(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    cls = find_class_col(df)
    y = df[cls].values
    Xdf = df.drop(columns=[cls]).select_dtypes(include=[np.number])
    if Xdf.shape[1] == 0:
        raise ValueError(f"No numeric feature columns in {csv_path}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return Xdf.values, y_enc, list(Xdf.columns)

def build_dataset(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Concatenate rows across files; group each row by tongue_id (not by file/patch).
    """
    X_list, y_list, g_list, feat_names = [], [], [], None
    for i, p in enumerate(files, 1):
        print(f"  - Loading {i}/{len(files)}: {p.name}")
        X, y, fn = load_table(p)
        if feat_names is None:
            feat_names = fn
        elif fn != feat_names:
            # align to intersection
            common = [c for c in feat_names if c in fn]
            if not common:
                raise ValueError(f"Feature mismatch; no common columns at {p}")
            # reload ensuring order
            df = pd.read_csv(p)
            cls = find_class_col(df)
            X = df.drop(columns=[cls]).select_dtypes(include=[np.number])[common].values
            feat_names = common
            # also align previously stacked arrays
            if X_list:
                idxs = [common.index(c) for c in common]  # no-op but explicit
                X_list = [Xi[:, idxs] for Xi in X_list]
        tid = tongue_id_from_path(p)
        X_list.append(X)
        y_list.append(y)
        g_list.append(np.array([tid] * len(y)))
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    groups = np.concatenate(g_list)  # tongue-level groups
    return X_all, y_all, groups, feat_names

def eval_k_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                vt: VarianceThreshold, k: int, score_func: str,
                seed: int) -> tuple[float, float]:
    Xv = vt.transform(X)
    k_eff = min(k, Xv.shape[1])
    if k_eff <= 0:
        return 0.0, 0.0
    skb = SelectKBest(f_classif if score_func == "f_classif" else mutual_info_classif, k=k_eff)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    logo = LeaveOneGroupOut()
    accs = []
    t0 = time.perf_counter()
    for tr, te in logo.split(Xv, y, groups):
        Xtr, Xte = Xv[tr], Xv[te]
        ytr, yte = y[tr], y[te]
        skb.fit(Xtr, ytr)
        clf.fit(skb.transform(Xtr), ytr)
        yhat = clf.predict(skb.transform(Xte))
        accs.append(accuracy_score(yte, yhat))
    elapsed = time.perf_counter() - t0
    return float(np.mean(accs)), float(elapsed)

def fit_final(X: np.ndarray, y: np.ndarray, vt: VarianceThreshold, k: int,
              score_func: str, seed: int):
    Xv = vt.transform(X)
    k_eff = min(k, Xv.shape[1])
    skb = SelectKBest(f_classif if score_func == "f_classif" else mutual_info_classif, k=k_eff)
    Xk = skb.fit_transform(Xv, y)
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed).fit(Xk, y)
    return skb, clf

def main():
    ap = argparse.ArgumentParser(description="Tongue-level holdout (LOGO) with VT + SelectKBest sweep.")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Folders that contain CSVs (recursively). Include all tongues you want considered.")
    ap.add_argument("--k_list", nargs="+", type=int, default=[10,20,30,40,50])
    ap.add_argument("--score_func", choices=["f_classif","mutual_info"], default="f_classif")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reduced_one_patch_per_tongue", action="store_true",
                    help="If set, subsample to one CSV per tongue for faster sweeps.")
    ap.add_argument("--outdir", type=str, default="models_k_select_by_tongue")
    args = ap.parse_args()

    roots = [Path(d).expanduser() for d in args.dirs]
    base_files, dist_files = scan_csvs(roots)

    if args.reduced_one_patch_per_tongue:
        print("\nReducing BASE to one patch per tongue…")
        base_files = one_patch_per_tongue(base_files, seed=args.seed)
        print("Reducing DISTANCE to one patch per tongue…")
        dist_files = one_patch_per_tongue(dist_files, seed=args.seed)

    print("\nLoading BASE dataset (grouped by tongue)…")
    Xb, yb, gb, fn_b = build_dataset(base_files)
    print(f"  Base X={Xb.shape}, y={yb.shape}, tongues={len(np.unique(gb))}")

    print("\nLoading DISTANCE dataset (grouped by tongue)…")
    Xd, yd, gd, fn_d = build_dataset(dist_files)
    print(f"  Distance X={Xd.shape}, y={yd.shape}, tongues={len(np.unique(gd))}")

    print("\nVarianceThreshold (0.0)…")
    vt_b = VarianceThreshold(0.0).fit(Xb)
    vt_d = VarianceThreshold(0.0).fit(Xd)
    print(f"  Base kept {vt_b.get_support().sum()} / {Xb.shape[1]} features")
    print(f"  Distance kept {vt_d.get_support().sum()} / {Xd.shape[1]} features")

    ks = sorted(args.k_list)
    rows = []
    print("\nSweeping k with Leave-One-Group-Out (entire tongue held out each fold)…")
    for k in ks:
        acc_b, t_b = eval_k_logo(Xb, yb, gb, vt_b, k, args.score_func, args.seed)
        acc_d, t_d = eval_k_logo(Xd, yd, gd, vt_d, k, args.score_func, args.seed)
        rows.append({"k": k, "base_acc": acc_b, "base_time_s": t_b,
                          "dist_acc": acc_d, "dist_time_s": t_d})
        print(f"  k={k:>2d} | base acc={acc_b:.4f}, {t_b:.1f}s | dist acc={acc_d:.4f}, {t_d:.1f}s")

    df = pd.DataFrame(rows).sort_values("k")
    peak_b, peak_d = df["base_acc"].max(), df["dist_acc"].max()
    thr_b, thr_d = 0.95*peak_b, 0.95*peak_d

    def pick(series, thr):
        ok = df.loc[series >= thr, "k"]
        return int(ok.min()) if not ok.empty else int(df.loc[series.idxmax(), "k"])

    k_b = pick(df["base_acc"], thr_b)
    k_d = pick(df["dist_acc"], thr_d)
    k_final = max(k_b, k_d)  # enforce identical feature budget

    print("\nSelection (≥95% of peak):")
    print(f"  Base peak {peak_b:.4f} → thr {thr_b:.4f} → k_b={k_b}")
    print(f"  Dist peak {peak_d:.4f} → thr {thr_d:.4f} → k_d={k_d}")
    print(f"  Using CONSISTENT k_final={k_final}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "k_sweep_by_tongue.csv", index=False)

    print("\nFitting final models at k_final…")
    skb_b, clf_b = fit_final(Xb, yb, vt_b, k_final, args.score_func, args.seed)
    joblib.dump({"variance_threshold": vt_b, "select_k": skb_b, "clf": clf_b},
                outdir / f"model_base_tongue_k{k_final}.joblib")

    skb_d, clf_d = fit_final(Xd, yd, vt_d, k_final, args.score_func, args.seed)
    joblib.dump({"variance_threshold": vt_d, "select_k": skb_d, "clf": clf_d},
                outdir / f"model_distance_tongue_k{k_final}.joblib")

    print("\nSaved:")
    print(f"  - {outdir/'model_base_tongue_k'+str(k_final)+'.joblib'}")
    print(f"  - {outdir/'model_distance_tongue_k'+str(k_final)+'.joblib'}")
    print(f"  - {outdir/'k_sweep_by_tongue.csv'}")

if __name__ == "__main__":
    main()
