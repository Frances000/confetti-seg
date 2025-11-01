import argparse
from pathlib import Path
import warnings
import re
import numpy as np, time
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    cohen_kappa_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score, matthews_corrcoef,
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning, FitFailedWarning

# Quiet noisy warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
np.seterr(all="ignore")

_MODEL_KIND_PATTERNS = [
    (re.compile(r"model_base_(rf|dt|svm|xgb)\.joblib$", re.IGNORECASE), "base"),
    (re.compile(r"model_distance_(rf|dt|svm|xgb)\.joblib$", re.IGNORECASE), "distance"),
]

# --- add near top of file ---
CLASS_ORDER_DEFAULT = [
    "BB","BC","BG","BR","BY",
    "CC","CG","CR","CY",
    "GG","GR","GY",
    "RR","RY",
    "YY",
]

def resolve_label_order(y_true_s=None, y_pred_s=None, prefer=CLASS_ORDER_DEFAULT):
    """Return a label list in the exact preferred order, filtered to those present.
       Any unexpected labels not in 'prefer' are appended (stable, alpha)."""
    present = set()
    if y_true_s is not None:
        present.update(np.asarray(y_true_s).astype(str))
    if y_pred_s is not None:
        present.update(np.asarray(y_pred_s).astype(str))
    # keep only those that are present
    ordered = [lab for lab in prefer if lab in present]
    # append any extras not in the preferred list (rare)
    extras = sorted([lab for lab in present if lab not in prefer])
    return ordered + extras

def infer_model_kind_and_tag(model_path: Path) -> tuple[str, str]:
    """
    Return ('base'|'distance', 'rf'|'dt'|'svm'|'xgb'|'unknown') from filename.
    Falls back to folder name hints if necessary.
    """
    name = model_path.name
    for pat, kind in _MODEL_KIND_PATTERNS:
        m = pat.search(name)
        if m:
            return kind, m.group(1).lower()

    # Fallbacks:
    kind = "distance" if "distance" in name.lower() or "distance" in model_path.parent.name.lower() else "base"
    tag = "unknown"
    for t in ("rf", "dt", "svm", "xgb"):
        if f"_{t}" in name.lower():
            tag = t
            break
    return kind, tag

# --- lightweight TIFF writer: prefer tifffile, fallback to imageio ---
def _tiff_write(path, arr):
    try:
        import tifffile as tiff  # pip install tifffile
        tiff.imwrite(str(path), arr)
    except Exception:
        import imageio.v3 as iio  # works for basic TIFF if tifffile unavailable
        iio.imwrite(str(path), arr, extension=".tif")

def load_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = None
    if "class" in df.columns:
        y = df["class"].astype(str).to_numpy()
        X = df.drop(columns=["class"])
    else:
        X = df
    return X, y

def _labels_to_index(arr_like, lab2idx, default_idx=0):
    # Map any label (num/str) to palette index; unknown → default_idx
    arr = np.asarray(arr_like)
    flat = arr.reshape(-1)
    out = np.fromiter((lab2idx.get(str(v).strip(), default_idx) for v in flat),
                      count=flat.size, dtype=np.int32)
    return out.reshape(arr.shape)

def _stack_panels_horizontal(*imgs, gap_px=4):
    # imgs: list of HxWx3 uint8
    H = max(im.shape[0] for im in imgs)
    W = sum(im.shape[1] for im in imgs) + gap_px * (len(imgs) - 1)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    x = 0
    for i, im in enumerate(imgs):
        h, w = im.shape[:2]
        out[:h, x:x+w, :] = im
        x += w
        if i < len(imgs)-1:
            out[:, x:x+gap_px, :] = 255  # white gap
            x += gap_px
    return out

def _render_label_rgb(idx_img, palette):
    return palette[idx_img].astype(np.uint8)

def save_side_by_side_tiff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_hw: tuple[int, int],
    out_base: Path,
    labels_sorted: np.ndarray,
    show_diff_panel: bool = True,
    diff_color: tuple[int, int, int] = (255, 0, 0)
):
    """
    Writes a single RGB TIFF with panels:
      [ GT | PRED | (optional) DIFF ]
    Also writes each panel individually for convenience.

    Returns: (compare_path, gt_rgb_path, pred_rgb_path, diff_rgb_path_or_None)
    """
    H, W = shape_hw
    # Build deterministic palette (b/r/y/g/c fixed)
    lab2idx, palette = build_palette(labels_sorted)

    # Prepare indexed images using SAME mapping for both GT and PRED
    idx_true = _labels_to_index(y_true.reshape(H, W), lab2idx, default_idx=0)
    idx_pred = _labels_to_index(y_pred.reshape(H, W), lab2idx, default_idx=0)

    # Render RGB panels
    gt_rgb   = _render_label_rgb(idx_true, palette)
    pred_rgb = _render_label_rgb(idx_pred, palette)

    # Optional DIFF panel: highlight mismatches in diff_color, matches = greyscale GT
    diff_rgb = None
    if show_diff_panel:
        mism = (idx_true != idx_pred)
        # greyscale GT as background
        gt_gray = (0.299*gt_rgb[...,0] + 0.587*gt_rgb[...,1] + 0.114*gt_rgb[...,2]).astype(np.uint8)
        diff_rgb = np.stack([gt_gray, gt_gray, gt_gray], axis=-1)
        # paint mismatches
        dr, dg, db = diff_color
        diff_rgb[mism] = np.array([dr, dg, db], dtype=np.uint8)

    # Save individual panels
    gt_rgb_path   = out_base.with_suffix("").as_posix() + "_gt_rgb.tif"
    pred_rgb_path = out_base.with_suffix("").as_posix() + "_pred_rgb.tif"
    _tiff_write(gt_rgb_path, gt_rgb)
    _tiff_write(pred_rgb_path, pred_rgb)

    diff_rgb_path = None
    if diff_rgb is not None:
        diff_rgb_path = out_base.with_suffix("").as_posix() + "_diff_rgb.tif"
        _tiff_write(diff_rgb_path, diff_rgb)

    # Compose side-by-side
    panels = [gt_rgb, pred_rgb] + ([diff_rgb] if diff_rgb is not None else [])
    compare = _stack_panels_horizontal(*panels, gap_px=4)
    compare_path = out_base.with_suffix("").as_posix() + "_compare_rgb.tif"
    _tiff_write(compare_path, compare)

    # Also refresh legend to match the palette used
    legend_path = out_base.with_suffix("").as_posix() + "_legend.csv"
    legend = pd.DataFrame({
        "index": np.arange(len(labels_sorted), dtype=int),
        "label": labels_sorted.astype(str),
        "R": palette[:,0],
        "G": palette[:,1],
        "B": palette[:,2],
    })
    Path(legend_path).parent.mkdir(parents=True, exist_ok=True)
    legend.to_csv(legend_path, index=False)

    return Path(compare_path), Path(gt_rgb_path), Path(pred_rgb_path), (Path(diff_rgb_path) if diff_rgb_path else None), Path(legend_path)

def align_columns_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure X has the exact columns the model was trained on."""
    if hasattr(model, "feature_names_in_"):
        want = list(model.feature_names_in_)
        for c in want:
            if c not in X.columns:
                X[c] = 0
        X = X[want]
    return X

def parse_shape(arg: str, n_rows: int) -> tuple[int, int]:
    if arg:
        s = arg.lower().replace("x", ",")
        parts = [p for p in s.split(",") if p.strip()]
        if len(parts) == 2:
            H, W = int(parts[0]), int(parts[1])
            if H * W != n_rows:
                raise ValueError(f"--shape {H}x{W} != number of rows ({n_rows})")
            return H, W
        raise ValueError("Use --shape like '224x160' or '224,160'")
    r = int(np.sqrt(n_rows))
    for h in range(r, 0, -1):
        if n_rows % h == 0:
            return h, n_rows // h
    return n_rows, 1

def _srgb_to_linear_byte(b: int) -> float:
    c = b / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb_byte(x: float) -> int:
    c = x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1/2.4)) - 0.055
    return int(round(max(0.0, min(1.0, c)) * 255))

def _mix_colors(cols, gamma_correct=True):
    if not cols:
        return (0, 0, 0)
    if gamma_correct:
        # average in linear space, then convert back to sRGB
        lin = [[_srgb_to_linear_byte(c[i]) for c in cols] for i in range(3)]
        avg = [sum(ch)/len(ch) for ch in lin]
        return tuple(_linear_to_srgb_byte(v) for v in avg)
    else:
        return tuple(int(round(sum(c[i] for c in cols) / len(cols))) for i in range(3))

def build_palette(labels_order: np.ndarray, *, gamma_correct: bool = True) -> tuple[dict[str, int], np.ndarray]:
    """
    Mixture-aware palette. Labels may be 'B','R','Y','G','C' or mixtures like 'BC','GR','RY', etc.
    Order is preserved (legend indices match labels_order).
    """
    CANON = {
        "B": (0, 0, 0),
        "R": (255, 0, 0),
        "Y": (255, 255, 0),
        "G": (0, 255, 0),
        "C": (0, 255, 255),
    }
    labels = [str(x).strip() for x in labels_order]
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    rgb_list: list[tuple[int, int, int]] = []

    for lab in labels:
        # Treat mixtures as order independent ('BG' == 'GB')
        chars = [ch.upper() for ch in lab if ch.strip()]
        # if any unknown component, fall back to distinct HSV (handled below)
        if all(ch in CANON for ch in chars) and len(chars) >= 1:
            cols = [CANON[ch] for ch in chars]
            rgb_list.append(_mix_colors(cols, gamma_correct=gamma_correct))
        else:
            rgb_list.append(None)  # fill later via HSV fallback

    # Assign stable HSV colours for truly unknown labels (should be none for your set)
    if any(v is None for v in rgb_list):
        import colorsys
        unknown_idxs = [i for i, v in enumerate(rgb_list) if v is None]
        # stable order by label text so it doesn't shuffle between runs
        unknown_labs_sorted = sorted((labels[i] for i in unknown_idxs))
        k = len(unknown_labs_sorted)
        for j, lab in enumerate(unknown_labs_sorted):
            h = (j / max(k, 1)) % 1.0
            s, v = 0.85, 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            col = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
            idx = labels.index(lab)
            rgb_list[idx] = col

    rgb8 = np.array(rgb_list, dtype=np.uint8)
    return lab2idx, rgb8


def save_prediction_tiffs(y_hat: np.ndarray, shape_hw: tuple[int,int],
                          out_base: Path, labels_sorted: np.ndarray):
    H, W = shape_hw
    lab2idx, palette = build_palette(labels_sorted)
    idx = np.vectorize(lab2idx.get)(y_hat.astype(str))
    idx_img = idx.reshape(H, W)
    idx_img = idx_img.astype(np.uint16 if idx_img.max() >= 256 else np.uint8)

    label_path = out_base.with_suffix("").as_posix() + "_labels.tif"
    color_path = out_base.with_suffix("").as_posix() + "_color.tif"
    legend_path = out_base.with_suffix("").as_posix() + "_legend.csv"

    _tiff_write(label_path, idx_img)
    color_img = palette[idx].reshape(H, W, 3).astype(np.uint8)
    _tiff_write(color_path, color_img)

    legend = pd.DataFrame({
        "index": np.arange(len(labels_sorted), dtype=int),
        "label": labels_sorted,
        "R": palette[:, 0],
        "G": palette[:, 1],
        "B": palette[:, 2],
    })
    Path(legend_path).parent.mkdir(parents=True, exist_ok=True)
    legend.to_csv(legend_path, index=False)
    return Path(label_path), Path(color_path), Path(legend_path)

def _augment_out_path(out_path: Path, csv_path: Path, model_kind: str, model_tag: str) -> Path:
    """
    Ensure the output filename contains '__<model_kind>_<model_tag>__'.
    If the provided name already contains 'base' or 'distance', leave it.
    """
    stem = out_path.stem
    lower = stem.lower()
    if ("__base_" in lower or "__distance_" in lower) or (" base_" in lower or " distance_" in lower):
        return out_path  # already annotated

    # Build a descriptive stem: <csv_stem>__<kind>_<tag>__eval
    csv_stem = csv_path.stem
    new_stem = f"{csv_stem}__{model_kind}_{model_tag}__eval"
    return out_path.with_name(new_stem + out_path.suffix)

def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained model on a single CSV stack and optionally write TIFF prediction maps.")
    ap.add_argument("--model_path", required=True, help="Path to .joblib model (e.g model_base_rf.joblib)")
    ap.add_argument("--csv_path",   required=True, help="Path to CSV with columns (incl. optional 'class')")
    ap.add_argument("--shape",      default=None,  help="Image shape as HxW (e.g 224x160). Required to write TIFFs.")
    ap.add_argument("--pred_tiff",  default=None,  help="Basename for prediction TIFFs (no extension). Example: /path/patch_00_preds")
    ap.add_argument("--out",        default=None,  help="TXT report path (default autoannotated to include base/distance & model tag)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    csv_path   = Path(args.csv_path)

    # Infer model kind/tag and build default output filename
    model_kind, model_tag = infer_model_kind_and_tag(model_path)
    default_out = csv_path.with_name(f"{csv_path.stem}__{model_kind}_{model_tag}__eval.txt")
    out_path = Path(args.out) if args.out else default_out
    out_path = _augment_out_path(out_path, csv_path, model_kind, model_tag)

    pred_base  = Path(args.pred_tiff) if args.pred_tiff else csv_path.with_name(f"{csv_path.stem}__{model_kind}_{model_tag}__preds.tif")

    # load
    model = joblib.load(model_path)
    X, y = load_csv(csv_path)

    # align features
    X_in = align_columns_to_model(X.copy(), model)

    # predict
    # predict
    y_hat = model.predict(X_in)
    n = min(len(y_hat), len(y_hat) if y is None else len(y))
    y_hat = y_hat[:n]

    # --- NORMALISE PREDICTIONS (handles XGBoost numeric outputs) ---
    # If model has a classes_ mapping and predictions are numeric, map back to label strings
    if hasattr(model, "classes_") and np.issubdtype(np.asarray(y_hat).dtype, np.number):
        y_hat = np.asarray(model.classes_)[y_hat]

    if y is not None:
        # --- make BOTH sides strings consistently ---
        y_eval_s = np.asarray(y[:n]).astype(str)
        y_hat_s  = np.asarray(y_hat).astype(str)

        acc = float((y_hat_s == y_eval_s).mean())

        # consistent label list as strings (list, not ndarray)
        # AFTER construct y_eval_s and y_hat_s
        labels_sorted = resolve_label_order(y_true_s=y_eval_s, y_pred_s=y_hat_s)


        # all downstream metrics MUST use *_s and labels_sorted
        cm = confusion_matrix(y_eval_s, y_hat_s, labels=labels_sorted)
        cr = classification_report(y_eval_s, y_hat_s, labels=labels_sorted, zero_division=0)
    else:
        acc = None
        labels_sorted = np.unique(np.asarray(y_hat).astype(str)).tolist()
        cm = None
        cr = None



        # write report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        print(f"Model: {model_path}", file=f)
        print(f"ModelKind: {model_kind}", file=f)
        print(f"ModelTag: {model_tag}", file=f)
        print(f"CSV:   {csv_path}", file=f)
        print(f"Rows:  {len(X_in)}", file=f)
        if hasattr(model, 'feature_names_in_'):
            print(f"Features used: {len(model.feature_names_in_)}", file=f)
        else:
            print(f"Features used: {X_in.shape[1]}", file=f)

        # -------------------------
        # Evaluation with ground truth
        # -------------------------
        if y_eval_s is not None:
            eval_start = time.time()

            # Existing summary (you already computed: n, acc, cm, cr, labels_sorted, y_hat)
            print("\n=== Evaluation (with ground truth) ===", file=f)
            print(f"Pixels: {n}", file=f)
            print(f"Accuracy: {acc:.6f}", file=f)
            print(f"Misclassification rate: {1.0 - acc:.6f}", file=f)
            print(f"Labels: {list(labels_sorted)}", file=f)

            # Extra statistics
            try:
                kappa = cohen_kappa_score(y_eval_s, y_hat_s)
            except Exception:
                kappa = np.nan
            try:
                bal_acc = balanced_accuracy_score(y_eval_s, y_hat_s)
            except Exception:
                bal_acc = np.nan
            try:
                mcc = matthews_corrcoef(y_eval_s, y_hat_s)
            except Exception:
                mcc = np.nan

            # Macro/weighted precision/recall/F1
            # Per-class as well (aligned to labels_sorted)
            try:
                per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
                    y_eval_s, y_hat_s, labels=labels_sorted, average=None, zero_division=0
                )
                macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
                    y_eval_s, y_hat_s, average="macro", zero_division=0
                )
                weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
                    y_eval_s, y_hat_s, average="weighted", zero_division=0
                )
            except Exception:
                per_prec = per_rec = per_f1 = per_sup = None
                macro_prec = macro_rec = macro_f1 = np.nan
                weighted_prec = weighted_rec = weighted_f1 = np.nan

            # ROC AUC (if probabilities available)
            auc_line = "ROC AUC: not computed"
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_in)
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        # binary
                        auc_val = roc_auc_score(y, y_prob[:, 1])
                    else:
                        # multiclass OVR macro
                        auc_val = roc_auc_score(y, y_prob, multi_class="ovr")
                    auc_line = f"ROC AUC: {auc_val:.6f}"
                except Exception as _e:
                    auc_line = f"ROC AUC: not computed"

            # Print extra stats
            print("\n--- Extra statistics ---", file=f)
            print(f"Cohen's kappa: {kappa:.6f}", file=f)
            print(f"Balanced accuracy: {bal_acc:.6f}", file=f)
            print(f"Matthews corr. coeff. (MCC): {mcc:.6f}", file=f)
            print(f"Macro P/R/F1: {macro_prec:.6f} / {macro_rec:.6f} / {macro_f1:.6f}", file=f)
            print(f"Weighted P/R/F1: {weighted_prec:.6f} / {weighted_rec:.6f} / {weighted_f1:.6f}", file=f)
            print(auc_line, file=f)

            # Confusion matrix + top confused pairs
            print("\nConfusion matrix:", file=f)
            print(cm, file=f)

            # Identify top-N most confused label pairs (off-diagonals)
            try:
                cm_arr = np.asarray(cm, dtype=float)
                off_diag = []
                for i in range(cm_arr.shape[0]):
                    for j in range(cm_arr.shape[1]):
                        if i != j and cm_arr[i, j] > 0:
                            off_diag.append((labels_sorted[i], labels_sorted[j], int(cm_arr[i, j])))
                off_diag.sort(key=lambda t: t[2], reverse=True)
                top_n = 10 if len(off_diag) > 10 else len(off_diag)
                print("\nTop confused label pairs (true → predicted):", file=f)
                if top_n == 0:
                    print("  (none)", file=f)
                else:
                    for k in range(top_n):
                        ti, pj, cnt = off_diag[k]
                        print(f"  {ti} → {pj}: {cnt}", file=f)
            except Exception:
                print("\nTop confused label pairs: (not computed)", file=f)

            # Per-class table
            if per_prec is not None:
                print("\nPer-class metrics:", file=f)
                print("label,precision,recall,f1,support", file=f)
                for lbl, p, r, f1, sup in zip(labels_sorted, per_prec, per_rec, per_f1, per_sup):
                    print(f"{lbl},{p:.6f},{r:.6f},{f1:.6f},{int(sup)}", file=f)

            # Class distributions
            try:
                true_counts = {str(cls): int(cnt) for cls, cnt in zip(labels_sorted, np.bincount(
                    np.searchsorted(labels_sorted, y), minlength=len(labels_sorted)))}
            except Exception:
                # fallback using pandas if y is not numeric sorted accordingly
                import pandas as _pd
                vc_true = _pd.Series(y_eval_s).value_counts().sort_index()
                true_counts = {str(k): int(v) for k, v in vc_true.items()}

            try:
                pred_counts = {str(cls): int(cnt) for cls, cnt in zip(labels_sorted, np.bincount(
                    np.searchsorted(labels_sorted, y_hat), minlength=len(labels_sorted)))}
            except Exception:
                import pandas as _pd
                vc_pred = _pd.Series(y_hat_s).value_counts().sort_index()
                pred_counts = {str(k): int(v) for k, v in vc_pred.items()}

            print("\nClass counts (true):", file=f)
            for cls in labels_sorted:
                print(f"  {cls}: {true_counts.get(str(cls), 0)}", file=f)

            print("Class counts (predicted):", file=f)
            for cls in labels_sorted:
                print(f"  {cls}: {pred_counts.get(str(cls), 0)}", file=f)

            # Your original textual report (sklearn string)
            print("\nClassification report:", file=f)
            print(cr, file=f)

            eval_time = time.time() - eval_start
            print(f"\nEval runtime (s): {eval_time:.3f}", file=f)

        else:
            # -------------------------
            # No ground truth available
            # -------------------------
            print("\n=== Predictions (no ground truth found) ===", file=f)
            vc = pd.Series(y_hat).value_counts().sort_index()
            print("Class counts:", file=f)
            for cls, cnt in vc.items():
                print(f"  {cls}: {cnt}", file=f)

    print(f"Report written to: {out_path}")
    H = W = None
    if args.shape:
        H, W = parse_shape(args.shape, n)

    if args.shape:
        # strip the .tif extension from pred_base because save_prediction_tiffs adds its own suffixes
        pred_base_noext = pred_base.with_suffix("")
        pred_base_noext.parent.mkdir(parents=True, exist_ok=True)
        # write labels with colour classes
        label_path, color_path, legend_path = save_prediction_tiffs(y_hat=y_hat, 
                                                                    shape_hw=(H, W), 
                                                                    out_base=pred_base_noext, 
                                                                    labels_sorted=np.array(labels_sorted, dtype=str))
        print(f"Wrote label TIFF: {label_path}")
        print(f"Wrote color TIFF: {color_path}")
        print(f"Legend: {legend_path}")


    # --- TIFF outputs (if shape was provided) ---
    if args.shape and y is not None:
        H, W = parse_shape(args.shape, n)
        # Ensure y and y_hat are length H*W in the SAME raster order as the CSV
        compare_path, gt_rgb_path, pred_rgb_path, diff_rgb_path, legend_path = save_side_by_side_tiff(
            y_true=y,                    # ground truth labels vector (len = H*W)
            y_pred=y_hat,                # predicted labels vector (len = H*W)
            shape_hw=(H, W),
            out_base=pred_base,          # same base you use for other outputs
            labels_sorted=np.array(labels_sorted, dtype=str),
            show_diff_panel=True,
            diff_color=(255, 0, 0)       # red for mismatches
        )
        print(f"Wrote comparison TIFF: {compare_path}")
        print(f"Panels: GT={gt_rgb_path}, PRED={pred_rgb_path}, DIFF={diff_rgb_path}")
        print(f"Legend: {legend_path}")
    else:
        print("No --shape provided; skipping TIFF export. Use --shape HxW to enable.")

if __name__ == "__main__":
    main()
