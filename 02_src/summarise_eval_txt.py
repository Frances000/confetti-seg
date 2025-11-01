#!/usr/bin/env python3
import argparse, re, sys, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# ---- Patterns for keyvalue lines already present ----
PAT_KV = re.compile(r"^(Model|ModelKind|ModelTag|CSV|Rows|Pixels|Accuracy|Features used):\s*(.*)$")
PAT_LABELS = re.compile(r"^Labels:\s*\[(.*)\]\s*$")

# ---- Patterns for new statistics----
PAT_MISCLASS = re.compile(r"^Misclassification rate:\s*([0-9.]+)")
PAT_KAPPA = re.compile(r"^Cohen's kappa:\s*([0-9.+-eE]+)")
PAT_BALACC = re.compile(r"^Balanced accuracy:\s*([0-9.+-eE]+)")
PAT_MCC = re.compile(r"^Matthews corr\. coeff\. \(MCC\):\s*([0-9.+-eE]+)")
PAT_MACRO = re.compile(r"^Macro P/R/F1:\s*([0-9.+-eE]+)\s*/\s*([0-9.+-eE]+)\s*/\s*([0-9.+-eE]+)")
PAT_WEIGHTED = re.compile(r"^Weighted P/R/F1:\s*([0-9.+-eE]+)\s*/\s*([0-9.+-eE]+)\s*/\s*([0-9.+-eE]+)")
PAT_AUC = re.compile(r"^ROC AUC:\s*([0-9.+-eE]+)")
PAT_EVAL_RUNTIME = re.compile(r"^Eval runtime \(s\):\s*([0-9.+-eE]+)")

# Section headers written by the report
HDR_EVAL = "=== Evaluation (with ground truth) ==="
HDR_CONFMAT = "Confusion matrix:"
HDR_TOPCONF = "Top confused label pairs (true → predicted):"
HDR_PERCLASS = "Per-class metrics:"
HDR_CLASS_TRUE = "Class counts (true):"
HDR_CLASS_PRED = "Class counts (predicted):"
HDR_CLASS_COUNTS_GENERIC = "Class counts:"  # when no ground truth
HDR_CLF_REPORT = "Classification report:"

def _parse_labels_block(s: str) -> List[str]:
    if not s:
        return []
    labs = [x.strip().strip("'\"") for x in s.split(",")]
    return [x for x in labs if x != ""]

def _read_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """
    Read a block (multi-line) until an empty line or a new section header.
    Returns (block_text, next_index_after_block).
    """
    buf = []
    i = start_idx
    while i < len(lines):
        ln = lines[i].rstrip("\n")
        if not ln.strip():
            break
        # stop if we hit any known header of another section
        if ln.startswith(HDR_TOPCONF) or ln.startswith(HDR_PERCLASS) or \
           ln.startswith(HDR_CLASS_TRUE) or ln.startswith(HDR_CLASS_PRED) or \
           ln.startswith(HDR_CLF_REPORT):
            break
        buf.append(ln)
        i += 1
    return ("\n".join(buf).strip(), i)

def _parse_top_confusions(lines: List[str], start_idx: int, report_path: str):
    rows = []
    i = start_idx
    # skip the header line itself
    i += 1
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            break
        # next section?
        if ln.startswith(HDR_PERCLASS) or ln.startswith(HDR_CLASS_TRUE) or ln.startswith(HDR_CLASS_PRED) or ln.startswith(HDR_CLF_REPORT):
            break
        # Format: "label_i → label_j: 12"
        m = re.match(r"^(.*)\s+→\s+(.*):\s*([0-9]+)$", ln)
        if m:
            rows.append({
                "report_path": report_path,
                "true_label": m.group(1).strip(),
                "pred_label": m.group(2).strip(),
                "count": int(m.group(3))
            })
        i += 1
    return rows, i

def _parse_per_class(lines: List[str], start_idx: int, report_path: str):
    """
    After 'Per-class metrics:' we expect:
      label,precision,recall,f1,support
      LBL,0.123...,0.456...,0.789...,12
      ...
    """
    out = []
    i = start_idx + 1
    # Skip header line if present
    if i < len(lines) and lines[i].strip().lower().startswith("label,"):
        i += 1
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            break
        if ln.startswith(HDR_CLASS_TRUE) or ln.startswith(HDR_CLASS_PRED) or ln.startswith(HDR_CLF_REPORT):
            break
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 5:
            lbl, p, r, f1, sup = parts
            try:
                out.append({
                    "report_path": report_path,
                    "label": lbl,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "support": int(float(sup))
                })
            except Exception:
                pass
        i += 1
    return out, i

def _parse_class_counts(lines: List[str], start_idx: int, report_path: str, which: str):
    """
    Parse the 'Class counts (true):' or '(predicted):' section.
    Lines look like: '  label: 123'
    """
    out = []
    i = start_idx + 1
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            break
        if ln.startswith(HDR_CLASS_PRED) or ln.startswith(HDR_CLF_REPORT) or ln.startswith(HDR_PERCLASS):
            break
        m = re.match(r"^(.*):\s*([0-9]+)$", ln)
        if m:
            out.append({
                "report_path": report_path,
                "kind": which,   # "true" or "predicted"
                "label": m.group(1).strip(),
                "count": int(m.group(2))
            })
        i += 1
    return out, i

def parse_eval_txt(fp: Path) -> Optional[Dict[str, object]]:
    """
    Parse a single *_eval.txt produced by test.py and return a dict of fields.
    Returns None if the file doesn't look like a valid report.
    """
    try:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    d: Dict[str, object] = {
        "report_path": str(fp),
        "report_stem": fp.stem,
        "basename": fp.name,
        "dir": str(fp.parent),
        "model_path": None,
        "model_kind": None,
        "model_tag": None,
        "csv_path": None,
        "rows": None,
        "pixels": None,
        "n_features": None,
        "accuracy": None,
        "misclass_rate": None,
        "kappa": None,
        "balanced_acc": None,
        "mcc": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
        "roc_auc": None,
        "eval_runtime_s": None,
        "labels": None,
        "confusion_matrix": None,  # raw block as text
    }

    for ln in lines:
        s = ln.strip()
        m = PAT_KV.match(s)
        if m:
            key, val = m.group(1), m.group(2)
            if key == "Model":
                d["model_path"] = val.strip()
            elif key == "ModelKind":
                d["model_kind"] = val.strip().lower()
            elif key == "ModelTag":
                d["model_tag"] = val.strip().lower()
            elif key == "CSV":
                d["csv_path"] = val.strip()
            elif key == "Rows":
                try: d["rows"] = int(val.strip())
                except: d["rows"] = None
            elif key == "Pixels":
                try: d["pixels"] = int(val.strip())
                except: d["pixels"] = None
            elif key == "Accuracy":
                try: d["accuracy"] = float(val.strip())
                except: d["accuracy"] = None
            elif key == "Features used":
                try: d["n_features"] = int(val.strip())
                except: d["n_features"] = None
            continue

        m2 = PAT_LABELS.match(s)
        if m2:
            d["labels"] = "|".join(_parse_labels_block(m2.group(1)))

        m = PAT_MISCLASS.match(s)
        if m: d["misclass_rate"] = float(m.group(1))
        m = PAT_KAPPA.match(s)
        if m: d["kappa"] = float(m.group(1))
        m = PAT_BALACC.match(s)
        if m: d["balanced_acc"] = float(m.group(1))
        m = PAT_MCC.match(s)
        if m: d["mcc"] = float(m.group(1))
        m = PAT_MACRO.match(s)
        if m:
            d["macro_precision"] = float(m.group(1))
            d["macro_recall"] = float(m.group(2))
            d["macro_f1"] = float(m.group(3))
        m = PAT_WEIGHTED.match(s)
        if m:
            d["weighted_precision"] = float(m.group(1))
            d["weighted_recall"] = float(m.group(2))
            d["weighted_f1"] = float(m.group(3))
        m = PAT_AUC.match(s)
        if m: d["roc_auc"] = float(m.group(1))
        m = PAT_EVAL_RUNTIME.match(s)
        if m: d["eval_runtime_s"] = float(m.group(1))

    # Second pass
    per_class_rows = []
    top_conf_rows = []
    class_count_rows = []

    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        if ln.startswith(HDR_CONFMAT):
            # Read confusion matrix block
            block, i2 = _read_block(lines, i+1)
            d["confusion_matrix"] = block if block else None
            i = i2
            continue

        if ln.startswith(HDR_TOPCONF):
            rows_tc, i2 = _parse_top_confusions(lines, i, d["report_path"])
            top_conf_rows.extend(rows_tc)
            i = i2
            continue

        if ln.startswith(HDR_PERCLASS):
            rows_pc, i2 = _parse_per_class(lines, i, d["report_path"])
            per_class_rows.extend(rows_pc)
            i = i2
            continue

        if ln.startswith(HDR_CLASS_TRUE):
            rows_cc, i2 = _parse_class_counts(lines, i, d["report_path"], "true")
            class_count_rows.extend(rows_cc)
            i = i2
            continue

        if ln.startswith(HDR_CLASS_PRED):
            rows_cc, i2 = _parse_class_counts(lines, i, d["report_path"], "predicted")
            class_count_rows.extend(rows_cc)
            i = i2
            continue

        i += 1

    # Basic sanity: must have model path and csv path at least
    if not d["model_path"] or not d["csv_path"]:
        return None

    # Derive convenience columns
    csv_stem = Path(d["csv_path"]).stem if d["csv_path"] else ""
    d["csv_stem"] = csv_stem

    # If kind/tag missing (legacy files), try to infer from filename patterns
    name = Path(d["model_path"]).name.lower() if d["model_path"] else ""
    if d["model_kind"] is None:
        d["model_kind"] = "distance" if "distance" in name else "base"
    if d["model_tag"] is None:
        for t in ("rf", "dt", "svm", "xgb"):
            if f"_{t}" in name:
                d["model_tag"] = t
                break

    # Attach side tables for caller
    d["_per_class_rows"] = per_class_rows
    d["_top_conf_rows"] = top_conf_rows
    d["_class_count_rows"] = class_count_rows
    return d

def collect_reports(inputs: List[Path], glob_pat: str) -> List[Path]:
    fps: List[Path] = []
    for p in inputs:
        if p.is_file() and p.name.endswith("_eval.txt"):
            fps.append(p)
        elif p.is_dir():
            fps.extend(list(p.rglob(glob_pat)))
    fps = sorted(set(fps))
    return fps

def main():
    ap = argparse.ArgumentParser(description="Summarise Confetti eval reports into CSVs.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more files/dirs. Dirs are searched recursively.")
    ap.add_argument("--glob", default="*_eval.txt", help="Filename glob used when searching directories (default: *_eval.txt)")
    ap.add_argument("--outdir", required=True, help="Where to write summary CSVs")
    args = ap.parse_args()

    inputs = [Path(x).expanduser() for x in args.inputs]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    files = collect_reports(inputs, args.glob)
    if not files:
        print("No *_eval.txt files found.", file=sys.stderr)
        sys.exit(2)

    main_rows = []
    per_class_all = []
    top_conf_all = []
    class_counts_all = []

    for fp in files:
        rec = parse_eval_txt(fp)
        if rec:
            # Pull out side tables
            per_class_all.extend(rec.pop("_per_class_rows", []))
            top_conf_all.extend(rec.pop("_top_conf_rows", []))
            class_counts_all.extend(rec.pop("_class_count_rows", []))
            main_rows.append(rec)

    if not main_rows:
        print("No parsable eval reports found.", file=sys.stderr)
        sys.exit(3)

    df = pd.DataFrame(main_rows)
    df["model_kind"] = df["model_kind"].astype(str)
    df["model_tag"]  = df["model_tag"].astype(str)

    # Column ordering
    cols = [
        "report_path","basename","dir",
        "model_kind","model_tag","accuracy","misclass_rate",
        "kappa","balanced_acc","mcc",
        "macro_precision","macro_recall","macro_f1",
        "weighted_precision","weighted_recall","weighted_f1",
        "roc_auc","eval_runtime_s",
        "pixels","rows","n_features",
        "labels","confusion_matrix","csv_path","csv_stem","model_path","report_stem"
    ]
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[cols]

    # Save row-level summary
    out_rows = outdir / "summary_rows.csv"
    df.to_csv(out_rows, index=False)

    # Aggregate by model kind/tag (means, std, min, max)
    agg_fields = {
        "accuracy": ["mean","std","min","max"],
        "misclass_rate": ["mean","std","min","max"],
        "kappa": ["mean","std","min","max"],
        "balanced_acc": ["mean","std","min","max"],
        "mcc": ["mean","std","min","max"],
        "macro_precision": ["mean","std","min","max"],
        "macro_recall": ["mean","std","min","max"],
        "macro_f1": ["mean","std","min","max"],
        "weighted_precision": ["mean","std","min","max"],
        "weighted_recall": ["mean","std","min","max"],
        "weighted_f1": ["mean","std","min","max"],
        "roc_auc": ["mean","std","min","max"],
        "eval_runtime_s": ["mean","std","min","max"],
        "pixels": ["sum","mean"],
        "rows": ["sum","mean"],
        "n_features": ["mean","max"]
    }
    g = df.groupby(["model_kind","model_tag"], dropna=False).agg(agg_fields)
    # Flatten MultiIndex columns
    g.columns = ["{}_{}".format(k, stat) for k, stat in g.columns]
    g = g.reset_index()
    g.insert(2, "n_reports", df.groupby(["model_kind","model_tag"], dropna=False)["report_path"].count().values)
    out_agg = outdir / "summary_by_model.csv"
    g.to_csv(out_agg, index=False)

    # Save per-class metrics (if any)
    if per_class_all:
        df_pc = pd.DataFrame(per_class_all)
        df_pc.to_csv(outdir / "summary_per_class.csv", index=False)

    # Save top confusions (if any)
    if top_conf_all:
        df_tc = pd.DataFrame(top_conf_all).sort_values(["report_path","count"], ascending=[True, False])
        df_tc.to_csv(outdir / "summary_top_confusions.csv", index=False)

    # Save class counts (if any)
    if class_counts_all:
        df_cc = pd.DataFrame(class_counts_all)
        df_cc.to_csv(outdir / "summary_class_counts.csv", index=False)

    print(f"Wrote: {out_rows}")
    print(f"Wrote: {out_agg}")
    if per_class_all: print(f"Wrote: {outdir / 'summary_per_class.csv'}")
    if top_conf_all: print(f"Wrote: {outdir / 'summary_top_confusions.csv'}")
    if class_counts_all: print(f"Wrote: {outdir / 'summary_class_counts.csv'}")

if __name__ == "__main__":
    main()
