#!/usr/bin/env python3
"""
Compose per-patch flattened CSVs from existing Base feature stacks.

Columns:
  [ class,
    <all non-mask slices from C (with _cyan suffix)>,
    <all non-mask slices from G (with _green suffix)>,
    <all non-mask slices from R (with _red suffix)>,
    <all non-mask slices from Y (with _yellow suffix)> ]

'class' construction:
  per pixel, read each channel's mask plane (first slice, or label 'mask');
  value 0 -> add nothing; 127 -> add one letter; 255/254 -> add two letters.
  After C,G,R,Y processed: if len<2, pad with 'B'; then alphabetise (e.g. 'YR'->'RY').

Usage:
  python compose_csv_from_stacks.py --root /path/to/dest --out /path/to/csv_out
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tifffile import imread

CHANNELS = ["C", "G", "R", "Y"]
CHANNEL_LONG = {"C": "cyan", "G": "green", "R": "red", "Y": "yellow"}

# --------------------------
# Helpers
# --------------------------
def _nearest_allele_count(u8: np.ndarray) -> np.ndarray:
    """
    Map uint8 mask plane to {0,1,2} by nearest of {0,127,255}. Treat 254 as 255.
    """
    u = u8.astype(np.uint8).copy()
    u[u == 254] = 255
    ref = np.array([0, 127, 255], dtype=np.float32).reshape(3, 1, 1)
    diffs = np.abs(u.astype(np.float32)[None, ...] - ref)
    idx = np.argmin(diffs, axis=0)  # 0,1,2
    return idx.astype(np.uint8)

def _load_stack_and_labels(stack_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Read (Z,H,W) stack and optional labels file (same stem + '_labels.txt').
    If labels missing/mismatch, synthesise labels 'slice_01..'.
    """
    stack = imread(stack_path)  # (Z,H,W)
    Z = stack.shape[0]
    lab_path = stack_path.with_name(stack_path.stem + "_labels.txt")
    labels: List[str]
    if lab_path.exists():
        with open(lab_path, "r") as f:
            labels = [ln.strip().split(": ", 1)[-1] for ln in f if ln.strip()]
        if len(labels) != Z:
            labels = [f"slice_{i+1:02d}" for i in range(Z)]
    else:
        labels = [f"slice_{i+1:02d}" for i in range(Z)]
    return stack, labels

# Base_<stem>_Feature-stack0001.tif  -> stem == "<patch_key>_<CH>"
BASE_RE = re.compile(r"^Base_(?P<stem>.+)_Feature-stack\d+\.tif$", re.IGNORECASE)
STEM_RE = re.compile(r"^(?P<key>.+)_(?P<ch>[CGRY])$")

def _discover_patch_groups(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan ROOT/{C,G,R,Y} for Base_*.tif and group by patch key.
    Returns: { key: { 'C': path_to_stack, ... } }
    """
    groups: Dict[str, Dict[str, Path]] = {}
    for ch in CHANNELS:
        d = root / ch
        if not d.is_dir():
            continue
        for p in d.glob("Base_*_Feature-stack*.tif"):
            m = BASE_RE.match(p.name)
            if not m:
                continue
            stem = m.group("stem")  # e.g., 'filiform_patch_00_C'
            m2 = STEM_RE.match(stem)
            if not m2:
                # if can't split, treat whole as key and trust folder for channel
                key = stem
                ch_tag = ch
            else:
                key = m2.group("key")
                ch_tag = m2.group("ch")
            if ch_tag != ch:
                ch_tag = ch  # trust folder
            groups.setdefault(key, {})[ch_tag] = p
    return groups

def _colname_for(label: str, ch: str) -> Optional[str]:
    """
    Make a per-slice column name. Skip 'mask'; rename 'original' nicely; for others,
    strip leading '<CH>_' if present, then suffix with channel long name.
    """
    if label.lower() == "mask":
        return None
    long = CHANNEL_LONG[ch]
    if label.lower() == "original":
        return f"original_{long}"
    if label.startswith(f"{ch}_"):
        base = label[len(ch) + 1:]
        return f"{base}_{long}"
    return f"{label}_{long}"

def _compose_class_from_masks(masks_by_ch: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised class string assembly from per-channel mask planes."""
    # Determine common H,W (crop to min across channels)
    H = min(m.shape[0] for m in masks_by_ch.values())
    W = min(m.shape[1] for m in masks_by_ch.values())
    N = H * W
    cls = np.array([""] * N, dtype=object)

    for ch in CHANNELS:
        if ch not in masks_by_ch:
            continue
        counts = _nearest_allele_count(masks_by_ch[ch][:H, :W])
        # vectorised expansion: "", "G", "GG" etc.
        add = np.where(counts == 0, "", np.where(counts == 1, ch, ch + ch)).reshape(-1)
        cls = np.char.add(cls, add.astype(object))

    # normalise to exactly 2 letters: pad with 'B', alphabetise, truncate if >2
    def _norm(s: str) -> str:
        if len(s) < 2:
            s = s + "B" * (2 - len(s))
        s = "".join(sorted(s))
        return s[:2]
    return np.vectorize(_norm, otypes=[object])(cls)

# --------------------------
# Main composer
# --------------------------
def compose_csvs(root: Path, out_dir: Path) -> None:
    """
    For each patch key found under ROOT/{C,G,R,Y}, build one CSV in out_dir.
    """
    groups = _discover_patch_groups(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not groups:
        print("No Base_* feature stacks discovered.")
        return

    for key, per_ch in groups.items():
        stacks: Dict[str, np.ndarray] = {}
        labels: Dict[str, List[str]] = {}
        masks: Dict[str, np.ndarray] = {}

        # Load stacks & labels; extract mask plane
        for ch, spath in per_ch.items():
            stk, lab = _load_stack_and_labels(spath)  # (Z,H,W), labels[Z]
            stacks[ch] = stk
            labels[ch] = lab
            # mask plane: prefer explicit 'mask' label, else assume first slice
            if "mask" in [l.lower() for l in lab]:
                idx = [l.lower() for l in lab].index("mask")
            else:
                idx = 0
            masks[ch] = stk[idx]

        # Build class vector (N,)
        class_vec = _compose_class_from_masks(masks)

        # Harmonise H,W and build per-channel feature matrices (skip mask)
        H = min(stacks[ch].shape[1] for ch in stacks)
        W = min(stacks[ch].shape[2] for ch in stacks)
        N = H * W

        cols = {"class": class_vec}
        order = ["C", "G", "R", "Y"]
        for ch in order:
            if ch not in stacks:
                continue
            stk = stacks[ch][:, :H, :W]  # (Z,H,W)
            labs = labels[ch]
            # find slice indices to include (non-mask)
            idxs = [i for i, l in enumerate(labs) if l.lower() != "mask"]
            for i in idxs:
                colname = _colname_for(labs[i], ch)
                if colname is None:
                    continue
                cols[colname] = stk[i].reshape(-1)

        df = pd.DataFrame(cols)
        out_path = out_dir / f"{key}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] {key}: {df.shape[0]} rows × {df.shape[1]} cols -> {out_path}")

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Compose per-patch CSVs from existing Base feature stacks (C,G,R,Y).")
    ap.add_argument("--root", required=True, help="Root containing channel subfolders with Base_*.tif and *_labels.txt")
    ap.add_argument("--out", required=True, help="Destination folder for CSVs")
    args = ap.parse_args()
    compose_csvs(Path(args.root), Path(args.out))

if __name__ == "__main__":
    main()
