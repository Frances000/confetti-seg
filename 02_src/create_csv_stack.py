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
  python create_csv_stack.py --root /path/to/dest --out /path/to/csv_out
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tifffile import imread, TiffFile

def _skip_key(key: str) -> bool:
    # Skip keys that start with "mask_" (case-insensitive).
    # Extend this if you want to skip other patterns.
    return key.lower().startswith("mask_")

CHANNELS = ["C", "G", "R", "Y"]
CHANNEL_LONG = {"C": "cyan", "G": "green", "R": "red", "Y": "yellow"}

# --------------------------
# Helpers
# --------------------------
def _mask_index(labels: list[str]) -> int | None:
    """Return index of 'mask' slice if present, else None."""
    lower = [l.lower() for l in labels]
    return lower.index("mask") if "mask" in lower else None


# def _nearest_allele_count(u8: np.ndarray) -> np.ndarray:
#     """
#     Map uint8 mask plane to {0,1,2} by nearest of {0,127,255}. Treat 254 as 255.
#     """
#     u = u8.astype(np.uint8).copy()
#     u[u == 254] = 255
#     ref = np.array([0, 127, 255], dtype=np.float32).reshape(3, 1, 1)
#     diffs = np.abs(u.astype(np.float32)[None, ...] - ref)
#     idx = np.argmin(diffs, axis=0)  # 0,1,2
#     return idx.astype(np.uint8)

def _nearest_allele_count(u8_2d: np.ndarray) -> np.ndarray:
    """
    Expect a 2-D uint8 plane; map intensities to {0,1,2} by nearest of {0,127,255}.
    Treat 254 as 255.
    """
    u = u8_2d.astype(np.uint8, copy=True)
    u[u == 254] = 255
    ref = np.array([0, 127, 255], dtype=np.float32)[:, None, None]  # (3,1,1)
    diffs = np.abs(u.astype(np.float32)[None, ...] - ref)           # (3,H,W)
    return diffs.argmin(axis=0).astype(np.uint8)                     # (H,W)

def _load_stack_and_labels(stack_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Read stack and labels, returning (Z,H,W) and a Z-length label list.
    """
    with TiffFile(str(stack_path)) as tf:
        stack_raw = tf.series[0].asarray()
    stack = _to_ZHW(stack_raw)  # (Z,H,W)
    Z = stack.shape[0]

    lab_path = stack_path.with_name(stack_path.stem + "_labels.txt")
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

#Distance map stem
DISTANCE_RE = re.compile(r"^Distance_(?P<stem>.+)_Feature-stack\d+\.tif$", re.IGNORECASE)
STEM_RE = re.compile(r"^(?P<key>.+)_(?P<ch>[CGRY])$")

def _discover_patch_groups_base(root: Path) -> Dict[str, Dict[str, Path]]:
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

def _to_ZHW(arr: np.ndarray) -> np.ndarray:
    """
    Coerce arbitrary TIFF array to (Z,H,W):
      - squeeze singleton axes
      - if 2D, treat as single-slice stack (1,H,W)
      - if 3D and one axis looks like channels/slices (<=32), put it as Z
      - if 3D and looks like (H,W,Z), move Z to front
    """
    a = np.squeeze(arr)
    if a.ndim == 2:
        return a[None, ...]  # (1,H,W)
    if a.ndim == 3:
        # Heuristics: pick the axis that is NOT clearly spatial.
        H, W = sorted(a.shape)[-2:]  # largest two are likely spatial
        # If last axis is small (channels/features), move it to front
        if a.shape[-1] <= 32 and a.shape[-2] == H and a.shape[-3] == W:
            return np.moveaxis(a, -1, 0)  # (Z,H,W)
        # If first axis is small, assume already (Z,H,W)
        if a.shape[0] <= 512 and a.shape[1] == H and a.shape[2] == W:
            return a  # (Z,H,W)
        # If looks like (H,W,Z), move Z to front
        if a.shape[0] == H and a.shape[1] == W:
            return np.moveaxis(a, -1, 0)
        # Fallback: pick the smallest axis as Z
        z_axis = int(np.argmin(a.shape))
        return np.moveaxis(a, z_axis, 0)
    raise ValueError(f"Unsupported TIFF array with ndim={a.ndim} and shape {a.shape}")

def _discover_patch_groups_distance(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan ROOT/{C,G,R,Y} for Base_*.tif and group by patch key.
    Returns: { key: { 'C': path_to_stack, ... } }
    """
    groups: Dict[str, Dict[str, Path]] = {}
    for ch in CHANNELS:
        d = root / ch
        if not d.is_dir():
            continue
        for p in d.glob("Distance_*_Feature-stack*.tif"):
            m = DISTANCE_RE.match(p.name)
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
    # Harmonise spatial size from available masks (use last two dims)
    H = min(np.squeeze(m).shape[-2] for m in masks_by_ch.values())
    W = min(np.squeeze(m).shape[-1] for m in masks_by_ch.values())
    N = H * W
    cls = np.array([""] * N, dtype=object)

    for ch in CHANNELS:
        if ch not in masks_by_ch:
            continue
        u = np.squeeze(masks_by_ch[ch])[..., :H, :W]  # crop on last axes
        # If still (K,H,W), reduce to a single plane (max is robust for masks)
        if u.ndim == 3:
            u = u.max(axis=0)
        if u.ndim != 2:
            raise ValueError(f"Mask for channel {ch} is not 2-D after squeeze; got {u.shape}")
        counts = _nearest_allele_count(u)
        add = np.where(counts == 0, "", np.where(counts == 1, ch, ch + ch)).reshape(-1)
        cls = np.char.add(cls, add.astype(object))

    def _norm(s: str) -> str:
        if len(s) < 2:
            s = s + "B" * (2 - len(s))
        s = "".join(sorted(s))
        return s[:2]
    return np.vectorize(_norm, otypes=[object])(cls)


# --------------------------
# Main composer
# --------------------------
def compose_csvs_base(root: Path, out_dir: Path, *, mode: str = "infer") -> None:
    groups = _discover_patch_groups_base(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not groups:
        print("No Base_* feature stacks discovered.")
        return

    for key, per_ch in groups.items():
        if _skip_key(key):
            print(f"[SKIP] {key}: suppressed (mask_* key)")
            continue
        stacks: dict[str, np.ndarray] = {}
        labels: dict[str, list[str]] = {}
        masks: dict[str, np.ndarray] = {}
        missing_any_mask = False

        # Load stacks & labels; collect masks ONLY if explicitly present
        for ch, spath in per_ch.items():
            stk, lab = _load_stack_and_labels(spath)  # (Z,H,W), labels[Z]
            stacks[ch], labels[ch] = stk, lab
            mi = _mask_index(lab)
            if mi is None:
                # heuristic: treat slice 0 as mask if it’s binary-looking
                u = stk[0]
                if np.unique(u).size <= 3:  # e.g., {0,127,255}
                    mi = 0
            masks[ch] = stk[mi]

        if mode == "train" and (missing_any_mask or len(masks) != len(per_ch)):
            raise RuntimeError(f"[{key}] Missing 'mask' slice in at least one channel (train mode).")

        # Build class ONLY if all masks present
        class_vec = None
        if not missing_any_mask and len(masks) == len(per_ch) and len(masks) > 0:
            class_vec = _compose_class_from_masks(masks)

        # Harmonise H,W; flatten features (skip 'mask' slice)
        H = min(stacks[ch].shape[1] for ch in stacks)
        W = min(stacks[ch].shape[2] for ch in stacks)
        cols: dict[str, np.ndarray] = {}

        if class_vec is not None:
            cols["class"] = class_vec

        order = ["C", "G", "R", "Y"]
        for ch in order:
            if ch not in stacks:
                continue
            stk = stacks[ch][:, :H, :W]
            labs = labels[ch]
            idxs = [i for i, l in enumerate(labs) if l.lower() != "mask"]
            for i in idxs:
                colname = _colname_for(labs[i], ch)
                if colname is None:
                    continue
                cols[colname] = stk[i].reshape(-1)

        df = pd.DataFrame(cols)
        out_path = out_dir / f"{key}.csv"
        df.to_csv(out_path, index=False)
        flag = "" if "class" in df.columns else " (no class; infer mode)"
        print(f"[OK] {key}: {df.shape[0]} rows × {df.shape[1]} cols -> {out_path}{flag}")


def compose_csvs_distance(root: Path, out_dir: Path, *, mode: str = "infer") -> None:
    groups = _discover_patch_groups_distance(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not groups:
        print("No Distance_* feature stacks discovered.")
        return

    for key, per_ch in groups.items():
        if _skip_key(key):
            print(f"[SKIP] {key}: suppressed (mask_* key)")
            continue
        stacks: dict[str, np.ndarray] = {}
        labels: dict[str, list[str]] = {}
        masks: dict[str, np.ndarray] = {}
        missing_any_mask = False

        for ch, spath in per_ch.items():
            stk, lab = _load_stack_and_labels(spath)
            stacks[ch], labels[ch] = stk, lab
            mi = _mask_index(lab)
            if mi is None:
                # heuristic: treat slice 0 as mask if it’s binary-looking
                u = stk[0]
                if np.unique(u).size <= 3:  # e.g., {0,127,255}
                    mi = 0
            masks[ch] = stk[mi]

        if mode == "train" and (missing_any_mask or len(masks) != len(per_ch)):
            raise RuntimeError(f"[{key}] Missing 'mask' slice in at least one channel (train mode).")

        class_vec = None
        if not missing_any_mask and len(masks) == len(per_ch) and len(masks) > 0:
            class_vec = _compose_class_from_masks(masks)

        H = min(stacks[ch].shape[1] for ch in stacks)
        W = min(stacks[ch].shape[2] for ch in stacks)
        cols: dict[str, np.ndarray] = {}

        if class_vec is not None:
            cols["class"] = class_vec

        order = ["C", "G", "R", "Y"]
        for ch in order:
            if ch not in stacks:
                continue
            stk = stacks[ch][:, :H, :W]
            labs = labels[ch]
            idxs = [i for i, l in enumerate(labs) if l.lower() != "mask"]
            for i in idxs:
                colname = _colname_for(labs[i], ch)
                if colname is None:
                    continue
                cols[colname] = stk[i].reshape(-1)

        df = pd.DataFrame(cols)
        out_path = out_dir / f"distance_{key}.csv"
        df.to_csv(out_path, index=False)
        flag = "" if "class" in df.columns else " (no class; infer mode)"
        print(f"[OK] {key}: {df.shape[0]} rows × {df.shape[1]} cols -> {out_path}{flag}")

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Compose per-patch CSVs from existing Base feature stacks (C,G,R,Y).")
    ap.add_argument("--root", required=True, help="Root containing channel subfolders with Base_*.tif and *_labels.txt")
    ap.add_argument("--out", required=True, help="Destination folder for CSVs")
    ap.add_argument("--mode", choices=["infer", "train"], default="train",
                help="infer: allow missing mask(s) and omit 'class'; train: require masks for class.")
    args = ap.parse_args()
    compose_csvs_base(Path(args.root), Path(args.out), mode=args.mode)
    compose_csvs_distance(Path(args.root), Path(args.out), mode=args.mode)

if __name__ == "__main__":
    main()
