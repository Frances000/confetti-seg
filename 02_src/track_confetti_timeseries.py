#!/usr/bin/env python3
"""
Track Confetti clones across a time series, visualize them (overlays + growth curves).

Usage examples
--------------
# 1) Track only
python track_and_visualize_confetti.py \
  --labels t0_labels.tif t1_labels.tif t2_labels.tif \
  --out_dir /path/to/ts_output

# 2) Visualize only (re-using existing tracks.csv)
python track_and_visualize_confetti.py \
  --viz_only \
  --labels t0_labels.tif t1_labels.tif t2_labels.tif \
  --tracks_csv /path/to/ts_output/tracks.csv \
  --out_dir /path/to/ts_output

# 3) Track + visualize in one go (default)
python track_and_visualize_confetti.py \
  --labels t0_labels.tif t1_labels.tif t2_labels.tif \
  --out_dir /path/to/ts_output

"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as cc_label, regionprops
from skimage.registration import phase_cross_correlation, optical_flow_tvl1

# ----------------------
# I/O & utilities
# ----------------------
def read_u8_or_u16(p: Path) -> np.ndarray:
    arr = imread(str(p))
    if arr.ndim != 2:
        raise ValueError(f"{p.name}: expected 2D image, got shape {arr.shape}")
    return arr

def as_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    m = float(arr.max())
    return arr / (m if m > 0 else 1.0)

# ----------------------
# Registration
# ----------------------
def estimate_translation(I0: np.ndarray, I1: np.ndarray) -> Tuple[float, float]:
    """Phase correlation (translation only), returns (shift_row, shift_col)."""
    shift, _, _ = phase_cross_correlation(I0, I1, upsample_factor=10)
    return float(shift[0]), float(shift[1])  # dr, dc

def warp_by_translation(arr: np.ndarray, dr: float, dc: float, order: int = 0) -> np.ndarray:
    H, W = arr.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([rr - dr, cc - dc])
    return map_coordinates(arr, coords, order=order, mode="nearest")

def estimate_flow(I0: np.ndarray, I1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """TV-L1 optical flow on float01 images; returns (vy, vx)."""
    v, u = optical_flow_tvl1(I0, I1)  # returns (vy, vx)
    return v.astype(np.float32), u.astype(np.float32)

def warp_by_flow(arr: np.ndarray, vy: np.ndarray, vx: np.ndarray, order: int = 0) -> np.ndarray:
    H, W = arr.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([rr - vy, cc - vx])
    return map_coordinates(arr, coords, order=order, mode="nearest")

# ----------------------
# Clone extraction & association
# ----------------------
def extract_clones(label_img: np.ndarray) -> List[Dict]:
    """Connected components per class index. Returns list of dicts with props + mask."""
    clones: List[Dict] = []
    classes = sorted(int(c) for c in np.unique(label_img) if c >= 0)
    idx = 1
    for c in classes:
        mask_c = (label_img == c)
        lab = cc_label(mask_c, connectivity=2)
        for rp in regionprops(lab):
            cmask = (lab == rp.label)
            area = int(rp.area)
            if area == 0:
                continue
            clones.append({
                "clone_id": idx,            # local id within a timepoint
                "class_index": int(c),
                "area_px": area,
                "centroid_r": float(rp.centroid[0]),
                "centroid_c": float(rp.centroid[1]),
                "mask": cmask,
            })
            idx += 1
    return clones


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(max(union, 1))


def associate_clones(clones_a: List[Dict], clones_b: List[Dict], iou_thresh=0.2) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    """Hungarian on cost=1-IoU, class-gated. Returns matches and unmatched ids."""
    if not clones_a or not clones_b:
        return [], [c["clone_id"] for c in clones_a], [c["clone_id"] for c in clones_b]

    A, B = len(clones_a), len(clones_b)
    M = np.zeros((A, B), dtype=np.float32)
    for i, ca in enumerate(clones_a):
        for j, cb in enumerate(clones_b):
            if ca["class_index"] != cb["class_index"]:
                continue
            M[i, j] = iou(ca["mask"], cb["mask"])  # 0..1

    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, ua, ub = [], set(range(A)), set(range(B))
    for i, j in zip(row_ind, col_ind):
        if M[i, j] >= iou_thresh:
            matches.append((clones_a[i]["clone_id"], clones_b[j]["clone_id"], float(M[i, j])))
            ua.discard(i); ub.discard(j)
    unmatched_a = [clones_a[i]["clone_id"] for i in sorted(list(ua))]
    unmatched_b = [clones_b[j]["clone_id"] for j in sorted(list(ub))]
    return matches, unmatched_a, unmatched_b

# ----------------------
# Tracking core
# ----------------------
def track_time_series(label_paths: List[Path],
                      intensity_paths: Optional[List[Path]] = None,
                      iou_thresh: float = 0.2,
                      use_flow: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (tracks_df, links_df).
    tracks_df: track_id, t, clone_id, class_index, area_px, centroid_r, centroid_c
    links_df:  t, clone_id_t, t1, clone_id_t1, iou, split_merge
    """
    T = len(label_paths)
    assert T >= 2, "Need at least 2 timepoints."
    labels = [read_u8_or_u16(p) for p in label_paths]

    if intensity_paths:
        intens = [as_float01(read_u8_or_u16(p)) for p in intensity_paths]
        intens = [gaussian_filter(I, sigma=0.5) for I in intens]
    else:
        intens = None

    # init tracks at t=0
    clones_t = extract_clones(labels[0])
    track_id_next = 1
    for c in clones_t:
        c["track_id"] = track_id_next
        track_id_next += 1

    tracks_rows = [{
        "track_id": c["track_id"], "t": 0,
        "clone_id": c["clone_id"], "class_index": c["class_index"],
        "area_px": c["area_px"], "centroid_r": c["centroid_r"], "centroid_c": c["centroid_c"]
    } for c in clones_t]

    links_rows: List[Dict] = []

    for t in range(T - 1):
        L0, L1 = labels[t], labels[t+1]
        if intens is not None:
            I0, I1 = intens[t], intens[t+1]
        else:
            I0, I1 = as_float01(L0), as_float01(L1)

        # rigid shift
        dr, dc = estimate_translation(I0, I1)
        L0w = warp_by_translation(L0, dr, dc, order=0)

        # nonrigid refinement
        if use_flow:
            vy, vx = estimate_flow(I0, I1)
            L0w = warp_by_flow(L0w, vy, vx, order=0)

        clones_w = extract_clones(L0w.astype(L0.dtype))
        clones_1 = extract_clones(L1)

        # carry track ids from previous set to warped set via nearest-centroid
        from scipy.spatial import cKDTree
        if clones_t:
            XY_prev = np.array([[c["centroid_r"], c["centroid_c"]] for c in clones_t], dtype=np.float32)
            tid_prev = np.array([c["track_id"] for c in clones_t], dtype=np.int32)
            tree = cKDTree(XY_prev)
            for cw in clones_w:
                d, k = tree.query([cw["centroid_r"], cw["centroid_c"]], k=1)
                cw["track_id"] = int(tid_prev[k])
        else:
            for cw in clones_w:
                cw["track_id"] = None

        # associate warped(t) -> (t+1)
        matches, ua, ub = associate_clones(clones_w, clones_1, iou_thresh=iou_thresh)

        # matched: inherit track id
        for id_w, id_1, ov in matches:
            trk = next(c["track_id"] for c in clones_w if c["clone_id"] == id_w)
            c1 = next(c for c in clones_1 if c["clone_id"] == id_1)
            c1["track_id"] = trk
            links_rows.append({"t": t, "clone_id_t": id_w, "t1": t+1, "clone_id_t1": id_1, "iou": ov, "split_merge": ""})

        # new clones at t+1
        for id_1 in ub:
            c1 = next(c for c in clones_1 if c["clone_id"] == id_1)
            c1["track_id"] = track_id_next
            track_id_next += 1
            links_rows.append({"t": t, "clone_id_t": -1, "t1": t+1, "clone_id_t1": id_1, "iou": 0.0, "split_merge": "new"})

        # commit rows for t+1
        for c in clones_1:
            tracks_rows.append({
                "track_id": c["track_id"], "t": t+1,
                "clone_id": c["clone_id"], "class_index": c["class_index"],
                "area_px": c["area_px"], "centroid_r": c["centroid_r"], "centroid_c": c["centroid_c"]
            })

        clones_t = clones_1

    return pd.DataFrame(tracks_rows), pd.DataFrame(links_rows)

# ----------------------
# Visualization
# ----------------------
def _color_table(track_ids) -> Dict[int, Tuple[int,int,int]]:
    rng = np.random.default_rng(12345)
    uniq = sorted(set(int(t) for t in track_ids))
    lut = {}
    for tid in uniq:
        rgb = (rng.random(3) * 205 + 50).astype(np.uint8)  # avoid too dark
        lut[int(tid)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return lut


def _draw_overlay(bg_u8: np.ndarray, label_img: np.ndarray, rows: pd.DataFrame, t: int,
                   lut: Dict[int, Tuple[int,int,int]], out_path: Path, max_history: int = 3) -> None:
    H, W = bg_u8.shape
    rgb = np.stack([bg_u8, bg_u8, bg_u8], axis=-1)

    rows_t = rows[rows.t == t]

    # Draw label boundaries in white (fast): any 4-neighbour transition
    bnd = (
        (label_img != np.pad(label_img, ((0,0),(1,0)), mode='edge')[:, :-1]) |
        (label_img != np.pad(label_img, ((0,0),(0,1)), mode='edge')[:, 1:]) |
        (label_img != np.pad(label_img, ((1,0),(0,0)), mode='edge')[:-1, :]) |
        (label_img != np.pad(label_img, ((0,1),(0,0)), mode='edge')[1:, :])
    )
    rgb[bnd] = (255, 255, 255)

    # Small centroid dots
    for _, r in rows_t.iterrows():
        rr, cc = int(round(r.centroid_r)), int(round(r.centroid_c))
        col = lut.get(int(r.track_id), (255, 255, 0))
        rr0, rr1 = max(rr-1,0), min(rr+2,H)
        cc0, cc1 = max(cc-1,0), min(cc+2,W)
        rgb[rr0:rr1, cc0:cc1] = col

    # Trails (last max_history frames)
    for tid in rows.track_id.unique():
        pts = rows[(rows.track_id == tid) & (rows.t <= t) & (rows.t > t - max_history)][["centroid_r", "centroid_c", "t"]].values
        if len(pts) < 2:
            continue
        col = lut.get(int(tid), (255, 255, 0))
        for i in range(len(pts) - 1):
            r0, c0, _ = pts[i]
            r1, c1, _ = pts[i+1]
            r0 = int(round(r0)); c0 = int(round(c0)); r1 = int(round(r1)); c1 = int(round(c1))
            # Bresenham line
            dr = abs(r1 - r0); dc = abs(c1 - c0)
            sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
            err = dr - dc
            r, c = r0, c0
            while True:
                if 0 <= r < H and 0 <= c < W:
                    rgb[r, c] = col
                if r == r1 and c == c1:
                    break
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc; r += sr
                if e2 < dr:
                    err += dr; c += sc

    imwrite(str(out_path), rgb.astype(np.uint8))


def visualize(labels: List[Path], intensity: Optional[List[Path]], tracks_csv: Path, out_dir: Path) -> None:
    lab_imgs = [imread(str(p)) for p in labels]
    if intensity and len(intensity) == len(labels):
        backs = [(as_float01(imread(str(p))) * 255).astype(np.uint8) for p in intensity]
    else:
        backs = [(as_float01(L) * 255).astype(np.uint8) for L in lab_imgs]

    tracks = pd.read_csv(tracks_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    lut = _color_table(tracks.track_id.unique())
    for t, (L, B) in enumerate(zip(lab_imgs, backs)):
        _draw_overlay(B, L, tracks, t, lut, out_dir / "overlays" / f"overlay_t{t:02d}.tif")

    # Growth curves for top-N tracks by total area
    N = 8
    agg = tracks.groupby(["track_id", "t"]).area_px.sum().reset_index()
    top = agg.groupby("track_id").area_px.sum().sort_values(ascending=False).head(N).index
    plt.figure()
    for tid in top:
        df = agg[agg.track_id == tid]
        plt.plot(df.t, df.area_px, marker="o", label=f"track {tid}")
    plt.xlabel("time")
    plt.ylabel("area (px)")
    plt.title("Top tracks: area over time")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "growth_curves.png", dpi=180)
    plt.close()

# ----------------------
# CLI
# ----------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track Confetti clones across time and visualize overlays/growth.")
    ap.add_argument("--labels", nargs="+", required=True, help="Ordered list of *_labels.tif for t=0..T-1.")
    ap.add_argument("--intensity", nargs="*", default=None, help="Optional ordered list of intensity images for registration/overlay.")
    ap.add_argument("--iou_thresh", type=float, default=0.20, help="IoU threshold for clone matching.")
    ap.add_argument("--no_flow", action="store_true", help="Disable TV-L1 optical-flow refinement.")
    ap.add_argument("--out_dir", required=True, help="Output directory (tracks.csv, links.csv, overlays/, growth_curves.png)")
    ap.add_argument("--viz_only", action="store_true", help="Skip tracking and only visualize from --tracks_csv.")
    ap.add_argument("--track_only", action="store_true", help="Run tracking only (no visualization).")
    ap.add_argument("--tracks_csv", default="", help="Path to existing tracks.csv (for --viz_only or to override).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    label_paths = [Path(p) for p in args.labels]
    intensity_paths = [Path(p) for p in args.intensity] if args.intensity else None
    out_dir = Path(args.out_dir)

    if intensity_paths and len(intensity_paths) != len(label_paths):
        raise ValueError("If provided, --intensity must have same length as --labels.")

    tracks_csv_path: Path

    if not args.viz_only:
        # Track
        tracks_df, links_df = track_time_series(label_paths, intensity_paths, iou_thresh=args.iou_thresh, use_flow=(not args.no_flow))
        out_dir.mkdir(parents=True, exist_ok=True)
        tracks_csv_path = out_dir / "tracks.csv"
        links_csv_path = out_dir / "links.csv"
        tracks_df.to_csv(tracks_csv_path, index=False)
        links_df.to_csv(links_csv_path, index=False)
        print(f"[track] wrote {tracks_csv_path} and {links_csv_path}")
    else:
        if not args.tracks_csv:
            raise ValueError("--viz_only requires --tracks_csv")
        tracks_csv_path = Path(args.tracks_csv)

    if not args.track_only:
        visualize(label_paths, intensity_paths, tracks_csv_path, out_dir)
        print(f"[viz] wrote overlays/ and growth_curves.png under {out_dir}")


if __name__ == "__main__":
    main()
