#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import measure, morphology, segmentation, feature, filters, exposure, util
from skimage.measure import regionprops_table, label as cc_label, perimeter as sk_perimeter
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation, disk, skeletonize
from skimage.draw import polygon_perimeter
from skimage.color import label2rgb
from skimage.io import imread, imsave

# -----------------------
# Utilities
# -----------------------
def _ensure_uint(img):
    if img.dtype.kind in "iu":
        return img
    return img.astype(np.uint16 if img.max() >= 256 else np.uint8)

def _boxcount_fd(binary_mask: np.ndarray, min_box=2, max_box=None, n_scales=8) -> float:
    """
    Box-counting fractal dimension on a binary (True=object) mask.
    Returns slope of log(N(eps)) vs log(1/eps).
    """
    mask = binary_mask.astype(bool)
    if max_box is None:
        max_box = min(mask.shape) // 2
    sizes = np.geomspace(min_box, max(4, max_box), num=n_scales).astype(int)
    sizes = np.unique(sizes)
    Ns, inv_eps = [], []
    for s in sizes:
        # pad to multiple of s
        H, W = mask.shape
        Hp = int(np.ceil(H / s) * s)
        Wp = int(np.ceil(W / s) * s)
        pad = np.zeros((Hp, Wp), dtype=bool)
        pad[:H, :W] = mask
        # reshape into s×s tiles and count boxes that intersect the object
        tiles = pad.reshape(Hp//s, s, Wp//s, s).any(axis=(1,3))
        Ns.append(tiles.sum())
        inv_eps.append(Hp / s)  # proportional to 1/ε
    # linear fit in log–log
    x = np.log(np.asarray(inv_eps, dtype=float) + 1e-12)
    y = np.log(np.asarray(Ns, dtype=float) + 1e-12)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)

def _protrusion_metrics(mask: np.ndarray, core_radius: int = 5) -> dict:
    """
    Estimate protrusions by subtracting a morphological 'core'
    and summarising skeleton branch lengths in the residual.
    """
    # Core via opening (erode->dilate): removes thin tips
    se = disk(core_radius)
    core = morphology.opening(mask, se)
    tips = np.logical_and(mask, np.logical_not(core))
    # Clean tiny specks
    tips = morphology.remove_small_objects(tips, min_size=3)
    # Skeletonize tips and measure branch lengths
    sk = skeletonize(tips)
    # Label skeleton branch endpoints
    # endpoints: pixels with exactly 1 neighbor in 8-neighborhood
    from scipy.ndimage import convolve
    K = np.ones((3,3), dtype=int); K[1,1] = 0
    deg = convolve(sk.astype(int), K, mode="constant", cval=0)
    endpoints = np.logical_and(sk, deg==1)

    # Connected components on tip blobs, then geodesic lengths on skeleton
    lab = cc_label(tips, connectivity=2)
    props = measure.regionprops(lab)
    lengths = []
    for p in props:
        sub = lab == p.label
        sk_sub = skeletonize(sub)
        if sk_sub.sum() < 2:
            continue
        # approximate length by geodesic distance between farthest endpoints
        ep = np.argwhere(np.logical_and(sk_sub, 
                 convolve(sk_sub.astype(int), K, mode="constant", cval=0)==1))
        if len(ep) >= 2:
            from scipy.spatial.distance import cdist
            D = cdist(ep, ep, metric='euclidean')
            lengths.append(float(D.max()))
        else:
            # fallback: pixel count along skeleton
            lengths.append(float(sk_sub.sum()))
    tip_count = len(lengths)
    return {
        "protrusion_count": tip_count,
        "protrusion_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "protrusion_len_max": float(np.max(lengths)) if lengths else 0.0,
        "protrusion_len_sum": float(np.sum(lengths)) if lengths else 0.0,
    }

def _safe_circularity(area: float, perim: float) -> float:
    return float((4.0 * math.pi * area) / (perim ** 2 + 1e-12))

# -----------------------
# Main analysis
# -----------------------
def analyze(label_img: np.ndarray,
            legend_df: pd.DataFrame | None,
            min_pixels: int = 20,
            pix_size_um: float | None = None,
            core_radius: int = 5):
    """
    label_img: 2D array of integer class indices (0..K-1).
    legend_df: optional legend with columns ['index','label','R','G','B'].
    """
    H, W = label_img.shape
    rows = []
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # Loop classes (exclude background index 0 if that’s your convention)
    classes = sorted(np.unique(label_img))
    for cls in classes:
        if cls < 0: 
            continue
        cls_mask = (label_img == cls)

        # Connected clones within this class
        lab = cc_label(cls_mask, connectivity=2)
        lab = remove_small_objects(lab, min_size=min_pixels)
        # Per-class colour for overlay (from legend if available)
        rgb = (0, 255, 255)
        name = str(cls)
        if legend_df is not None and "index" in legend_df.columns:
            row = legend_df.loc[legend_df["index"] == cls]
            if len(row) == 1:
                r, g, b = int(row["R"].values[0]), int(row["G"].values[0]), int(row["B"].values[0])
                rgb = (r, g, b)
                if "label" in row.columns:
                    name = str(row["label"].values[0])

        props = measure.regionprops(lab)
        for p in props:
            clone_mask = (lab == p.label)
            area_px = float(p.area)
            if area_px < min_pixels:
                continue
            perim_px = float(sk_perimeter(clone_mask, neighborhood=8))
            circ = _safe_circularity(area_px, perim_px)
            solidity = float(p.solidity) if hasattr(p, "solidity") else float(area_px / max(1, p.convex_area))
            elong = float((p.major_axis_length + 1e-12) / (p.minor_axis_length + 1e-12))
            fd = _boxcount_fd(clone_mask)
            protr = _protrusion_metrics(clone_mask, core_radius=core_radius)

            # Optional µm scaling
            area_um2 = perim_um = maj_um = min_um = eqdiam_um = None
            eq_diam = math.sqrt(4.0 * area_px / math.pi)
            if pix_size_um:
                area_um2 = area_px * (pix_size_um ** 2)
                perim_um = perim_px * pix_size_um
                maj_um = p.major_axis_length * pix_size_um
                min_um = p.minor_axis_length * pix_size_um
                eqdiam_um = eq_diam * pix_size_um

            rows.append({
                "class_index": int(cls),
                "class_label": name,
                "clone_label": int(p.label),
                "area_px": area_px,
                "perimeter_px": perim_px,
                "equiv_diameter_px": float(eq_diam),
                "major_axis_px": float(p.major_axis_length),
                "minor_axis_px": float(p.minor_axis_length),
                "elongation": elong,
                "eccentricity": float(p.eccentricity),
                "convex_area_px": float(p.convex_area),
                "circularity": circ,
                "solidity": solidity,
                "fractal_dimension": fd,
                **protr,
                "centroid_row": float(p.centroid[0]),
                "centroid_col": float(p.centroid[1]),
                "bbox_minr": int(p.bbox[0]), "bbox_minc": int(p.bbox[1]),
                "bbox_maxr": int(p.bbox[2]), "bbox_maxc": int(p.bbox[3]),
                # scaled
                "area_um2": area_um2, "perimeter_um": perim_um,
                "major_axis_um": maj_um, "minor_axis_um": min_um, "equiv_diameter_um": eqdiam_um,
            })

            # Draw boundary into overlay
            contours = measure.find_contours(clone_mask.astype(float), 0.5)
            for c in contours:
                rr = np.clip(np.round(c[:,0]).astype(int), 0, H-1)
                cc = np.clip(np.round(c[:,1]).astype(int), 0, W-1)
                overlay[rr, cc] = rgb

    df = pd.DataFrame(rows)
    return df, overlay

def main():
    ap = argparse.ArgumentParser(description="Clone analysis from labelled class image.")
    ap.add_argument("--label_tiff", required=True, help="Path to *_labels.tif (integer class indices).")
    ap.add_argument("--legend_csv", required=False, help="Legend CSV with columns: index,label,R,G,B.")
    ap.add_argument("--min_pixels", type=int, default=20, help="Remove clones smaller than this.")
    ap.add_argument("--pix_size_um", type=float, default=None, help="Pixel size in micrometres for physical units.")
    ap.add_argument("--core_radius", type=int, default=5, help="Core radius (px) to detect protrusions.")
    ap.add_argument("--csv_out", required=True, help="Where to write per-clone metrics CSV.")
    ap.add_argument("--overlay_out", required=True, help="Where to write boundary overlay PNG/TIFF.")
    args = ap.parse_args()

    label_img = imread(args.label_tiff)
    if label_img.ndim != 2:
        raise ValueError("label_tiff must be a 2D single-channel labelled image (indices).")
    legend = pd.read_csv(args.legend_csv) if args.legend_csv else None

    df, overlay = analyze(label_img, legend, args.min_pixels, args.pix_size_um, args.core_radius)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    imsave(args.overlay_out, overlay.astype(np.uint8))
    print(f"Wrote: {args.csv_out}\nWrote: {args.overlay_out}\nRows: {len(df)}")

if __name__ == "__main__":
    main()
