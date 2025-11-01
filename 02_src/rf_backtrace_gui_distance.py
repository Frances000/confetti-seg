#!/usr/bin/env python3
"""
Confetti Back-Tracer GUI (Tkinter, Random Forest, distance-variant features)

- Loads images named: Aa01F-145201-YYYYMMDD-merge.tif (or .tiff)
- Extracts date from the filename and sorts ascending
- Tri-panel viewer shows (t, t-1, t-2) with latest on the right
- First ROI locks (W,H); later frames auto-propagate both ROIs and allow repositioning
- Two regions per timepoint: Tumour (red) and Normal (green)
- Builds distance-variant feature planes in-memory (percentile & Otsu binaries,
  skeletons, distance maps) using the same logic as the user's stack generator
- RandomForest (.joblib) inference over both ROIs; computes simple metrics
- No CSV written unless --export_csv is provided; JSON session is always saved

python 02_src/rf_backtrace_gui_distance.py \
    --images_dir /Volumes/Lyons_X5/real_confetti_test/4NQO_stitched/Aa01F-145201-stitched_gui_test_copy \
    --rf_model /Volumes/Lyons_X5/distance_inclusion_variation/confetti-seg-training-reduced/all_dt_models/reduced_stack_no_selection_distance_trained_models/model_distance_rf.joblib

"""


# ---------- Backend & stdlib ----------
from matplotlib import use as mpl_use
mpl_use("TkAgg")  # Tkinter backend (no Qt)
import sys, re, json, argparse
from pathlib import Path
from datetime import datetime

# ---------- GUI: Tkinter + Matplotlib embed ----------
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

# ---------- Scientific stack ----------
import numpy as np
import pandas as pd
import cv2
from tifffile import imread
from sklearn.ensemble import RandomForestClassifier
import joblib

# scikit-image operators
from skimage import util, feature, restoration, color
from skimage.filters import gabor, sobel_h, sobel_v
from skimage.filters.rank import entropy
from skimage.filters import scharr_h, scharr_v, prewitt_h, prewitt_v, gaussian
from skimage.morphology import disk, skeletonize
from skimage.measure import label as cc_label, regionprops_table

# ---------- Filename date parsing ----------
DATE_PAT = re.compile(r".*-(\d{8})-merge(?:\.[Tt][Ii][Ff]{1,2})?$")

def parse_date_from_name(path: Path) -> datetime:
    m = DATE_PAT.match(path.stem)
    if not m:
        raise ValueError(f"Cannot parse date from: {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d")

def load_time_series(images_dir: Path):
    candidates = []
    for p in list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff")):
        try:
            dt = parse_date_from_name(p)
            candidates.append((dt, p))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No images matched '*-YYYYMMDD-merge.tif[f]' in the folder.")
    candidates.sort(key=lambda x: x[0])
    return candidates

# ---------- Display & drawing helpers ----------
def to_display(arr):
    if arr is None:
        return None
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        vmin, vmax = np.percentile(arr, (1, 99))
        return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
    if arr.ndim == 3 and arr.shape[-1] in (3,4):
        vmin, vmax = np.percentile(arr, (1, 99))
        return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
    # fallback (shouldn't be used since we read 2D brightfield)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

def _clear_patches(ax):
    for p in list(ax.patches):
        p.remove()

def _clear_overlays(ax, label):
    for im in list(ax.images):
        if im.get_label() == label:
            im.remove()

def _draw_overlay(ax, overlay_img, label, alpha=0.65):
    _clear_overlays(ax, label)
    ax.imshow(overlay_img, alpha=alpha, label=label)

def ensure_u8_gray(img):
    if img is None or img.size == 0:
        raise ValueError("ensure_u8_gray received an empty image.")
    if img.ndim == 2:
        arr = img
    else:
        if img.shape[-1] >= 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img[..., :3]
            arr = util.img_as_ubyte(color.rgb2gray(rgb))
        else:
            arr = img.max(axis=0)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257).astype(np.uint8)
    # normalize float to 0..255
    a = arr.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a[:] = 0
    return (a * 255.0 + 0.5).astype(np.uint8)

def plt_rect(ax, x, y, w, h, color='lime', lw=2.0):
    rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    return rect

# ---------- Distance-variant feature construction ----------
def _float_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    rng = np.ptp(x)
    if rng > 0:
        x = (x - x.min()) / rng
    else:
        x[:] = 0.0
    return x

def _u8_unit(x: np.ndarray) -> np.ndarray:
    return util.img_as_ubyte(np.clip(x, 0, 1))

def otsu_binary(u8: np.ndarray) -> np.ndarray:
    if u8.dtype != np.uint8:
        u8 = util.img_as_ubyte(u8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def percentile_binary(u8: np.ndarray, pct_top: float = 14.0) -> np.ndarray:
    if u8.dtype != np.uint8:
        u8 = util.img_as_ubyte(u8)
    thr_val = float(np.percentile(u8, 100.0 - pct_top))
    return (u8 >= thr_val).astype(np.uint8) * 255

def _dm_from_binary(mask_u8: np.ndarray) -> np.ndarray:
    inv = cv2.bitwise_not(mask_u8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=5)
    return _u8_unit(_float_norm(dist))

def _skeletonize_u8(mask_u8: np.ndarray) -> np.ndarray:
    bin01 = (mask_u8 > 0)
    skel = skeletonize(bin01)
    return (skel.astype(np.uint8) * 255)

def features_for_channel(Iu8: np.ndarray) -> list[tuple[str, np.ndarray]]:
    I  = Iu8.astype(np.uint8)
    If = I.astype(np.float32) / 255.0
    feats = []
    k = 7
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean   = cv2.filter2D(If, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_sqmean = cv2.filter2D(If**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_var    = np.clip(local_sqmean - local_mean**2, 0, None)
    feats.append(("Mean",      _u8_unit(local_mean)))
    feats.append(("Variance",  _u8_unit(_float_norm(local_var))))

    str_el = np.ones((k, k), np.uint8)
    feats.append(("Minimum",   cv2.erode(I, str_el, borderType=cv2.BORDER_REFLECT)))
    feats.append(("Maximum",   cv2.dilate(I, str_el, borderType=cv2.BORDER_REFLECT)))
    feats.append(("Median",    cv2.medianBlur(I, k if k % 2 else k + 1)))

    tv = restoration.denoise_tv_chambolle(If, weight=0.1)
    feats.append(("Anisotropic_diffusion", _u8_unit(tv)))
    feats.append(("Bilateral", cv2.bilateralFilter(I, d=7, sigmaColor=25, sigmaSpace=7)))

    gx = cv2.Sobel(If, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(If, cv2.CV_32F, 0, 1, ksize=3)
    feats.append(("Lipschitz", _u8_unit(_float_norm(np.hypot(gx, gy)))))

    local_min = cv2.erode(I, str_el, borderType=cv2.BORDER_REFLECT)
    local_max = cv2.dilate(I, str_el, borderType=cv2.BORDER_REFLECT)
    kuwahara_proxy = ((local_min.astype(np.uint16) + local_max.astype(np.uint16)) // 2).astype(np.uint8)
    feats.append(("Kuwahara", kuwahara_proxy))

    real, imag = gabor(If, frequency=0.2, theta=0)
    feats.append(("Gabor_0_0.2", _u8_unit(_float_norm(np.hypot(real, imag)))))

    img = gaussian(If, sigma=1, preserve_range=True)
    feats.append(("SobelX_s1",  _u8_unit(_float_norm(sobel_h(img)))))
    feats.append(("SobelY_s1",  _u8_unit(_float_norm(sobel_v(img)))))
    feats.append(("ScharrX_s1", _u8_unit(_float_norm(scharr_h(img)))))
    feats.append(("ScharrY_s1", _u8_unit(_float_norm(scharr_v(img)))))
    feats.append(("PrewittX_s1",_u8_unit(_float_norm(prewitt_h(img)))))
    feats.append(("PrewittY_s1",_u8_unit(_float_norm(prewitt_v(img)))))

    feats.append(("Derivatives_X", _u8_unit(_float_norm(sobel_h(If)))))
    feats.append(("Derivatives_Y", _u8_unit(_float_norm(sobel_v(If)))))

    A_elems = feature.structure_tensor(If, sigma=1.0)
    l1, l2 = feature.structure_tensor_eigenvalues(A_elems)
    feats.append(("Structure_lambda1", _u8_unit(_float_norm(l1))))
    feats.append(("Structure_lambda2", _u8_unit(_float_norm(l2))))

    ent = entropy(I, disk(4))
    feats.append(("Entropy_r_4", _u8_unit(_float_norm(ent))))

    for r in (8, 16):
        feats.append((f"Neighbors_r{r}", cv2.blur(I, (r, r))))
    return feats

def build_distance_variant_planes(Iu8: np.ndarray, pct_top: float = 14.0):
    planes, labels = [], []
    planes.append(Iu8.astype(np.uint8)); labels.append("original")
    for fname, fimg in features_for_channel(Iu8):
        planes.append(fimg.astype(np.uint8)); labels.append(f"F_{fname}")

    aug_p, aug_l = [], []
    for lbl, img in zip(labels, planes):
        img_u8 = img.astype(np.uint8)
        aug_p.append(img_u8); aug_l.append(lbl)

        m14 = percentile_binary(img_u8, pct_top=pct_top)
        m50 = percentile_binary(img_u8, pct_top=50.0)
        mot = otsu_binary(img_u8)
        aug_p.extend([m14, m50, mot])
        aug_l.extend([f"{lbl}_Pctl{int(round(pct_top))}", f"{lbl}_Pctl50", f"{lbl}_Otsu"])

        dm14, dm50, dmots = _dm_from_binary(m14), _dm_from_binary(m50), _dm_from_binary(mot)
        sk14, sk50, skots = _skeletonize_u8(m14), _skeletonize_u8(m50), _skeletonize_u8(mot)
        dmsk14, dmsk50, dmskots = _dm_from_binary(sk14), _dm_from_binary(sk50), _dm_from_binary(skots)
        aug_p.extend([dm14, dm50, dmots, dmsk14, dmsk50, dmskots])
        aug_l.extend([f"DM14pcThresh_{lbl}", f"DM50pcThresh_{lbl}", f"DMotsuThresh_{lbl}",
                      f"DM14pcSkel_{lbl}",  f"DM50pcSkel_{lbl}",  f"DMotsuSkel_{lbl}"])
    return aug_p, aug_l

def build_distance_features_X(Iu8: np.ndarray, pct_top: float = 14.0):
    planes, labels = build_distance_variant_planes(Iu8, pct_top=pct_top)
    stack = np.stack(planes, axis=-1)     # (H,W,F)
    H, W, F = stack.shape
    X = stack.reshape(-1, F).astype(np.uint8)
    return X, (H, W), labels

# ---------- Metrics ----------
def compute_metrics(label_img, min_area_px=8):
    labs = np.unique(label_img); labs = labs[labs != 0]
    total_clones, comp_areas_all = 0, []
    per_label_area = {}
    for lab in labs:
        mask = (label_img == lab).astype(np.uint8)
        if mask.sum() == 0: continue
        cc = cc_label(mask, connectivity=2)
        props = regionprops_table(cc, properties=('label', 'area'))
        if len(props['area']) == 0: continue
        areas = np.array(props['area'], dtype=np.int32)
        areas = areas[areas >= min_area_px]
        total_clones += len(areas)
        comp_areas_all.extend(areas.tolist())
        per_label_area[int(lab)] = int(mask.sum())
    mean_area = float(np.mean(comp_areas_all)) if comp_areas_all else 0.0
    total_area = int(np.sum([v for v in per_label_area.values()])) if per_label_area else 0
    return {
        "total_clones": int(total_clones),
        "total_area_px": total_area,
        "mean_component_area_px": float(mean_area),
        "per_label_area_px": per_label_area
    }

# ---------- Tri-panel widget ----------
class TriPanel:
    def __init__(self, parent):
        self.fig = Figure(figsize=(12, 4), tight_layout=True)
        self.ax = [self.fig.add_subplot(1,3,i+1) for i in range(3)]
        for a in self.ax: a.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()

    def grid(self, **kwargs):
        self.canvas_widget.grid(**kwargs)

    def show_triplet(self, imgs, titles):
        for k in range(3):
            ax = self.ax[k]
            ax.cla(); ax.set_axis_off()
            if imgs[k] is not None:
                disp = to_display(imgs[k])
                if disp is not None:
                    if disp.ndim == 3:
                        ax.imshow(disp)
                    else:
                        ax.imshow(disp, cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles[k], fontsize=10)
        self.canvas.draw_idle()

# ---------- App ----------
class App(tk.Tk):
    def __init__(self, images, rf_model_path, export_csv=None, session_json=None, pct_top=14.0):
        super().__init__()
        self.title("Confetti Back-Tracer (RF, Distance Features) — Tkinter")
        self.images = images
        self.export_csv = export_csv
        self.session_json = session_json
        self.pct_top = float(pct_top)
        self.model: RandomForestClassifier = joblib.load(rf_model_path)

        self.idx = len(self.images) - 1
        self.fixed_size = None                  # (w,h) — locked on first selection
        self.skips = set()                      # skipped indices

        # Per-region ROI dictionaries: index -> (x,y,w,h)
        self.rois_tumour = {}
        self.rois_normal = {}

        # Per-region metrics: index -> dict
        self.metrics_tumour = {}
        self.metrics_normal = {}

        # Per-region overlay caches: index -> RGB overlay
        self.overlay_tumour = {}
        self.overlay_normal = {}

        # Top controls
        top = ttk.Frame(self); top.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.btn_prev = ttk.Button(top, text="← Previous", command=self.go_prev)
        self.btn_next = ttk.Button(top, text="Next →", command=self.go_next)
        self.skip_var = tk.BooleanVar(value=False)
        self.chk_skip = ttk.Checkbutton(top, text="Skip this timepoint",
                                        variable=self.skip_var, command=self.toggle_skip)
        self.btn_prev.grid(row=0, column=0, padx=4)
        self.btn_next.grid(row=0, column=1, padx=4)
        top.grid_columnconfigure(2, weight=1)
        self.chk_skip.grid(row=0, column=3, padx=4)

        # Region picker (Tumour / Normal)
        region_row = ttk.Frame(self)
        region_row.grid(row=0, column=0, sticky="e", padx=8, pady=(0,4))
        self.active_region = tk.StringVar(value="tumour")
        ttk.Label(region_row, text="Active region:").grid(row=0, column=0, padx=(0,6))
        ttk.Radiobutton(region_row, text="Tumour", value="tumour",
                        variable=self.active_region).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(region_row, text="Normal", value="normal",
                        variable=self.active_region).grid(row=0, column=2, padx=4)

        # Tri-panel
        self.panel = TriPanel(self)
        self.panel.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Bottom controls
        bottom = ttk.Frame(self); bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        self.btn_set_roi = ttk.Button(bottom, text="Set/Move ROI (current)", command=self.activate_roi_selector)
        self.btn_predict = ttk.Button(bottom, text="Predict Regions (current)", command=self.predict_both_current)
        self.btn_save = ttk.Button(bottom, text="Save Session", command=self.save_session)
        self.btn_set_roi.grid(row=0, column=0, padx=4)
        self.btn_predict.grid(row=0, column=1, padx=4)
        bottom.grid_columnconfigure(2, weight=1)
        self.btn_save.grid(row=0, column=3, padx=4)

        # Status
        self.info_lbl = ttk.Label(self, text="Load complete.", wraplength=1000, anchor="w", justify="left")
        self.info_lbl.grid(row=3, column=0, sticky="ew", padx=8, pady=6)

        # Rectangle selector (Matplotlib)
        self.rect_selector = None
        self._rs_active = False

        self.refresh()

    # ----- dict accessors -----
    def _get_roi_dict(self, region: str):
        return self.rois_tumour if region == "tumour" else self.rois_normal

    def _get_metrics_dict(self, region: str):
        return self.metrics_tumour if region == "tumour" else self.metrics_normal

    def _get_overlay_dict(self, region: str):
        return self.overlay_tumour if region == "tumour" else self.overlay_normal

    # ----- utility -----
    def _clamp_roi_to_image(self, roi, img_shape):
        x,y,w,h = roi
        H_all, W_all = img_shape[:2]
        w = min(w, W_all); h = min(h, H_all)
        x = max(0, min(x, max(0, W_all - w)))
        y = max(0, min(y, max(0, H_all - h)))
        return (x,y,w,h)

    # --- image loading (LEFT=prev/newer, MIDDLE=current, RIGHT=older) ---
    def load_triplet(self):
        def read_by_index(i: int):
            if i is None or i < 0 or i >= len(self.images):
                return None, ""
            dt, p = self.images[i]
            img = None
            try:
                # Brightfield only: 3rd page (index 2)
                img = imread(str(p), key=2)
            except Exception:
                try:
                    full = imread(str(p))
                    if full.ndim == 3 and full.shape[-1] not in (3, 4) and full.shape[0] >= 3:
                        img = full[2]
                    elif full.ndim == 2:
                        img = full
                    else:
                        img = full
                except Exception:
                    img = None

            if img is None:
                return None, f"{p.stem}"
            # force 2D u8
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                img = util.img_as_ubyte(color.rgb2gray(img[..., :3]))
            else:
                if img.dtype == np.uint16:
                    img = (img / 257).astype(np.uint8)
                elif img.dtype != np.uint8:
                    arr = img.astype(np.float32)
                    vmin, vmax = np.min(arr), np.max(arr)
                    img = ((arr - vmin) / (vmax - vmin + 1e-9) * 255.0).astype(np.uint8)
            return img, f"{p.stem}"

        i_cur  = self.idx
        i_prev = self.idx + 1 if self.idx < len(self.images) - 1 else None   # newer (guidance)
        i_old  = self.idx - 1 if self.idx > 0 else None                      # older

        imL, tL = read_by_index(i_prev)   # left  = previous/newer
        imM, tM = read_by_index(i_cur)    # middle = current
        imR, tR = read_by_index(i_old)    # right = older

        titles = [tL, tM, tR]
        return [imL, imM, imR], titles

    def refresh(self):
        imgs, titles = self.load_triplet()
        img_current = imgs[1]  # middle is current

        # Strict back-trace: copy EXACT coords from prev/newer (idx+1) into current if missing
        prev_idx = self.idx + 1 if self.idx < len(self.images) - 1 else None
        if img_current is not None and prev_idx is not None:
            Hc, Wc = img_current.shape[:2]
            if self.idx not in self.rois_tumour and prev_idx in self.rois_tumour:
                self.rois_tumour[self.idx] = self._clamp_roi_to_image(self.rois_tumour[prev_idx], img_current.shape)
            if self.idx not in self.rois_normal and prev_idx in self.rois_normal:
                self.rois_normal[self.idx] = self._clamp_roi_to_image(self.rois_normal[prev_idx], img_current.shape)

        # Draw panels
        self.panel.show_triplet(imgs, titles)
        axL, axM, axR = self.panel.ax  # left, middle, right

        # Current (middle): solid boxes + overlays; tumour=red, normal=green
        if self.idx in self.rois_tumour:
            x,y,w,h = self.rois_tumour[self.idx]
            plt_rect(axM, x, y, w, h, color='red', lw=1.6)
            if self.idx in self.overlay_tumour:
                _draw_overlay(axM, self.overlay_tumour[self.idx], label="overlay_tumour")
        if self.idx in self.rois_normal:
            x,y,w,h = self.rois_normal[self.idx]
            plt_rect(axM, x, y, w, h, color='green', lw=1.6)
            if self.idx in self.overlay_normal:
                _draw_overlay(axM, self.overlay_normal[self.idx], label="overlay_normal")

        # Previous/newer (left): guidance with dashed boxes
        if prev_idx is not None:
            if prev_idx in self.rois_tumour:
                x,y,w,h = self.rois_tumour[prev_idx]
                rect = plt_rect(axL, x, y, w, h, color='red', lw=1.2)
                rect.set_linestyle('--')
            if prev_idx in self.rois_normal:
                x,y,w,h = self.rois_normal[prev_idx]
                rect = plt_rect(axL, x, y, w, h, color='green', lw=1.2)
                rect.set_linestyle('--')

        self.panel.canvas.draw_idle()

        self.skip_var.set(self.idx in self.skips)

        dt, p = self.images[self.idx]
        roi_t = self.rois_tumour.get(self.idx, 'None')
        roi_n = self.rois_normal.get(self.idx, 'None')
        mt_t = self.metrics_tumour.get(self.idx)
        mt_n = self.metrics_normal.get(self.idx)
        mtxt_t = f"T metrics: {mt_t}" if mt_t else "T metrics: —"
        mtxt_n = f"N metrics: {mt_n}" if mt_n else "N metrics: —"
        self.info_lbl.config(
            text=f"Current: {p.name} (date {dt.date()}) | "
                 f"ROI[T]: {roi_t} | ROI[N]: {roi_n} | Fixed size: {self.fixed_size} | "
                 f"{mtxt_t} | {mtxt_n}"
        )

    # --- nav/skip ---
    def go_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.refresh()

    def go_next(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self.refresh()

    def toggle_skip(self):
        if self.skip_var.get():
            self.skips.add(self.idx)
        else:
            self.skips.discard(self.idx)
        self.refresh()

    # --- ROI selection on the MIDDLE panel ---
    def activate_roi_selector(self):
        ax = self.panel.ax[1]  # middle panel is current
        region = self.active_region.get()                 # "tumour" or "normal"
        roi_dict = self._get_roi_dict(region)

        if self._rs_active:
            self.rect_selector.set_active(False)
            self._rs_active = False
            self.btn_set_roi.config(text="Set/Move ROI (current)")
            return

        def onselect(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                messagebox.showwarning("ROI", "Invalid ROI selection.")
                return
            x0, y0 = int(eclick.xdata),  int(eclick.ydata)
            x1, y1 = int(erelease.xdata), int(erelease.ydata)
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w <= 0 or h <= 0:
                messagebox.showwarning("ROI", "Invalid ROI selection.")
                return

            # Cap initial fixed size to current image size
            imgs, _ = self.load_triplet()
            img_current = imgs[1]
            if img_current is not None:
                H_all, W_all = img_current.shape[:2]
                w = min(w, W_all)
                h = min(h, H_all)

            # Lock size on first ever ROI (either region)
            if self.fixed_size is None:
                self.fixed_size = (w, h)
            else:
                w, h = self.fixed_size

            # Clamp position inside current image bounds
            x = max(0, min(x, W_all - w))
            y = max(0, min(y, H_all - h))

            roi_dict[self.idx] = (x, y, w, h)

            # Redraw both ROIs distinctly (tumour=red, normal=green) on current/middle
            axL, axM, axR = self.panel.ax
            _clear_patches(axM)
            if self.idx in self.rois_tumour:
                tx,ty,tw,th = self.rois_tumour[self.idx]
                plt_rect(axM, tx, ty, tw, th, color='red', lw=1.6)
            if self.idx in self.rois_normal:
                nx,ny,nw,nh = self.rois_normal[self.idx]
                plt_rect(axM, nx, ny, nw, nh, color='green', lw=1.6)
            self.panel.canvas.draw_idle()

            self.info_lbl.config(text=f"Set {region} ROI at idx={self.idx}: {(x,y,w,h)} | Fixed size={self.fixed_size}")

            # If this is the very first ROI at last frame:
            if self.idx == len(self.images) - 1 and (len(self.rois_tumour)+len(self.rois_normal)) == 1:
                messagebox.showinfo("ROI locked",
                                    "ROI size locked. Move backwards and reposition as needed (size stays fixed).")

        self.rect_selector = RectangleSelector(
            ax, onselect,
            useblit=True, interactive=True,
            button=[1], minspanx=2, minspany=2
        )
        self._rs_active = True
        self.btn_set_roi.config(text="Finish ROI selection")

    # --- prediction helpers ---
    def _predict_one_region(self, region):
        roi_dict      = self._get_roi_dict(region)
        metrics_dict  = self._get_metrics_dict(region)
        overlay_dict  = self._get_overlay_dict(region)

        if self.idx not in roi_dict:
            return  # silently skip if this region not defined

        imgs, _ = self.load_triplet()
        img_current = imgs[1]  # middle/current
        if img_current is None:
            return

        # --- clamp ROI to current image and validate ---
        x, y, w, h = roi_dict[self.idx]
        H_all, W_all = img_current.shape[:2]

        # shrink ROI width/height if needed
        w = min(w, W_all)
        h = min(h, H_all)

        # clamp top-left so the box stays inside
        x = max(0, min(x, W_all - w))
        y = max(0, min(y, H_all - h))

        x0, x1 = x, x + w
        y0, y1 = y, y + h
        if x0 >= x1 or y0 >= y1:
            messagebox.showwarning("ROI out of view",
                                   f"{region.capitalize()} ROI is outside the current image; adjust its position.")
            return

        crop = img_current[y0:y1, x0:x1]
        if crop.size == 0:
            messagebox.showwarning("Empty crop", f"{region.capitalize()} ROI produced an empty crop; adjust its position.")
            return

        # Build features + predict
        gray_u8 = ensure_u8_gray(crop)
        X, (H, W), _ = build_distance_features_X(gray_u8, pct_top=self.pct_top)
        yhat = self.model.predict(X).astype(np.int32)
        label_img = yhat.reshape(H, W)

        # Metrics
        metr = compute_metrics(label_img)
        metrics_dict[self.idx] = metr

        # Edges overlay: tumour -> RED channel; normal -> GREEN channel
        edges = (cv2.Canny((label_img > 0).astype(np.uint8) * 255, 0, 1) > 0).astype(np.float32)
        overlay = np.zeros((img_current.shape[0], img_current.shape[1], 3), dtype=np.float32)
        if region == "tumour":
            overlay[y0:y1, x0:x1, 0] = edges  # R
        else:
            overlay[y0:y1, x0:x1, 1] = edges  # G
        overlay_dict[self.idx] = overlay

    def predict_both_current(self):
        if self.idx in self.skips:
            messagebox.showinfo("Skipped", "This timepoint is skipped.")
            return

        self._predict_one_region("tumour")
        self._predict_one_region("normal")

        # Redraw both regions & overlays on the current/middle panel cleanly
        axL, axM, axR = self.panel.ax
        _clear_patches(axM)
        _clear_overlays(axM, "overlay_tumour")
        _clear_overlays(axM, "overlay_normal")

        if self.idx in self.rois_tumour:
            tx,ty,tw,th = self.rois_tumour[self.idx]
            plt_rect(axM, tx, ty, tw, th, color='red', lw=1.6)
            if self.idx in self.overlay_tumour:
                _draw_overlay(axM, self.overlay_tumour[self.idx], "overlay_tumour")
        if self.idx in self.rois_normal:
            nx,ny,nw,nh = self.rois_normal[self.idx]
            plt_rect(axM, nx, ny, nw, nh, color='green', lw=1.6)
            if self.idx in self.overlay_normal:
                _draw_overlay(axM, self.overlay_normal[self.idx], "overlay_normal")

        self.panel.canvas.draw_idle()
        self.refresh()

    # --- save ---
    def save_session(self):
        session = {
            "images": [str(p) for _, p in self.images],
            "dates": [dt.strftime("%Y-%m-%d") for dt, _ in self.images],
            "rois_tumour": {str(k): v for k, v in self.rois_tumour.items()},
            "rois_normal": {str(k): v for k, v in self.rois_normal.items()},
            "skips": list(self.skips),
            "metrics_tumour": self.metrics_tumour,
            "metrics_normal": self.metrics_normal,
            "pct_top": self.pct_top
        }

        out_json = self.session_json or str(Path.cwd() / "rf_backtrace_session.json")
        with open(out_json, "w") as f:
            json.dump(session, f, indent=2)

        if self.export_csv:
            rows = []
            for region, metrics in (("tumour", self.metrics_tumour), ("normal", self.metrics_normal)):
                for idx, metr in metrics.items():
                    dt, p = self.images[idx]
                    rows.append({
                        "region": region,
                        "index": idx,
                        "date": dt.strftime("%Y-%m-%d"),
                        "filename": p.name,
                        "skipped": (idx in self.skips),
                        "total_clones": metr["total_clones"],
                        "total_area_px": metr["total_area_px"],
                        "mean_component_area_px": metr["mean_component_area_px"],
                        "per_label_area_px": json.dumps(metr["per_label_area_px"])
                    })
            pd.DataFrame(rows).sort_values(["date","region","index"]).to_csv(self.export_csv, index=False)

        messagebox.showinfo("Saved", f"Session saved to:\n{out_json}" +
                            (f"\nCSV: {self.export_csv}" if self.export_csv else ""))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Confetti Back-Tracer (Tkinter, RF, distance features)")
    ap.add_argument("--images_dir", required=True, type=Path, help="Folder with *-YYYYMMDD-merge.tif[f]")
    ap.add_argument("--rf_model", required=True, type=Path, help="Path to RandomForest .joblib trained on distance features")
    ap.add_argument("--pct_top", type=float, default=14.0, help="Percentile top-cut for distance binaries (default 14%)")
    ap.add_argument("--export_csv", type=Path, default=None, help="Optional path to export metrics CSV")
    ap.add_argument("--session_json", type=Path, default=None, help="Optional path to save session JSON")
    args = ap.parse_args()

    imgs = load_time_series(args.images_dir)

    app = App(imgs, str(args.rf_model),
              export_csv=str(args.export_csv) if args.export_csv else None,
              session_json=str(args.session_json) if args.session_json else None,
              pct_top=float(args.pct_top))
    app.geometry("1200x700")
    app.mainloop()

if __name__ == "__main__":
    main()
