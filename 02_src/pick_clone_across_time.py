#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict

from skimage.measure import regionprops, label as sklabel
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from typing import Optional, Tuple, List


# ---------------- IO helpers ----------------
def natsort_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def _pair_by_stem(dir_path: Path) -> dict[str, dict[str, Path]]:
    """
    Group distance_* preds files by a shared stem before '__preds_'.
    Returns: { stem: {'legend': Path, 'labels': Path} }
    """
    pairs = defaultdict(dict)
    for p in dir_path.glob("distance_*__preds_legend.csv"):
        stem = p.name.split("__preds_", 1)[0]
        pairs[stem]['legend'] = p
    for p in dir_path.glob("distance_*__preds_labels.tif"):
        stem = p.name.split("__preds_", 1)[0]
        pairs[stem]['labels'] = p
    # keep only stems that have both
    return {s: f for s, f in pairs.items() if 'legend' in f and 'labels' in f}

def discover_all_csv_bundles(root: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Return a list of (legend_csv, labels_tif, csv_dir) for ALL csv_* dirs under root.
    Sorting: directory name ascending; within a dir, stems sorted ascending.
    """
    if not root.is_dir():
        return []
    csv_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("csv_")],
                      key=lambda p: p.name)
    results: List[Tuple[Path, Path, Path]] = []
    for d in csv_dirs:
        pairs = _pair_by_stem(d)
        for stem in sorted(pairs.keys()):
            results.append((pairs[stem]['legend'], pairs[stem]['labels'], d))
    return results

def instances_from_class_index(idx_img: np.ndarray, bg_indices: set[int]) -> tuple[np.ndarray, dict[int, int]]:
    """
    Turn a class index image (HxW ints) into an instance label image (HxW ints),
    by connected components within each non-background class.
    Returns:
      inst (HxW int): 0=background, 1..N=instances
      region2class: {instance_label -> class_index}
    """
    idx = np.asarray(idx_img)
    H, W = idx.shape[:2]
    inst = np.zeros((H, W), dtype=np.int32)
    region2class: dict[int, int] = {}
    next_id = 1

    for cls in np.unique(idx):
        if int(cls) in bg_indices:
            continue
        cc = sklabel(idx == int(cls), connectivity=2)
        if cc.max() == 0:
            continue
        for k in range(1, cc.max() + 1):
            inst[cc == k] = next_id
            region2class[next_id] = int(cls)
            next_id += 1
    return inst, region2class



def _legend_to_palette(legend_df: pd.DataFrame):
    """
    Normalise columns & return:
      palette: dict[int] -> (R,G,B) uint8
      idx2lab: dict[int] -> class label string
      lab2idx: dict[str] -> int
      bg_indices: set[int] (indices to treat as background)
    """
    cols = {c.lower(): c for c in legend_df.columns}
    need = {"index", "label", "r", "g", "b"}
    if not need.issubset(set(cols)):
        raise ValueError(f"Legend CSV needs columns {need}, got {set(legend_df.columns)}")
    df = legend_df.rename(columns={cols["index"]: "index",
                                   cols["label"]: "label",
                                   cols["r"]: "R", cols["g"]: "G", cols["b"]: "B"})
    df["index"] = df["index"].astype(int)
    df = df.sort_values("index")
    palette = {int(r["index"]): (int(r["R"]), int(r["G"]), int(r["B"])) for _, r in df.iterrows()}
    idx2lab = {int(r["index"]): str(r["label"]) for _, r in df.iterrows()}
    lab2idx = {v: k for k, v in idx2lab.items()}

    # Background heuristics: label 'B' (or 'background'), or exact black (0,0,0)
    bg_indices = set()
    for idx, lab in idx2lab.items():
        rgb = palette[idx]
        if lab.strip().upper() in {"B", "BG", "BACKGROUND"} or rgb == (0, 0, 0):
            bg_indices.add(idx)
    if not bg_indices and 0 in palette:
        # sensible fallback
        bg_indices.add(0)
    return palette, idx2lab, lab2idx, bg_indices

def instances_from_class_index(idx_img: np.ndarray, bg_indices: set[int]) -> tuple[np.ndarray, dict[int,int]]:
    """
    Connected components *within each non-background class* -> instance image.
    Returns (HxW ints), and {instance_id -> class_index}.
    """
    idx = np.asarray(idx_img)
    H, W = idx.shape[:2]
    inst = np.zeros((H, W), dtype=np.int32)
    region2class: Dict[int,int] = {}
    next_id = 1
    for cls in np.unique(idx):
        c = int(cls)
        if c in bg_indices: continue
        cc = sklabel(idx == c, connectivity=2)
        if cc.max() == 0: continue
        for k in range(1, cc.max()+1):
            inst[cc == k] = next_id
            region2class[next_id] = c
            next_id += 1
    return inst, region2class

def list_images(p: Path):
    if p.is_dir():
        fs = sorted([f for f in p.iterdir()
                     if f.suffix.lower() in {".tif", ".tiff", ".png"}],
                    key=lambda x: natsort_key(x.name))
        if not fs:
            raise FileNotFoundError(f"No images in {p}")
        return fs
    return [p]

def load_stack_any(path: Path):
    """
    Load:
      - folder of single-page images
      - single multipage TIFF
      - HxW or HxWxC image (if C present we use channel 0)
    Returns: list[np.ndarray], list[str] (names)
    """
    files = list_images(path)
    if len(files) == 1:
        arr = tiff.imread(str(files[0]))
        name = files[0].name
        if arr.ndim == 2:
            return [arr], [name]
        if arr.ndim == 3:
            # T,H,W  OR  H,W,C
            if arr.shape[-1] in (3,4):
                return [arr[...,0]], [name]
            return [arr[i] for i in range(arr.shape[0])], [f"{name}::frame{i}" for i in range(arr.shape[0])]
        if arr.ndim == 4:
            # T,H,W,C -> take channel 0
            return [arr[i,...,0] for i in range(arr.shape[0])], [f"{name}::frame{i}" for i in range(arr.shape[0])]
        raise ValueError(f"Unsupported image shape {arr.shape} for {name}")
    # multiple separate files
    return [tiff.imread(str(f)) for f in files], [f.name for f in files]

def ensure_labelled(mask: np.ndarray) -> np.ndarray:
    """
    Convert binary masks to connected-component labels.
    Preserve integer labels if already present.
    """
    m = mask
    if m.ndim > 2:
        m = m[...,0]
    m = np.asarray(m)
    if np.issubdtype(m.dtype, np.floating):
        # floats coming from some tiffs; cast to int
        m = m.astype(np.int64)
    uniq = np.unique(m)
    if uniq.size <= 3 and set(uniq.tolist()) <= {0,1,255}:
        return sklabel((m > 0).astype(np.uint8), connectivity=2)
    return m

# ---------------- Picker UI ----------------
class MultiFrameClonePicker:
    """
    Show all mask frames in a grid. User clicks any panels in any order to select
    the same clone across time. Optional raw images can be shown underneath.

    Mouse:
      - Left click: select clone at pointer (must be nonzero label)

    Keys:
      - 'enter' or 'n': finish & save
      - 'q': quit early (saves current progress)
      - 'r' (hovering a panel): clear that panel
      - 's' (hovering a panel): mark that panel skipped
      - 'a': clear ALL panels
      - 'b': toggle boundaries visibility
    """
    def __init__(self,
                 masks: list[np.ndarray],
                 names: list[str],
                 out_csv: Path,
                 raw_images: list[np.ndarray] | None = None,
                 overlay_alpha: float = 0.35,
                 px_size_um=None,
                 frame_palettes: list[dict[int, tuple[int,int,int]]] | None = None,
                 frame_idx2lab:  list[dict[int, str]] | None = None,
                 precomputed_region_class: list[dict[int,int]] | None = None):
        self.masks = [ensure_labelled(m) for m in masks]
        self.names = names
        self.N = len(self.masks)
        self.raw_images = raw_images if (raw_images is not None and len(raw_images) == self.N) else None
        self.out_csv = out_csv
        self.overlay_alpha = float(overlay_alpha)

        # per-frame state
        self.x = [np.nan]*self.N
        self.y = [np.nan]*self.N
        self.label_id = [np.nan]*self.N
        self.centroid_xy = [(np.nan, np.nan)]*self.N
        self.area = [np.nan]*self.N
        self.skipped = [False]*self.N

        # Matplotlib state
        self.fig = None
        self.axes = []
        self.markers = [None]*self.N
        self.texts = [None]*self.N
        self.boundary_overlays = [None]*self.N
        self.base_images = [None]*self.N  # the RGB label composite

        # toggles
        self.show_bounds = True

        # metadata for quick lookup
        self.props = []
        for m in self.masks:
            pmap = {p.label: p for p in regionprops(m)}
            self.props.append(pmap)

        self.precomputed_region_class = precomputed_region_class

        self.class_palette = None
        self.bg_indices = set()
        self.px_size_um = px_size_um
        self.frame_palettes = frame_palettes or [dict() for _ in range(len(masks))]
        self.frame_idx2lab  = frame_idx2lab  or [dict() for _ in range(len(masks))]
        self.precomputed_region_class = precomputed_region_class or [dict() for _ in range(len(masks))]

        # per-panel computed colour & class
        self.region_class_index = [dict() for _ in range(self.N)]
        self.region_class_color = [dict() for _ in range(self.N)]
    # -------- utilities --------

    def _assign_region_classes(self, i: int):
        mask = self.masks[i]
        labels = np.unique(mask); labels = labels[labels != 0]
        # precomputed region→class (authoritative)
        r2c = self.precomputed_region_class[i] or {}
        self.region_class_index[i] = dict(r2c)
        pal = self.frame_palettes[i] or {}
        # paint colour from per-frame legend; fallback to spectral
        for lbl in labels:
            cls = self.region_class_index[i].get(lbl, None)
            if cls is not None and cls in pal:
                r,g,b = pal[cls]; self.region_class_color[i][lbl] = (r/255, g/255, b/255)
            else:
                # fallback if missing
                maxlbl = int(mask.max())
                self.region_class_color[i][lbl] = self._make_colors(maxlbl)[lbl-1]



    def _grid_shape(self, N):
        cols = math.ceil(math.sqrt(N))
        rows = math.ceil(N/cols)
        return rows, cols
    
    def _compose_rgb_with_classes(self, i: int) -> np.ndarray:
        mask = self.masks[i]; h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for lbl, col in self.region_class_color[i].items():
            m = (mask == lbl)
            r,g,b = col
            rgb[m,0]=r; rgb[m,1]=g; rgb[m,2]=b
        return rgb
    
    def _label_info(self, panel_idx: int, lbl: int) -> str:
        rp = self.props[panel_idx].get(lbl, None)
        area_px = int(rp.area) if rp is not None else 0
        cy, cx = (rp.centroid if rp is not None else (np.nan, np.nan))

        cls_name = self._class_name(panel_idx, lbl)
        # physical units optional, if you added px_size_um previously
        area_line = f"area: {area_px} px"
        if hasattr(self, "px_size_um") and self.px_size_um:
            a_um2 = area_px * (self.px_size_um ** 2)
            area_line = f"area: {area_px} px  ({a_um2:.1f} µm²)"

        info = (f"{self.names[panel_idx]}"
                f"\nclass: {cls_name}"
                f"\n{area_line}"
                f"\ncentroid: ({int(round(cx))}, {int(round(cy))})")
        return info

    def _class_name(self, panel_idx: int, lbl: int) -> str:
        """Return the class name (e.g., 'BR', 'RR') for this region; '?' if unknown."""
        cls_idx = None
        if hasattr(self, "region_class_index") and self.region_class_index:
            cls_idx = self.region_class_index[panel_idx].get(lbl, None)

        # per-frame legend mapping (preferred if present)
        if hasattr(self, "frame_idx2lab") and self.frame_idx2lab and self.frame_idx2lab[panel_idx]:
            idx2lab = self.frame_idx2lab[panel_idx]
            return idx2lab.get(cls_idx, "?") if cls_idx is not None else "?"
        # single-legend mapping fallback (if you used a single legend_df earlier)
        if hasattr(self, "idx2lab") and self.idx2lab:
            return self.idx2lab.get(cls_idx, "?") if cls_idx is not None else "?"
        return "?"

    def _panel_title(self, i):
        li = self.label_id[i]
        if self.skipped[i]:
            status = "SKIP"
        elif not np.isnan(li):
            status = f"class {self._class_name(i, int(li))}"
        else:
            status = "—"
        return f"#{i} {self.names[i]}  |  {status}"


    def _refresh(self):
        # small GUI pump helps certain backends register updates promptly
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
    
    def _make_colors(self, K: int):
        """
        Deterministic palette for labels 1..K.
        Returns a list of RGB tuples (len = K), ignoring label 0.
        """
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('nipy_spectral')  # vivid, spaced palette
        # skip very dark values at the ends; spread colours evenly
        return [cmap((i + 1) / (K + 2))[:3] for i in range(K)]

    def _draw_hover(self, i, xi, yi, lbl):
        ax = self.axes[i]
        self._clear_hover(i)

        # class colour for this region
        col = None
        if hasattr(self, "region_class_color") and self.region_class_color:
            col = self.region_class_color[i].get(lbl, None)
        if col is None:
            col = (1, 1, 1)  # fallback

        m = (self.masks[i] == lbl).astype(np.uint8)
        try:
            self._hover_outline[i] = ax.contour(m, levels=[0.5], colors=[col], linewidths=2, alpha=0.95)
        except Exception:
            self._hover_outline[i] = None

        txt = self._label_info(i, lbl)
        self._hover_text[i] = ax.text(
            xi + 5, yi + 10, txt, fontsize=6, color='w',
            va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.65, pad=3, edgecolor='none')
        )


    def _highlight_panel(self, i, on=True):
        ax = self.axes[i]
        for spine in ax.spines.values():
            spine.set_linewidth(2 if on else 0.7)
            spine.set_edgecolor('tab:blue' if on else '0.5')

    def _compose_rgb(self, i):
        """Create colourised label image, optionally blended over grayscale raw."""
        mask = self.masks[i]
        if self.raw_images is not None:
            base = self.raw_images[i].astype(np.float32)
            vmin, vmax = np.percentile(base, [1, 99])
            denom = max(vmax - vmin, 1e-6)
            base_norm = np.clip((base - vmin) / denom, 0, 1)
            # blend labels over the grayscale base
            rgb = label2rgb(mask, image=base_norm, bg_label=0,
                            alpha=self.overlay_alpha, kind='overlay')
        else:
            # NO image provided → use a deterministic palette (no 'avg'!)
            maxlbl = int(mask.max())
            if maxlbl <= 0:
                # no labels; return black RGB
                h, w = mask.shape
                rgb = np.zeros((h, w, 3), dtype=np.float32)
            else:
                colors = self._make_colors(maxlbl)   # list for labels 1..maxlbl
                rgb = label2rgb(mask, colors=colors, bg_label=0, bg_color=(0, 0, 0))
        return rgb


    # -------- draw/launch --------
    def launch(self):
        rows, cols = self._grid_shape(self.N)
        self.fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
        self.axes = axs.ravel().tolist()

        for i, (ax, mask) in enumerate(zip(self.axes, self.masks)):
            H, W = mask.shape
            extent = (0, W, H, 0)

            self._assign_region_classes(i)
            rgb = self._compose_rgb_with_classes(i)
            self.base_images[i] = ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")

            # boundaries overlay on top
            bounds = find_boundaries(mask, mode="outer").astype(np.float32)
            self.boundary_overlays[i] = ax.imshow(
                np.dstack([bounds]*3), extent=extent, origin="upper",
                interpolation="nearest", alpha=0.35, visible=self.show_bounds
            )

            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_title(self._panel_title(i), fontsize=5)
            ax.set_xticks([]); ax.set_yticks([])

        # instructions
        self.fig.suptitle("Click clones in any order across panels  |  keys: [enter/n]=finish, r=clear, s=skip, a=clear all, b=toggle bounds, q=quit",
                          fontsize=11)

        # connect events
        self.cid_click  = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key    = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        plt.tight_layout()
        plt.show()

    # -------- events --------
    def _axes_index(self, ax):
        try:
            return self.axes.index(ax)
        except ValueError:
            return None

    def _on_motion(self, event):
        # show which panel is "hot" for panel-scoped keys
        for j, ax in enumerate(self.axes):
            self._highlight_panel(j, on=(event.inaxes is ax))
        self._refresh()

    def _on_click(self, event):
        if event.inaxes is None:
            return
        idx = self._axes_index(event.inaxes)
        if idx is None:
            return
        mask = self.masks[idx]
        H, W = mask.shape
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xi = int(np.clip(round(x), 0, W-1))
        yi = int(np.clip(round(y), 0, H-1))

        lbl = int(mask[yi, xi])
        if lbl == 0:
            self._show_status(idx, "Background selected; click inside a clone.", color='y')
            return

        # region props
        pmap = self.props[idx]
        if lbl in pmap:
            region = pmap[lbl]
            cy, cx = region.centroid
            area = float(region.area)
        else:
            cy = cx = area = np.nan

        # store and draw
        self.x[idx], self.y[idx] = xi, yi
        self.label_id[idx] = lbl
        self.centroid_xy[idx] = (cy, cx)
        self.area[idx] = area
        self.skipped[idx] = False

        cls_name = self._class_name(idx, lbl)
        self._draw_marker(idx, xi, yi, f"(x={xi}, y={yi}) → class {cls_name}")

        self.axes[idx].set_title(self._panel_title(idx), fontsize=5)
        self._refresh()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k in ("enter", "return", "n"):
            self._finalise_and_save()
            plt.close(self.fig)
            return
        if k == "q":
            self._save_progress()
            plt.close(self.fig)
            print("User quit early (progress saved).", file=sys.stderr)
            return

        ax = event.inaxes
        idx = self._axes_index(ax) if ax is not None else None

        if k == "r" and idx is not None:
            self._clear_panel(idx)
            return
        if k == "s" and idx is not None:
            self._skip_panel(idx)
            return
        if k == "a":
            self._clear_all()
            return
        if k == "b":
            self.show_bounds = not self.show_bounds
            for ov in self.boundary_overlays:
                if ov is not None:
                    ov.set_visible(self.show_bounds)
            self._refresh()
            return

    # -------- drawing helpers --------
    def _draw_marker(self, i, xi, yi, msg):
        ax = self.axes[i]
        if self.markers[i] is not None:
            self.markers[i].remove(); self.markers[i] = None
        if self.texts[i] is not None:
            self.texts[i].remove(); self.texts[i] = None

        self.markers[i] = ax.plot([xi],[yi], marker='o', markersize=9,
                                  markerfacecolor='none', markeredgecolor='w', lw=2)[0]
        self.texts[i] = ax.text(5, 15, msg, color='w', fontsize=5,
                                bbox=dict(facecolor='black', alpha=0.55))

    def _show_status(self, i, msg, color='w'):
        ax = self.axes[i]
        if self.texts[i] is not None:
            self.texts[i].remove(); self.texts[i] = None
        self.texts[i] = ax.text(5, 15, msg, color=color, fontsize=5,
                                bbox=dict(facecolor='black', alpha=0.55))
        self._refresh()

    def _clear_panel(self, i):
        self.x[i]=self.y[i]=np.nan
        self.label_id[i]=np.nan
        self.centroid_xy[i]=(np.nan, np.nan)
        self.area[i]=np.nan
        self.skipped[i]=False
        if self.markers[i] is not None:
            self.markers[i].remove(); self.markers[i]=None
        if self.texts[i] is not None:
            self.texts[i].remove(); self.texts[i]=None
        self.axes[i].set_title(self._panel_title(i), fontsize=5)
        self._refresh()

    def _skip_panel(self, i):
        self._clear_panel(i)
        self.skipped[i] = True
        self.axes[i].set_title(self._panel_title(i), fontsize=5, color='tab:orange')
        self._refresh()

    def _clear_all(self):
        for i in range(self.N):
            self._clear_panel(i)
            self.axes[i].set_title(self._panel_title(i), fontsize=5)
        self._refresh()

    # -------- persistence --------
    def _to_dataframe(self):
        rows = []
        for i in range(self.N):
            cy, cx = self.centroid_xy[i]
            rows.append(dict(
                frame=i,
                image_name=self.names[i],
                x=self.x[i], y=self.y[i],
                label_id=self.label_id[i],
                centroid_y=cy, centroid_x=cx,
                area=self.area[i],
                skipped=self.skipped[i],
            ))
        df = pd.DataFrame(rows).sort_values("frame")
        return df

    def _save_progress(self):
        df = self._to_dataframe()
        df.to_csv(self.out_csv, index=False)
        print(f"[info] Progress saved to: {self.out_csv.resolve()}")

    def _finalise_and_save(self):
        self._save_progress()
        # quick area-time plot (ignores skipped/NaN)
        try:
            df = self._to_dataframe()
            valid = df[~df['skipped'] & df['area'].notna()]
            if not valid.empty:
                plt.figure(figsize=(7.5,4.2))
                plt.plot(valid['frame'], valid['area'], marker='o')
                plt.xlabel("Frame"); plt.ylabel("Area (pixels)")
                plt.title("Selected clone area over time")
                plt.grid(True, alpha=0.3)
                plot_path = self.out_csv.with_suffix('.png')
                plt.tight_layout(); plt.savefig(plot_path, dpi=150)
                print(f"[info] Area plot saved to: {plot_path.resolve()}")
                plt.close()
        except Exception as e:
            print(f"[warn] Plotting failed: {e}", file=sys.stderr)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
    description="Class-aware clone picker with auto-discovered legend."
    )
    ap.add_argument("--root", required=True,
                    help="Root that contains csv_* folders with distance_*__preds_legend.csv and *_preds_labels.tif")
    ap.add_argument("--out_csv", default="clone_selections.csv")
    ap.add_argument("--px_size_um", type=float, default=None,
                    help="Pixel size (µm) for physical area reporting (optional).")
    args = ap.parse_args()

    root = Path(args.root)
    pairs = discover_all_csv_bundles(root)  # [(legend_csv, labels_tif, csv_dir), ...]
    if not pairs:
        sys.exit(f"[error] No distance_*__preds [legend,labels] pairs under {root}")

    mask_frames: list[np.ndarray] = []
    names: list[str] = []
    frame_palettes: list[dict[int, tuple[int,int,int]]] = []
    frame_idx2lab:  list[dict[int, str]] = []
    pre_region_map: list[dict[int,int]] = []

    for legend_path, labels_path, d in pairs:
        # read legend -> per-bundle palette & background set
        ldf = pd.read_csv(legend_path)
        palette, idx2lab, _, bg_indices = _legend_to_palette(ldf)

        # read class-index frames for this bundle
        idx_frames, idx_names = load_stack_any(labels_path)
        tag = d.name.replace("csv_", "")
        if not idx_names or len(idx_names) != len(idx_frames):
            idx_names = [f"{tag}::frame{i}" for i in range(len(idx_frames))]
        else:
            idx_names = [f"{tag}::{n}" for n in idx_names]

        # for each frame, derive instance labels and store per-frame legend/mapping
        for f, nm in zip(idx_frames, idx_names):
            f = np.asarray(f).astype(np.int64)
            inst, r2c = instances_from_class_index(f, bg_indices)
            mask_frames.append(inst)
            names.append(nm)
            pre_region_map.append(r2c)
            frame_palettes.append(palette)  # same palette for all frames in this bundle
            frame_idx2lab.append(idx2lab)

    # build picker with per-frame legend/mappings
    picker = MultiFrameClonePicker(
        masks=mask_frames,
        names=names,
        out_csv=Path(args.out_csv),
        raw_images=None,
        overlay_alpha=1.0,
        px_size_um=getattr(args, "px_size_um", None),
        frame_palettes=frame_palettes,
        frame_idx2lab=frame_idx2lab,
        precomputed_region_class=pre_region_map
    )
    picker.launch()



if __name__ == "__main__":
    main()
