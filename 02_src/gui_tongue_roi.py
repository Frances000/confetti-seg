import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
from matplotlib.widgets import RectangleSelector
import glob
import itertools
import time

# ---------------------------
# Constants for configuration 
# ---------------------------

# Dimensions for sample (pixels), currently following Bettega's previously defined region size
PATCH_WIDTH = 100
PATCH_HEIGHT = 200
# Confetti allele code: red, yellow, green, cyan, black
ALLELES = ["R", "Y", "G", "C", "B"]
# All possible heterzygous and homozygous allele pairs 
PAIR_CODES = list(itertools.combinations_with_replacement(ALLELES, 2))
# Assignment of weights - 1 for homozygosity, 2 for heterozygosity
PAIR_WEIGHTS = np.array([1 if a == b else 2 for (a, b) in PAIR_CODES], dtype=float)
PAIR_WEIGHTS /= PAIR_WEIGHTS.sum()
# Number of randomly allocated patches
NUM_PATCHES_PER_REGION = 5
# Region labels
LABELS = ["filiform", "foliate_L", "foliate_R"]
# size of coloured sub-tiles within each patch - square
TILE_SIZE = 20  
TARGET_RATIO = 1.8
CHANNELS = ["R", "Y", "G", "C"]
# RGB mapping for channels
CHANNEL_RGB = {
    "R": (255, 0,   0),
    "Y": (255, 255, 0),
    "G": (0,   255, 0),
    "C": (0,   255, 255),
}

# ----------------------------
# File selection and ROI input
# ----------------------------

def choose_image():
    # open file dialogue, returns path to selected image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Brightfield Tongue Image",
        # main functionality just needs to support tif and tiff (TODO: haven't tested the rest..)
        filetypes=[("Image files", "*.tif *.tiff *.png *.jpg")]
    )
    return file_path

def select_regions(image):
    # GUI-based drag-to-select using RectangleSelector
    # input grayscale image (ndarray) and output selections as a list of tuples
    # tuple form: (xmin, ymin, xmax, ymax)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Select 3 rectangular ROIs - front (filiform), left & right (foliate)")
    selections = []
    
    def onselect(eclick, erelease):
        # Convert click/release coordinates to integers and order correctly
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        print(f"Selected region: ({xmin}, {ymin}) → ({xmax}, {ymax})")
        selections.append((xmin, ymin, xmax, ymax))
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        fig.canvas.draw()
        if len(selections) >= 3:
            # after selecting three regions, close the selector
            plt.close()
    selector = RectangleSelector(ax, onselect, useblit=True,
                                  button=[1],  
                                  interactive=True)
    plt.show()
    return selections

# ----------------------------
# ROI and patch processing
# ----------------------------
from skimage.draw import disk
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max

def get_circular_roi_mask(image_shape, centers, radius):
    # create boolean mask with filled circles at given coordinates
    mask = np.zeros(image_shape, dtype=bool)
    for cy, cx in centers:
        rr, cc = disk((cy, cx), radius, shape=image_shape)
        mask[rr, cc] = True
    return mask

def detect_peaks_and_troughs(patch, num_rois=5, radius=5):
    # cast patch to uint and find the n brightest pixels at least 2x the radius apart
    patch = patch.astype(np.uint8)
    h, w = patch.shape

    def find_extrema(coords_list, min_dist):
        selected = []
        for y, x in coords_list:
            if all(np.hypot(y - py, x - px) >= min_dist for py, px in selected):
                selected.append((y, x))
                if len(selected) == num_rois:
                    break
        return selected

    # srt all pixels by intensity descending (for peaks)
    flat_idx_desc = np.argsort(patch.ravel())[::-1]
    coords_desc = np.column_stack(np.unravel_index(flat_idx_desc, patch.shape))
    peak_centers = find_extrema(coords_desc, min_dist=radius*2)

    # sort all pixels by intensity ascending (for troughs)
    flat_idx_asc = np.argsort(patch.ravel())
    coords_asc = np.column_stack(np.unravel_index(flat_idx_asc, patch.shape))
    trough_centers = find_extrema(coords_asc, min_dist=radius*2)

    #construct masks - draw disk of radius r on the 0 array 
    peak_mask = np.zeros_like(patch, dtype=bool)
    trough_mask = np.zeros_like(patch, dtype=bool)

    for cy, cx in peak_centers:
        rr, cc = disk((cy, cx), radius, shape=patch.shape)
        peak_mask[rr, cc] = True

    for cy, cx in trough_centers:
        rr, cc = disk((cy, cx), radius, shape=patch.shape)
        trough_mask[rr, cc] = True

    return peak_mask, trough_mask, peak_centers

def compute_mean_intensity(img, mask):
    # mean of peaks
    return img[mask].mean()

def sample_unordered_pair():
    idx = np.random.choice(len(PAIR_CODES), p=PAIR_WEIGHTS)
    return PAIR_CODES[idx]

def generate_channel_masks(shape, tile_size):
    H, W = shape
    rows, cols = H // tile_size, W // tile_size

    masks = {ch: np.zeros((H, W), dtype=np.uint8) for ch in CHANNELS}

    for r in range(rows):
        for c in range(cols):
            a1, a2 = sample_unordered_pair()
            counts = {ch: 0 for ch in CHANNELS}
            # count alleles into channels; 'B' contributes to none
            for a in (a1, a2):
                if a in counts:
                    counts[a] += 1

            y0, x0 = r * tile_size, c * tile_size
            for ch in CHANNELS:
                masks[ch][y0:y0+tile_size, x0:x0+tile_size] = counts[ch]

    return masks

def adjust_to_ratio(patch, peak_mask, trough_mask, target_ratio=1.8):
    patch = patch.astype(np.float32)
    peak_mean = compute_mean_intensity(patch, peak_mask)
    trough_mean = compute_mean_intensity(patch, trough_mask)
    # print(f"trough_mean: {trough_mean}, peak_mean: {peak_mean}")
    if trough_mean == 0:
        return patch.astype(np.uint8)
    b = (target_ratio * trough_mean - peak_mean) / (1 - target_ratio)
    adjusted = patch + b
    # Rescale to full [0,255] range
    min_val = adjusted.min()
    max_val = adjusted.max()
    if min_val != max_val:  
        # avoid division by zero
        adjusted = (adjusted - min_val) * (255.0 / (max_val - min_val))
    else:
        adjusted.fill(0)  
        # black background value
    adjusted = adjusted.astype(np.uint8)
    # print(f"adjusted patch:\n{adjusted}")
    return adjusted

def make_random_allele_grid(shape, tile_size=TILE_SIZE):
    H, W = shape
    rows, cols = H // tile_size, W // tile_size
    codes = ["B","R","Y","RR","RY","YY"]
    layout = np.random.choice(codes, (rows, cols))
    cmap = {"B":0,"R":1,"Y":2,"RR":2,"RY":1,"YY":2}
    mask = np.zeros((H, W), int)
    for r in range(rows):
        for c in range(cols):
            val = cmap[layout[r, c]]
            y0, x0 = r*tile_size, c*tile_size
            mask[y0:y0+tile_size, x0:x0+tile_size] = val
    return mask

def code_to_overlay(mask, color):
    H, W = mask.shape
    ov = np.zeros((H, W, 3), np.uint8)
    for code, inten in ((1,127),(2,255)):
        ov[mask==code] = np.array(color)*inten
    return ov

def add_stippling_green(gt, ov, dots_per_tile=20, jitter=0.25):

    im = ov.copy()
    H, W = gt.shape
    rows, cols = H // TILE_SIZE, W // TILE_SIZE

    # choose a near-square grid
    if dots_per_tile == 20:
        gy, gx = 4, 5
    else:
        gy = max(1, int(np.floor(np.sqrt(dots_per_tile))))
        gx = int(np.ceil(dots_per_tile / gy))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if gt[r_idx * TILE_SIZE, c_idx * TILE_SIZE] > 0:
                y0, x0 = r_idx * TILE_SIZE, c_idx * TILE_SIZE

                # clear green fill in tile so only stipples remain
                im[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE, 1] = 0

                cell_h = TILE_SIZE / gy
                cell_w = TILE_SIZE / gx
                count = 0
                for i in range(gy):
                    for j in range(gx):
                        if count >= dots_per_tile:
                            break
                        cy = y0 + (i + 0.5) * cell_h
                        cx = x0 + (j + 0.5) * cell_w
                        dy = (np.random.rand() - 0.5) * 2 * jitter * cell_h
                        dx = (np.random.rand() - 0.5) * 2 * jitter * cell_w
                        y = int(np.clip(round(cy + dy), y0, y0 + TILE_SIZE - 1))
                        x = int(np.clip(round(cx + dx), x0, x0 + TILE_SIZE - 1))

                        # set green channel pixel
                        rr, cc = disk((y, x), 1.5, shape=im.shape[:2])
                        im[rr, cc, 1] = 255
                        count += 1

    return im

def composite_overlay(patch, ov, alpha=0.5):
    if patch.ndim == 2:
        base = np.stack([patch]*3, axis=-1)
    elif patch.ndim == 3 and patch.shape[2] == 3:
        base = patch.copy()
    else:
        raise ValueError(f"Unsupported patch shape {patch.shape}")
    mask2d = ov.sum(axis=-1) > 0
    blended = (alpha * ov + (1 - alpha) * base).astype(np.uint8)
    base[mask2d] = blended[mask2d]
    return base

def build_channel_overlay(mask, rgb_triplet, half_inten=127, full_inten=255):
    H, W = mask.shape
    ov = np.zeros((H, W, 3), np.uint8)
    # half intensity for 1 allele, full for 2
    ov[mask == 1] = np.array(rgb_triplet) * (half_inten // 255)
    ov[mask == 2] = np.array(rgb_triplet) * (full_inten // 255)
    ov = np.zeros((H, W, 3), np.uint8)
    for inten, scale in ((1, half_inten), (2, full_inten)):
        ov[mask == inten] = (np.array(rgb_triplet) * (scale/255.0)).astype(np.uint8)
    return ov

def sample_and_show_pseudoimages(image, regions):
    start_time = time.time()
    os.makedirs("output", exist_ok=True)
    for idx, (xmin, ymin, xmax, ymax) in enumerate(regions):
        lbl = LABELS[idx]
        folder = f"output/{lbl}_{idx+1:03d}"
        os.makedirs(folder, exist_ok=True)
        if xmax - xmin < PATCH_WIDTH or ymax - ymin < PATCH_HEIGHT:
            continue
        for i in range(NUM_PATCHES_PER_REGION):
            x = random.randint(xmin, xmax - PATCH_WIDTH)
            y = random.randint(ymin, ymax - PATCH_HEIGHT)
            patch = image[y:y+PATCH_HEIGHT, x:x+PATCH_WIDTH]
            pmask, tmask, peaks = detect_peaks_and_troughs(patch)
            patch = adjust_to_ratio(patch, pmask, tmask)
            Image.fromarray(patch).save(os.path.join(folder, f"{lbl}_patch_{i:02d}.tiff"))
    fig, axs = plt.subplots(3, NUM_PATCHES_PER_REGION, figsize=(NUM_PATCHES_PER_REGION*2, 6))

    for idx, label in enumerate(LABELS):
        folder = f"output/{label}_{idx+1:03d}"
        paths = sorted(glob.glob(folder + "/*.tiff"))[:NUM_PATCHES_PER_REGION]
        for j, path in enumerate(paths):
            patch = np.array(Image.open(path))
            # 1) per-channel allele count masks
            ch_masks = generate_channel_masks(patch.shape[:2], TILE_SIZE)
            # 2) save each channel as a separate image layer (simulated acquisition)
            for ch in CHANNELS:
                # map 0/1/2 to 0/127/255 grayscale for that channel layer
                ov = build_channel_overlay(ch_masks[ch], CHANNEL_RGB[ch])
                if ch == "G":  # apply stippling only to green
                    ov = add_stippling_green(ch_masks[ch], ov, dots_per_tile=20, jitter=0.25)
                coloured_patch = composite_overlay(patch, ov)  
                Image.fromarray(coloured_patch).save(
                    os.path.join(folder, f"{label}_patch_{j:02d}_{ch}.tiff")
                )
            # 3) build colour overlays for each channel and composite them
            pseudo = patch.copy()
            for ch in CHANNELS:
                ov = build_channel_overlay(ch_masks[ch], CHANNEL_RGB[ch])
                if ch == "G":  
                    # apply stippling only to green
                    ov = add_stippling_green(ch_masks[ch], ov, dots_per_tile=20, jitter=0.25)
                pseudo = composite_overlay(pseudo, ov, alpha=0.5)
            Image.fromarray(pseudo).save(
                os.path.join(folder, f"{label}_patch_{j:02d}_composite.tiff"))
            axs[idx, j].imshow(pseudo)
            axs[idx, j].axis("off")

    plt.tight_layout()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Image generation complete in {elapsed:.2f} seconds.")
    plt.suptitle("Pseudoimages with channel-wise Confetti allele simulation", y=1.02)
    plt.show()


def show_mosaic(output_base="output"):
    fig, axes = plt.subplots(3, NUM_PATCHES_PER_REGION, figsize=(NUM_PATCHES_PER_REGION * 2, 6))
    fig.suptitle("Cropped Patches by Region", fontsize=14)
    
    for i, label in enumerate(LABELS):
        folder = os.path.join(output_base, f"{label}_{i+1:03d}")
        patch_paths = sorted(glob.glob(os.path.join(folder, "*.tiff")))[:NUM_PATCHES_PER_REGION]
        
        for j, ax in enumerate(axes[i]):
            if j < len(patch_paths):
                img = Image.open(patch_paths[j])
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{label} {j+1}")
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def main():
    image_path = choose_image()
    if not image_path:
        print("No image selected.")
        return

    img = mpimg.imread(image_path)
    if img.ndim == 3:
        img = img[:, :, 0]  
        # convert to grayscale

    regions = select_regions(img)
    if len(regions) != 3:
        print("Exactly 3 regions must be selected.")
        return
    sample_and_show_pseudoimages(img, regions)

if __name__ == "__main__":
    main()
