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

PATCH_WIDTH = 100
PATCH_HEIGHT = 200
# Variable - currently following Bettega's previously defined region size
NUM_PATCHES_PER_REGION = 5
# currently testing on 5
LABELS = ["filiform", "foliate_L", "foliate_R"]
TILE_SIZE = 20  # size of coloured sub-tiles within each patch
TARGET_RATIO = 1.8

# File selection GUI, currently drag and drop for singular, can be extended later
def choose_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Brightfield Tongue Image",
        # main functionality just needs to support tif and tiff (TODO: haven't tested the rest..)
        filetypes=[("Image files", "*.tif *.tiff *.png *.jpg")]
    )
    return file_path

# GUI-based drag-to-select using RectangleSelector
def select_regions(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Select 3 rectangular ROIs - front (filiform), left & right (foliate)")
    
    selections = []
    
    def onselect(eclick, erelease):
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
            plt.close()

    selector = RectangleSelector(ax, onselect, useblit=True,
                                  button=[1],  
                                  interactive=True)
    plt.show()
    return selections

# randomised selection of [X] 100×200 patches from within ROI
# def sample_and_save_patches(image, region_coords, output_base="output"):
from skimage.draw import disk
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max

def get_circular_roi_mask(image_shape, centers, radius):
    # create boolean mask 
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

def adjust_to_ratio(patch, peak_mask, trough_mask, target_ratio=1.8):
    # Bettega defined the target ratio to be 1.8
    # thus we want to find the additive bias such that (peak_mean + b)\(trough_mean+b) = 1,8
    # TODO: CHECK THIS LATER!! b = (1.8*(trough - peak))/-0.8 ??
    patch = patch.astype(np.float32)
    peak_mean = compute_mean_intensity(patch, peak_mask)
    trough_mean = compute_mean_intensity(patch, trough_mask)
    if trough_mean == 0:
        return patch.astype(np.uint8)
    b = (target_ratio * trough_mean - peak_mean) / (1 - target_ratio)
    # now we add the bias to each pixel
    adjusted = np.clip(patch + b, 0, 255).astype(np.uint8)
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

def add_stippling(gt, ov, dots=5, r=2):
    im = Image.fromarray(ov)
    dr = ImageDraw.Draw(im)
    H, W = gt.shape
    rows, cols = H//TILE_SIZE, W//TILE_SIZE
    for r_idx in range(rows):
        for c_idx in range(cols):
            if gt[r_idx*TILE_SIZE, c_idx*TILE_SIZE] > 0:
                y0, x0 = r_idx*TILE_SIZE, c_idx*TILE_SIZE
                for _ in range(dots):
                    y = np.random.randint(y0, y0+TILE_SIZE)
                    x = np.random.randint(x0, x0+TILE_SIZE)
                    dr.ellipse([(x-r, y-r),(x+r,y+r)], fill=(0,255,0), outline=(0,255,0))
    return np.array(im)

def composite_overlay(patch, ov, alpha=0.5):
    # 1) ensure `base` is H×W×3
    if patch.ndim == 2:
        base = np.stack([patch]*3, axis=-1)
    elif patch.ndim == 3 and patch.shape[2] == 3:
        base = patch.copy()
    else:
        raise ValueError(f"Unsupported patch shape {patch.shape}")

    # 2) mask where overlay is nonzero
    mask2d = ov.sum(axis=-1) > 0

    # 3) compute the blended filter
    blended = (alpha * ov + (1 - alpha) * base).astype(np.uint8)

    # 4) copy blended pixels back into base only where mask is True
    for ch in range(3):
        bch = base[..., ch]
        bch[mask2d] = blended[..., ch][mask2d]
        base[..., ch] = bch

    return base

def sample_and_show_pseudoimages(image, regions):
    os.makedirs("output", exist_ok=True)
    # extract patches
    for idx,(xmin,ymin,xmax,ymax) in enumerate(regions):
        lbl = LABELS[idx]
        folder = f"output/{lbl}_{idx+1:03d}"
        os.makedirs(folder, exist_ok=True)
        if xmax-xmin<PATCH_WIDTH or ymax-ymin<PATCH_HEIGHT:
            continue
        for i in range(NUM_PATCHES_PER_REGION):
            x = random.randint(xmin, xmax-PATCH_WIDTH)
            y = random.randint(ymin, ymax-PATCH_HEIGHT)
            patch = image[y:y+PATCH_HEIGHT, x:x+PATCH_WIDTH]
            fname = f"{lbl}_patch_{i:02d}.tiff"
            Image.fromarray(patch).save(os.path.join(folder, fname))
            # adjust first patch
            if idx==0 and i==0:
                pmask, tmask, peaks = detect_peaks_and_troughs(patch)
                adj = adjust_to_ratio(patch, pmask, tmask)
                Image.fromarray(adj).save(os.path.join(folder, f"{lbl}_patch_{i:02d}_adj.tiff"))
    # now build pseudoimages
    fig, axs = plt.subplots(3, NUM_PATCHES_PER_REGION, figsize=(NUM_PATCHES_PER_REGION*2,6))
        # … inside sample_and_show_pseudoimages(), in the loop that builds the mosaic …

    for idx, label in enumerate(LABELS):
        folder = f"output/{label}_{idx+1:03d}"
        paths = sorted(glob.glob(folder + "/*.tiff"))[:NUM_PATCHES_PER_REGION]
        for j, path in enumerate(paths):
            patch = np.array(Image.open(path))

            # TODO: generate the coloured overlay as a mask (dictionary mask?)
            # use colour codes, R, Y, G, C, B (all can recombine, compute relative probabilities)
            # must be able to separately construct the masked channels too (red only, etc)
            # images would normally be collected in these separate channels 

            # 1) make two independent random 20×20 allele grids
            red_gt    = make_random_allele_grid(patch.shape)
            yellow_gt = make_random_allele_grid(patch.shape)

            # 2) build red overlay: 1→half red, 2→full red
            r_ov = np.zeros((*patch.shape, 3), np.uint8)
            r_ov[red_gt == 1] = (127, 0,   0)
            r_ov[red_gt == 2] = (255, 0,   0)

            # 3) build yellow overlay: 1→half yellow, 2→full yellow
            y_ov = np.zeros((*patch.shape, 3), np.uint8)
            y_ov[yellow_gt == 1] = (255,225, 224)
            y_ov[yellow_gt == 2] = (255,255, 0)

            # 4) combine (no unintended overlap unless both masks >0)
            combined_ov = (r_ov + y_ov).clip(0,255).astype(np.uint8)
            pseudo = patch.copy()
            for ov in (r_ov, y_ov):
                pseudo = composite_overlay(pseudo, ov, alpha=0.5)

            axs[idx,j].imshow(pseudo)
            axs[idx,j].axis("off")

    plt.tight_layout()
    plt.suptitle("Pseudoimages with randomised coloured overlay", y=1.02)
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

    # sample_and_save_patches(img, regions)
    # print("All patches saved successfully.")
    # show_mosaic("output")

if __name__ == "__main__":
    main()
