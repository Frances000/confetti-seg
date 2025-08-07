import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2
import random
from matplotlib.widgets import RectangleSelector
import glob

PATCH_WIDTH = 100
PATCH_HEIGHT = 200
# Variable - currently following Bettega's previously defined region size
NUM_PATCHES_PER_REGION = 10
# currently testing on 5
LABELS = ["filiform", "foliate_L", "foliate_R"]

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
def sample_and_save_patches(image, region_coords, output_base="output"):
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

    os.makedirs(output_base, exist_ok=True)
    image_pil = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)

    # loop over regions and label
    for idx, (xmin, ymin, xmax, ymax) in enumerate(region_coords):
        region_label = LABELS[idx]
        folder = os.path.join(output_base, f"{region_label}_{idx+1:03d}")
        os.makedirs(folder, exist_ok=True)

        region_width = xmax - xmin
        region_height = ymax - ymin

        if region_width < PATCH_WIDTH or region_height < PATCH_HEIGHT:
            print(f"Error - Region {region_label} too small for patch size, skipped.")
            continue
        # randomly choose patches to crop from the original array and save
        for i in range(NUM_PATCHES_PER_REGION):
            x = random.randint(xmin, xmax - PATCH_WIDTH)
            y = random.randint(ymin, ymax - PATCH_HEIGHT)
            patch = image[y:y+PATCH_HEIGHT, x:x+PATCH_WIDTH]
            patch_filename = f"{region_label}_patch_{x}_{y}.tiff"
            patch_path = os.path.join(folder, patch_filename)
            Image.fromarray(patch).save(patch_path)
            print(f"Saved: {patch_filename}")

            # VISUAL SANITY CHECK TODO: REMOVE THIS LATER
            if idx == 0 and i == 0:
                print(">> Performing peak:trough adjustment and visualisation on first patch.")

                peak_mask, trough_mask, peak_centers = detect_peaks_and_troughs(patch)
                adjusted = adjust_to_ratio(patch, peak_mask, trough_mask)

                # Save adjusted patch
                adjusted_filename = f"{region_label}_patch_{x}_{y}_adjusted.tiff"
                Image.fromarray(adjusted).save(os.path.join(folder, adjusted_filename))
                print(f"Adjusted version saved: {adjusted_filename}")

                # Show peak locations overlay
                fig, ax = plt.subplots()
                ax.imshow(patch, cmap='gray')
                for cy, cx in peak_centers:
                    ax.add_patch(plt.Circle((cx, cy), radius=2, color='red', fill=False))
                ax.set_title("Detected Peak Locations (red)")
                plt.axis('off')
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

    sample_and_save_patches(img, regions)
    print("All patches saved successfully.")
    show_mosaic("output")

if __name__ == "__main__":
    main()
