import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
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
    os.makedirs(output_base, exist_ok=True)
    image_pil = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(region_coords):
        label = LABELS[idx]
        folder = os.path.join(output_base, f"{label}_{idx+1:03d}")
        os.makedirs(folder, exist_ok=True)

        region_width = xmax - xmin
        region_height = ymax - ymin

        if region_width < PATCH_WIDTH or region_height < PATCH_HEIGHT:
            print(f"Error - Region {label} too small for patch size, skipped.")
            continue

        for i in range(NUM_PATCHES_PER_REGION):
            x = random.randint(xmin, xmax - PATCH_WIDTH)
            y = random.randint(ymin, ymax - PATCH_HEIGHT)
            patch = image_pil.crop((x, y, x + PATCH_WIDTH, y + PATCH_HEIGHT))
            filename = f"{label}_patch_{x}_{y}.tiff"
            patch.save(os.path.join(folder, filename))
            print(f"Saved: {filename}")

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
        img = img[:, :, 0]  # convert to grayscale if RGB

    regions = select_regions(img)
    if len(regions) != 3:
        print("Exactly 3 regions must be selected.")
        return

    sample_and_save_patches(img, regions)
    print("All patches saved successfully.")
    show_mosaic("output")

if __name__ == "__main__":
    main()
