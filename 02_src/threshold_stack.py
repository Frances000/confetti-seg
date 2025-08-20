import argparse
from pathlib import Path

import numpy as np
import cv2
from tifffile import imwrite
from tqdm import tqdm
from skimage import util, feature, restoration, color
from skimage.filters import gabor, sobel_h, sobel_v
from skimage.filters.rank import entropy
from skimage.filters import scharr_h, scharr_v, prewitt_h, prewitt_v
from skimage.filters import gaussian, laplace
from skimage.morphology import disk, skeletonize

def read_image_u8(path: Path) -> np.ndarray:
    # Read image as single-channel uint8 (no RGB mixing).
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    # If already single-channel:
    if img.ndim == 2:
        arr = img
    else:
        # If RGB take first channel as is (no compositing).
        arr_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = util.img_as_ubyte(color.rgb2gray(arr_rgb))
    # Normalise dtype to uint8
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 257).astype(np.uint8)
    return util.img_as_ubyte(arr.astype(np.float32))

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

# -----------------------------
# Threshold helper funcitons
# -----------------------------

def otsu_binary(u8: np.ndarray) -> np.ndarray:
    # Global Otsu binarisation, returns uint8 {0,255}
    # 8-bit single channel
    if u8.dtype != np.uint8:
        u8 = util.img_as_ubyte(u8)
    _, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def percentile_binary(u8: np.ndarray, pct_top: float = 14.0) -> np.ndarray:
    # Percentile threshold: foreground = top `pct_top`% brightest pixels.
    # Threshold value is the (100 - pct_top)-th percentile of intensities.
    # Returns uint8 {0,255}.
    if u8.dtype != np.uint8:
        u8 = util.img_as_ubyte(u8)
    # Compute threshold at (100 - pct_top) percentile
    thr_val = float(np.percentile(u8, 100.0 - pct_top))
    mask = (u8 >= thr_val).astype(np.uint8) * 255
    return mask

# -----------------------------
# Distance map helper
# -----------------------------

def _dm_from_binary(mask_u8: np.ndarray) -> np.ndarray:
    # Distance to foreground (target set) like the IJ macro:
    #   invert mask so target (255) -> 0
    #   distanceTransform gives distance to nearest zero (i.e., to target)
    #   normalise to 0..255 for compact storage
    inv = cv2.bitwise_not(mask_u8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=5)
    return _u8_unit(_float_norm(dist))

def _skeletonize_u8(mask_u8: np.ndarray) -> np.ndarray:
    bin01 = (mask_u8 > 0)
    skel = skeletonize(bin01)
    return (skel.astype(np.uint8) * 255)

# -----------------------------
# Feature stack (approx TWS)
# -----------------------------

def features_for_channel(Iu8: np.ndarray) -> list[tuple[str, np.ndarray]]:

    # Compute a suite of features similar to the Trainable Weka Segmentation toggles:
    # Mean, Variance, Minimum, Maximum, Median,
    # Anisotropic diffusion (TV denoise proxy), Bilateral,
    # Lipschitz (edge magnitude proxy), Kuwahara (approx.),
    # Gabor (max over orientations), Derivatives X/Y,
    # Laplacian, Structure tensor eigvals (λ1, λ2),
    # Entropy, and 'Neighbors' at a few radii.
    # Returns list of (name, uint8 image).

    I = Iu8.astype(np.uint8)
    If = I.astype(np.float32) / 255.0
    feats = []
    # Local stats window
    k = 7
    kernel = np.ones((k, k), np.float32) / (k * k)
    # Mean
    local_mean = cv2.filter2D(If, -1, kernel, borderType=cv2.BORDER_REFLECT)
    feats.append(("Mean", _u8_unit(local_mean)))
    # Variance
    local_sqmean = cv2.filter2D(If**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    local_var = np.clip(local_sqmean - local_mean**2, 0, None)
    feats.append(("Variance", _u8_unit(_float_norm(local_var))))
    # Min/Max via morphology
    str_el = np.ones((k, k), np.uint8)
    feats.append(("Minimum", cv2.erode(I, str_el, borderType=cv2.BORDER_REFLECT)))
    feats.append(("Maximum", cv2.dilate(I, str_el, borderType=cv2.BORDER_REFLECT)))
    # Median
    feats.append(("Median", cv2.medianBlur(I, k if k % 2 else k + 1)))
    # Anisotropic diffusion 
    tv = restoration.denoise_tv_chambolle(If, weight=0.1)
    feats.append(("Anisotropic_diffusion", _u8_unit(tv)))
    # Bilateral
    feats.append(("Bilateral", cv2.bilateralFilter(I, d=7, sigmaColor=25, sigmaSpace=7)))
    # Lipschitz proxy = gradient magnitude
    gx = cv2.Sobel(If, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(If, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    feats.append(("Lipschitz", _u8_unit(_float_norm(grad_mag))))
    # Kuwahara approximation (simple region-preserving proxy)
    local_min = cv2.erode(I, str_el, borderType=cv2.BORDER_REFLECT)
    local_max = cv2.dilate(I, str_el, borderType=cv2.BORDER_REFLECT)
    kuwahara_proxy = ((local_min.astype(np.uint16) + local_max.astype(np.uint16)) // 2).astype(np.uint8)
    feats.append(("Kuwahara", kuwahara_proxy))
    # Gabor
    If_g = If  # avoid shadowing
    gabor_accum = np.zeros_like(If_g)
    for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
        for frequency in (0.1, 0.2, 0.4, 0.8):
            real, imag = gabor(If_g, frequency=frequency, theta=theta)
            gabor_accum = np.maximum(gabor_accum, np.hypot(real, imag))
            feats.append((f"Gabor {theta} {frequency}", _u8_unit(_float_norm(gabor_accum))))
    for sigma in (0, 1, 2, 4, 8, 16):
        if sigma > 0:
            img = gaussian(If, sigma=sigma, preserve_range=True)
        else:
            img = If
        # Sobel
        feats.append((f"SobelX_s{sigma}", _u8_unit(_float_norm(sobel_h(img)))))
        feats.append((f"SobelY_s{sigma}", _u8_unit(_float_norm(sobel_v(img)))))
        # Scharr
        feats.append((f"ScharrX_s{sigma}", _u8_unit(_float_norm(scharr_h(img)))))
        feats.append((f"ScharrY_s{sigma}", _u8_unit(_float_norm(scharr_v(img)))))
        # Prewitt
        feats.append((f"PrewittX_s{sigma}", _u8_unit(_float_norm(prewitt_h(img)))))
        feats.append((f"PrewittY_s{sigma}", _u8_unit(_float_norm(prewitt_v(img)))))
        # Derivatives
        feats.append(("Derivatives_X", _u8_unit(_float_norm(sobel_h(If)))))
        feats.append(("Derivatives_Y", _u8_unit(_float_norm(sobel_v(If)))))
    # Laplacian
    for ksize in (3, 5, 7):
        lap = cv2.Laplacian(I, cv2.CV_16S, ksize=ksize)
        feats.append(("Laplacian", cv2.convertScaleAbs(lap)))
    # Structure tensor eigenvalues
    A_elems = feature.structure_tensor(If, sigma=1.0)
    l1, l2 = feature.structure_tensor_eigenvalues(A_elems)
    feats.append(("Structure_lambda1", _u8_unit(_float_norm(l1))))
    feats.append(("Structure_lambda2", _u8_unit(_float_norm(l2))))
    # Entropy (local)
    for radii in (1, 2, 4, 8, 16):
        ent = entropy(I, disk(radii))
        feats.append((f"Entropy_r_{radii}", _u8_unit(_float_norm(ent))))
    # Neighbors at a few radii
    for r in (1, 2, 4, 8, 16):
        feats.append((f"Neighbors_r{r}", cv2.blur(I, (r, r))))

    return feats

def _read_mask_if_exists(img_path: Path) -> np.ndarray | None:
    """
    Try to read a mask image alongside `img_path`, named `mask_<basename>.<ext>`.
    Returns uint8 2D image or None if not found/unreadable.
    """
    mask_path = img_path.with_name(f"mask_{img_path.stem}{img_path.suffix}")
    if not mask_path.exists():
        return None

    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None

    # normalise to single-channel uint8
    if m.ndim == 3:
        # convert RGB/BGR -> gray
        if m.shape[2] == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        else:
            m = m[:, :, 0]
    if m.dtype == np.uint8:
        return m
    if m.dtype == np.uint16:
        return (m / 257).astype(np.uint8)
    return util.img_as_ubyte(m.astype(np.float32))

# -----------------------------
# Threshold and save
# -----------------------------

def process_image(img_path: Path, out_dir: Path, channel_tag: str, pct_top: float = 14.0):
    I = read_image_u8(img_path)
    # load mask image with sibling path
    mask_u8 = _read_mask_if_exists(img_path)
    # base planes: original + features
    base_planes: list[np.ndarray] = []
    base_labels: list[str] = []
    # append the base image
    if mask_u8 is not None:
        base_planes.append(mask_u8)
        base_labels.append("mask")
    else:
        print(f"[warn] no mask for {img_path.name}")
        return
    base_planes.append(I)
    base_labels.append("original")

    for fname, fimg in features_for_channel(I):
        base_planes.append(fimg)
        base_labels.append(f"{channel_tag}_{fname}")

    # Output planes & labels
    planes: list[np.ndarray] = []
    labels: list[str] = []

    # Begin to generate the distance-mapped variant
    for lbl, img in zip(base_labels, base_planes):
        img_u8 = img.astype(np.uint8)
        # 1) keep the base plane
        planes.append(img_u8)
        labels.append(lbl)

    # Output the base stack 
    stack = np.stack(planes, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"Base_{img_path.stem}_Feature-stack0001.tif"
    out_path = out_dir / out_name
    imwrite(
        str(out_path),
        stack,
        photometric="minisblack",
        metadata={"axes": "Z", "ImageDescription": "Feature stack (split channels) without binarisation/thresholding"},
        description="Base feature stack",
        contiguous=True,
    )
    with open(out_dir / f"Base_{img_path.stem}_Feature-stack0001_labels.txt", "w") as f:
        for i, tag in enumerate(labels, 1):
            f.write(f"{i:03d}: {tag}\n")

    # reset the output planes & labels
    planes: list[np.ndarray] = []
    labels: list[str] = []

    # Begin to generate the distance-mapped variant
    for lbl, img in zip(base_labels, base_planes):
        img_u8 = img.astype(np.uint8)

        # 1) keep the base plane
        planes.append(img_u8)
        labels.append(lbl)

        # If this plane is the mask we just prepend - no derivatory thresholding
        if lbl == "mask":
            continue

        # 2) make three binary planes (non-base)
        mask_p14 = percentile_binary(img_u8, pct_top=pct_top)
        mask_p50 = percentile_binary(img_u8, pct_top=50.0)
        mask_ots = otsu_binary(img_u8)

        planes.extend([mask_p14, mask_p50, mask_ots])
        labels.extend([f"{lbl}_Pctl{int(round(pct_top))}",
                       f"{lbl}_Pctl50",
                       f"{lbl}_Otsu"])

        # 3) create distance maps only from those binary 
        #    (thresh-only and skeleton versions)
        dm_p14   = _dm_from_binary(mask_p14)
        dm_p50   = _dm_from_binary(mask_p50)
        dm_ots   = _dm_from_binary(mask_ots)

        sk_p14   = _skeletonize_u8(mask_p14)
        sk_p50   = _skeletonize_u8(mask_p50)
        sk_ots   = _skeletonize_u8(mask_ots)

        dm_sk_p14 = _dm_from_binary(sk_p14)
        dm_sk_p50 = _dm_from_binary(sk_p50)
        dm_sk_ots = _dm_from_binary(sk_ots)

        planes.extend([dm_p14, dm_p50, dm_ots, dm_sk_p14, dm_sk_p50, dm_sk_ots])
        labels.extend([f"DM14pcThresh_{lbl}",
                       f"DM50pcThresh_{lbl}",
                       f"DMotsuThresh_{lbl}",
                       f"DM14pcSkel_{lbl}",
                       f"DM50pcSkel_{lbl}",
                       f"DMotsuSkel_{lbl}"])

    # save distance map variant of stack
    stack = np.stack(planes, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"Distance_{img_path.stem}_Feature-stack0001.tif"
    out_path = out_dir / out_name
    imwrite(
        str(out_path),
        stack,
        photometric="minisblack",
        metadata={"axes": "Z", "ImageDescription": "Feature stack (split channels) with Otsu & Percentile thresholds"},
        description="Feature stack with thresholds",
        contiguous=True,
    )
    with open(out_dir / f"Distance_{img_path.stem}_Feature-stack0001_labels.txt", "w") as f:
        for i, tag in enumerate(labels, 1):
            f.write(f"{i:03d}: {tag}\n")

def main():
    parser = argparse.ArgumentParser(description="Feature stacks for split channel folders C, R, G, Y, with threshold variants.")
    parser.add_argument("--source", required=True, help="Root folder containing C/, R/, G/, Y/ subfolders")
    parser.add_argument("--dest",   required=True, help="Destination root folder")
    parser.add_argument("--exts", nargs="*", default=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
                        help="Allowed file extensions")
    parser.add_argument("--pct_top", type=float, default=14.0,
                        help="Percent of brightest pixels to keep as foreground in percentile threshold (default 14.0)")
    args = parser.parse_args()

    src_root = Path(args.source)
    dst_root = Path(args.dest)
    exts = {e.lower() for e in args.exts}

    channels = ["C", "R", "G", "Y"]
    for ch in channels:
        in_dir = src_root / ch
        if not in_dir.is_dir():
            continue  
        # skip missing channels without raising errors
        files = sorted(p for p in in_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in exts)
        out_dir = dst_root / ch
        print(f"[{ch}] {len(files)} files -> {out_dir}")
        for p in tqdm(files, desc=f"Channel {ch}"):
            process_image(p, out_dir, ch, pct_top=args.pct_top)

if __name__ == "__main__":
    main()
