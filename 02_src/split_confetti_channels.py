#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tifffile import TiffFile, imwrite

CANON = {"C","R","G","Y","BF"}  # BF = brightfield/gray/composite-ish

def central_crop(arr2d: np.ndarray, crop_h: int = 500, crop_w: int = 600) -> np.ndarray:
    H, W = arr2d.shape[:2]
    if crop_h > H or crop_w > W:
        raise ValueError(f"Crop {crop_h}x{crop_w} exceeds image {H}x{W}")
    r0 = (H - crop_h) // 2
    c0 = (W - crop_w) // 2
    return arr2d[r0:r0+crop_h, c0:c0+crop_w]

def parse_chnum_map(s: Optional[str]) -> Dict[int, str]:
    """'1=R,2=G,3=BF,4=C,5=Y' -> {1:'R', 2:'G', ...}"""
    if not s: return {}
    out: Dict[int, str] = {}
    for part in s.split(","):
        if not part.strip(): continue
        k, v = [t.strip() for t in part.split("=", 1)]
        v = v.upper()
        if v not in CANON:
            raise ValueError(f"Invalid channel code '{v}' in map; allowed {sorted(CANON)}")
        out[int(k)] = v
    return out

def _lut_to_code(lut: np.ndarray) -> str:
    """
    Map an ImageJ LUT (shape ~ (3,256) or (4,256)) to one of R,G,C,Y,BF.
    Heuristic: look at average LUT intensities per RGB channel.
    """
    lut = np.asarray(lut)
    if lut.ndim != 2:  # (channels, 256)
        return "BF"
    # ensure 3 rows (RGB); ignore alpha if present
    if lut.shape[0] >= 3:
        R = lut[0].astype(float).mean()
        G = lut[1].astype(float).mean()
        B = lut[2].astype(float).mean()
    else:
        return "BF"

    # normalise
    m = max(R, G, B, 1e-6)
    r, g, b = R/m, G/m, B/m

    # classify
    if r > 0.75 and g < 0.35 and b < 0.35:  # red
        return "R"
    if g > 0.75 and r < 0.35 and b < 0.35:  # green
        return "G"
    if g > 0.6 and b > 0.6 and r < 0.4:     # cyan
        return "C"
    if r > 0.6 and g > 0.6 and b < 0.4:     # yellow
        return "Y"
    return "BF"

def infer_plane_codes_from_imagej(tf: TiffFile) -> List[str]:
    """Return per-plane codes using ImageJ metadata LUTs if present; else empty list."""
    ij = tf.imagej_metadata or {}
    luts = ij.get("LUTs") or []
    if not luts:  # sometimes LUTs hang off pages
        luts = [getattr(p, "colormap", None) for p in tf.pages]
    codes: List[str] = []
    for lut in luts:
        if lut is None:
            codes.append("BF")
        else:
            # tifffile gives LUTs as (3, 256) uint16 or uint8
            if lut.dtype != np.uint8:
                lut8 = (lut.astype(np.float32) / max(lut.max(), 1)) * 255.0
                lut8 = lut8.astype(np.uint8)
            else:
                lut8 = lut
            codes.append(_lut_to_code(lut8))
    # If number of LUTs != pages, return empty (unreliable)
    return codes if len(codes) == len(tf.pages) else []

def infer_plane_codes_from_ome(tf: TiffFile) -> List[str]:
    """
    Use OME-XML channel Name or Color to map planes.
    Returns [] if not OME or mapping unreliable.
    """
    try:
        xml = tf.ome_metadata or ""
    except Exception:
        return []
    if not xml:
        return []
    # very light-weight parse to extract <Channel Name="..." Color="...">
    # We won't fully parse XML; simple regex suffices for typical files.
    chan_tags = re.findall(r"<Channel[^>]*?(Name|Name=)\"([^\"]*)\"[^>]*?>", xml)
    colors = re.findall(r"Color=\"(\d+)\"", xml)  # decimal RGB like 16711680
    names  = re.findall(r"<Channel[^>]*?Name=\"([^\"]+)\"[^>]*?>", xml)
    codes: List[str] = []
    # prefer names first
    for nm in names:
        n = nm.lower()
        if "red" in n:   codes.append("R")
        elif "green" in n: codes.append("G")
        elif "cyan" in n: codes.append("C")
        elif "yellow" in n: codes.append("Y")
        elif "bf" in n or "bright" in n or "gray" in n or "grey" in n:
            codes.append("BF")
        else:
            codes.append("BF")
    if codes:
        return codes
    # fallback: colors as 0xRRGGBB decimal
    for c in colors:
        try:
            val = int(c)
            R = (val >> 16) & 0xFF
            G = (val >> 8) & 0xFF
            B = (val >> 0) & 0xFF
            codes.append(_lut_to_code(np.array([[R]*256,[G]*256,[B]*256], dtype=np.uint8)))
        except Exception:
            codes.append("BF")
    return codes if codes else []

def load_planes(path: Path) -> Tuple[List[np.ndarray], str]:
    """Return list of 2D planes and dtype name."""
    with TiffFile(path) as tf:
        pages = list(tf.pages)
        planes: List[np.ndarray] = []
        for p in pages:
            arr = p.asarray()
            if arr.ndim == 3 and arr.shape[-1] in (3,4):
                # an RGB(A) plane; split channels to three 2D planes
                for c in range(3):
                    planes.append(arr[..., c])
            elif arr.ndim == 2:
                planes.append(arr)
            else:
                raise ValueError(f"Unsupported page shape {arr.shape} in {path.name}")
        dtype = str(planes[0].dtype)
    return planes, dtype

def infer_codes_for_planes(path: Path, chnum_map: Dict[int,str]) -> List[str]:
    """
    Try OME names -> ImageJ LUTs -> numeric ch mapping -> default 'BF'.
    Returns list of codes length == #planes.
    """
    with TiffFile(path) as tf:
        codes = infer_plane_codes_from_ome(tf)
        if not codes:
            codes = infer_plane_codes_from_imagej(tf)
        if not codes:
            # numeric fallback: chnum_map maps 1-based indices
            codes = [chnum_map.get(i+1, "BF") for i in range(len(tf.pages))]
        # If RGB page expanded into 3 planes, above pages length was 1; handle later
    # If RGB expansion happened in load_planes(), align codes length
    planes, _ = load_planes(path)
    if len(codes) != len(planes):
        # best-effort: repeat last code or fill BF
        if len(codes) == 1:
            codes = codes * len(planes)
        else:
            codes = (codes + ["BF"] * len(planes))[:len(planes)]
    # final sanity to allowed set
    codes = [c if c in CANON else "BF" for c in codes]
    return codes

def ensure_dirs(root: Path, include_bf: bool):
    for d in ["C","R","G","Y"] + (["BF"] if include_bf else []):
        (root / d).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Split composite Confetti TIFF into C/R/G/Y (and optional BF), crop center, and save to dest_root/<channel>/.")
    ap.add_argument("--image", required=True, help="Path to composite TIFF (e.g., *-merge.tif).")
    ap.add_argument("--dest_root", required=True, help="Destination root containing C,R,G,Y (and optional BF) subfolders.")
    ap.add_argument("--crop", default="300x600", help="Central crop HxW, default 300x600. Use 'none' to disable.")
    ap.add_argument("--chnum_map", default="", help="Optional numeric channel mapping, e.g. '1=R,2=G,3=BF,4=C,5=Y'.")
    ap.add_argument("--include_bf", action="store_true", help="Also write BF/ (brightfield/gray) planes.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist.")
    ap.add_argument("--report_json", default="", help="Optional path to write a JSON mapping report.")
    args = ap.parse_args()

    image = Path(args.image)
    dest = Path(args.dest_root)
    ensure_dirs(dest, include_bf=args.include_bf)

    crop = None if args.crop.lower()=="none" else tuple(map(int, args.crop.lower().replace("x",",").split(",")))
    chnum_map = parse_chnum_map(args.chnum_map)

    # Load planes & infer codes
    planes, dtype_name = load_planes(image)
    codes = infer_codes_for_planes(image, chnum_map)

    stem = image.stem
    written: List[Tuple[str,str]] = []  # (code, path)
    counts: Dict[str,int] = {}

    for i, (plane, code) in enumerate(zip(planes, codes), start=1):
        if code not in {"C","R","G","Y","BF"}:
            code = "BF"
        if code == "BF" and not args.include_bf:
            continue

        arr = plane
        if crop is not None:
            arr = central_crop(arr, crop_h=crop[0], crop_w=crop[1])

        # construct path; if multiple per code, suffix _i
        counts[code] = counts.get(code, 0) + 1
        suffix = "" if counts[code] == 1 else f"_{counts[code]:02d}"
        out_path = dest / code / f"{stem}_{code}{suffix}.tif"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Exists: {out_path}. Use --overwrite to replace.")

        imwrite(str(out_path), arr, photometric="minisblack")
        written.append((code, str(out_path)))

    print(f"Wrote {len(written)} planes to {dest}")
    for code, path in written:
        print(f"  [{code}] {path}")

    if args.report_json:
        report = {
            "image": str(image),
            "dtype": dtype_name,
            "planes": len(planes),
            "mapping": [{"index": i+1, "code": c} for i, c in enumerate(codes)],
            "written": [{"code": c, "path": p} for c, p in written],
        }
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report: {args.report_json}")

if __name__ == "__main__":
    main()
