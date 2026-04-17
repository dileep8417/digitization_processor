"""
Embroidery Preprocessing Pipeline
==================================
Converts LLM-generated cartoon-like portraits into embroidery-friendly images
suitable for auto-digitizing tools (e.g., Wilcom).

Design principle:
  - Outlines = structure (dark strokes, preserved exactly as black)
  - Whites = highlights (teeth, eye whites, preserved as pure white)
  - Colors = fill (quantized, cleaned, flattened)
  - Background removal via rembg (on by default)

Dependencies: OpenCV (cv2), NumPy
Optional:     rembg[cpu] (for background removal)
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "width": 512,
    "K": 6,                        # fill colors (more = better nose/cheek depth)
    "bilateral_d": 3,
    "bilateral_sigma_color": 30,
    "bilateral_sigma_space": 30,
    "morph_kernel_size": 3,
    "dark_threshold": 80,
    "white_threshold": 200,        # bright pixel threshold for eye whites/teeth
    "remove_bg": True,             # use rembg for background removal
    "bg_alpha_cutoff": 128,
    "debug": False,
    "debug_dir": "debug_output",
}

_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Stage 1 – Resize
# ---------------------------------------------------------------------------
def resize_image(image: np.ndarray, width: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Stage 2 – Extract outline mask
# ---------------------------------------------------------------------------
def extract_outline_mask(image: np.ndarray, dark_threshold: int = 80) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray < dark_threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Stage 3 – Extract white highlight mask (teeth, eye whites)
# ---------------------------------------------------------------------------
def extract_white_mask(image: np.ndarray, white_threshold: int = 200) -> np.ndarray:
    """
    Detect eye whites and teeth: small bright regions tightly enclosed
    by dark outlines. White clothing is excluded because those regions
    are larger and not tightly surrounded by dark pixels.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = (gray > white_threshold).astype(np.uint8) * 255
    dark = (gray < 80).astype(np.uint8) * 255

    # Flood fill from corners to exclude background bright pixels
    bg_bright = np.zeros((h, w), dtype=np.uint8)
    for seed in [(5, 5), (w - 5, 5), (5, h - 5), (w - 5, h - 5)]:
        if bright[seed[1], seed[0]] > 0:
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(bright.copy(), flood_mask, seed, 255,
                          (0,), (0,),
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            bg_bright = cv2.bitwise_or(bg_bright, flood_mask[1:-1, 1:-1])

    enclosed = cv2.bitwise_and(bright, cv2.bitwise_not(bg_bright))

    # For each enclosed region, check if it's tightly surrounded by dark outlines
    # (eyes/teeth are small bright patches inside dark strokes)
    max_area = int(0.005 * h * w)  # eyes/teeth are tiny
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        enclosed, connectivity=8
    )
    result = np.zeros((h, w), dtype=np.uint8)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= max_area:
            continue

        # Dilate the region slightly and check how much dark outline surrounds it
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        dilated = cv2.dilate(comp_mask, np.ones((5, 5), np.uint8), iterations=2)
        border = (dilated > 0) & (comp_mask == 0)  # ring around the region

        if np.sum(border) == 0:
            continue

        # Ratio of dark pixels in the surrounding border
        dark_ratio = np.sum((dark > 0) & border) / np.sum(border)

        # Eye whites/teeth have >30% dark outline in their immediate border
        if dark_ratio > 0.3:
            result[labels == lbl] = 255

    return result


# ---------------------------------------------------------------------------
# Stage 4 – Build color layer
# ---------------------------------------------------------------------------
def build_color_layer(image: np.ndarray, protected_mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(image, protected_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Stage 5 – Bilateral filter
# ---------------------------------------------------------------------------
def apply_bilateral_filter(image: np.ndarray, d: int = 5,
                           sigma_color: int = 40, sigma_space: int = 40) -> np.ndarray:
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Stage 6 – K-means color quantization
# ---------------------------------------------------------------------------
def kmeans_quantization(image: np.ndarray, K: int = 5,
                        fg_mask: np.ndarray | None = None) -> np.ndarray:
    """Quantize in CIELAB space for perceptually meaningful clusters.
    LAB separates skin tones better than BGR — subtle lip/cheek/shadow
    differences that matter visually get their own clusters instead of
    being merged with near-identical grays.

    If fg_mask is provided, only foreground pixels are used for clustering.
    Background pixels are then assigned to the nearest cluster."""
    h, w, c = image.shape
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    all_pixels = lab.reshape(-1, 3).astype(np.float32)

    if fg_mask is not None:
        fg_flat = fg_mask.flatten() > 0
        cluster_pixels = all_pixels[fg_flat]
    else:
        fg_flat = None
        cluster_pixels = all_pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(_RNG_SEED)
    _, labels, centers = cv2.kmeans(cluster_pixels, K, None, criteria, 10,
                                     cv2.KMEANS_PP_CENTERS)

    if fg_flat is not None:
        # Assign ALL pixels (including bg) to nearest center
        all_labels = np.zeros(all_pixels.shape[0], dtype=np.int32)
        min_dist = np.full(all_pixels.shape[0], np.inf, dtype=np.float32)
        for i, center in enumerate(centers):
            dist = np.sum((all_pixels - center) ** 2, axis=1)
            closer = dist < min_dist
            all_labels[closer] = i
            min_dist[closer] = dist[closer]
    else:
        all_labels = labels.flatten()

    quantized_lab = np.uint8(centers)[all_labels].reshape(h, w, 3)
    return cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Stage 7 – Remove small noisy regions
# ---------------------------------------------------------------------------
def remove_small_regions(image: np.ndarray, min_area: int = 500) -> np.ndarray:
    h, w, c = image.shape
    flat = image.reshape(-1, c)
    color_ids = (flat[:, 0].astype(np.int32) * 65536 +
                 flat[:, 1].astype(np.int32) * 256 +
                 flat[:, 2].astype(np.int32))
    color_id_map = color_ids.reshape(h, w)
    result = image.copy()

    for uid in np.unique(color_id_map):
        mask = np.uint8(color_id_map == uid) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                continue
            comp_mask = labels == lbl
            dilated = cv2.dilate(comp_mask.astype(np.uint8) * 255,
                                 np.ones((5, 5), np.uint8), iterations=2)
            neighbor_mask = (dilated > 0) & ~comp_mask
            if not np.any(neighbor_mask):
                continue
            nc = result[neighbor_mask]
            nc_enc = (nc[:, 0].astype(np.int64) * 65536 +
                      nc[:, 1].astype(np.int64) * 256 +
                      nc[:, 2].astype(np.int64))
            values, counts = np.unique(nc_enc, return_counts=True)
            d = values[np.argmax(counts)]
            result[comp_mask] = np.array(
                [(d >> 16) & 0xFF, (d >> 8) & 0xFF, d & 0xFF], dtype=np.uint8
            )
    return result


# ---------------------------------------------------------------------------
# Stage 8 – Morphological closing
# ---------------------------------------------------------------------------
def apply_morphology(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# ---------------------------------------------------------------------------
# Stage 9 – Merge near-duplicate colors
# ---------------------------------------------------------------------------
def merge_similar_colors(image: np.ndarray, min_distance: int = 12) -> np.ndarray:
    """Merge perceptually similar colors using LAB Delta-E distance,
    consistent with the LAB-space K-means quantization."""
    unique_bgr = np.unique(image.reshape(-1, 3), axis=0)
    if len(unique_bgr) <= 1:
        return image

    # Convert to LAB for perceptual comparison
    unique_bgr_3d = unique_bgr.reshape(1, -1, 3).astype(np.uint8)
    unique_lab = cv2.cvtColor(unique_bgr_3d, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    merge_to = {}
    used = set()
    for i in range(len(unique_lab)):
        k1 = tuple(unique_bgr[i].astype(int))
        if k1 in used:
            continue
        used.add(k1)
        for j in range(i + 1, len(unique_lab)):
            k2 = tuple(unique_bgr[j].astype(int))
            if k2 in used:
                continue
            if np.linalg.norm(unique_lab[i] - unique_lab[j]) < min_distance:
                merge_to[k2] = k1
                used.add(k2)

    if not merge_to:
        return image
    result = image.copy()
    for src, dst in merge_to.items():
        mask = np.all(result == np.array(src, dtype=np.uint8), axis=2)
        result[mask] = np.array(dst, dtype=np.uint8)
    return result

def normalize_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    kernel = np.ones((thickness, thickness), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

# ---------------------------------------------------------------------------
# Stage 11 – Composite layers
# ---------------------------------------------------------------------------
def composite_final(color_layer: np.ndarray, outline_mask: np.ndarray,
                    white_mask: np.ndarray) -> np.ndarray:
    result = color_layer.copy()
    result[white_mask > 0] = [255, 255, 255]
    result[outline_mask > 0] = [0, 0, 0]
    return result


# ---------------------------------------------------------------------------
# Stage 12 – Background removal via rembg
# ---------------------------------------------------------------------------
def remove_background(processed: np.ndarray, original: np.ndarray,
                      outline_mask: np.ndarray, white_mask: np.ndarray,
                      alpha_cutoff: int = 128) -> np.ndarray:
    """
    Remove background using rembg on the ORIGINAL image (better shading cues),
    then apply the alpha mask to the processed image with palette snapping.
    """
    try:
        from rembg import remove as rembg_remove
        from PIL import Image
    except ImportError:
        raise ImportError(
            'rembg is required for background removal. '
            'Install with: pip install "rembg[cpu]"'
        )

    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pil_result = rembg_remove(Image.fromarray(rgb))
    rgba = np.array(pil_result)

    alpha = np.where(rgba[:, :, 3] >= alpha_cutoff, 255, 0).astype(np.uint8)
    bgr = processed.copy()

    # Safety check: if rembg removed too much (>85% transparent),
    # it likely ate the subject — fall back to no bg removal
    transparent_pct = np.sum(alpha == 0) / alpha.size
    if transparent_pct > 0.85:
        print(f"Warning: rembg removed {transparent_pct*100:.0f}% of image "
              f"(likely white clothing on white bg). Skipping bg removal.")
        return cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)

    # Re-burn protected layers on opaque pixels
    opaque = alpha == 255
    bgr[(white_mask > 0) & opaque] = [255, 255, 255]
    bgr[(outline_mask > 0) & opaque] = [0, 0, 0]

    return np.dstack([bgr, alpha])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_output(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def _debug_save(image: np.ndarray, debug_dir: str, name: str) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), image,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, "images")
_GENERATED_DIR = os.path.join(_SCRIPT_DIR, "generated")


def _resolve_input_path(name: str) -> str:
    """Resolve an image name to its full path in the images/ folder."""
    # If it's already a full/relative path that exists, use it
    if os.path.isfile(name):
        return name

    # Try images/<name> directly
    candidate = os.path.join(_IMAGES_DIR, name)
    if os.path.isfile(candidate):
        return candidate

    # Try images/<name>.png, .jpg, .jpeg
    base, ext = os.path.splitext(name)
    if not ext:
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(_IMAGES_DIR, base + ext)
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"Image not found: tried '{name}' and images/{name}[.png|.jpg|.jpeg]"
    )


def _default_output_path(input_path: str) -> str:
    """Generate output path: generated/<name>_embroidery.png"""
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(_GENERATED_DIR, exist_ok=True)
    return os.path.join(_GENERATED_DIR, f"{base}_embroidery.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_image(input_path: str, output_path: str | None = None,
                  config: dict | None = None) -> np.ndarray:
    if output_path is None:
        output_path = _default_output_path(input_path)

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    debug = cfg["debug"]
    ddir = cfg["debug_dir"]

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # 1. Resize
    image = resize_image(image, cfg["width"])
    original_resized = image.copy()  # keep for rembg (full shading cues)

    # Compute adaptive min_area based on image size
    adaptive_min_area = int(0.0015 * image.shape[0] * image.shape[1])

    # 2. Extract outlines (facial features, strong strokes)
    outline_mask = extract_outline_mask(image, cfg["dark_threshold"])
    outline_mask = normalize_outline(outline_mask, 2)
    if debug:
        _debug_save(outline_mask, ddir, "01_outline_mask")

    # 3. Extract white highlights (teeth, eye whites)
    white_mask = extract_white_mask(image, cfg["white_threshold"])
    if debug:
        _debug_save(white_mask, ddir, "02_white_mask")

    # 4. Build color layer (inpaint over protected pixels)
    protected = cv2.bitwise_or(outline_mask, white_mask)
    color_layer = build_color_layer(image, protected)
    if debug:
        _debug_save(color_layer, ddir, "03_color_layer")

    # 5. Bilateral filter (two passes for smoother fills)
    for _ in range(2):
        color_layer = apply_bilateral_filter(
            color_layer, cfg["bilateral_d"],
            cfg["bilateral_sigma_color"], cfg["bilateral_sigma_space"]
        )

    # 5b. Pre-compute foreground mask so K-means ignores background.
    #     Without this, the bright background steals cluster budget and
    #     lighter skin tones get merged into the background color.
    fg_mask = None
    if cfg["remove_bg"]:
        try:
            from rembg import remove as _rembg_remove
            from PIL import Image as _Image
            _rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
            _rgba = np.array(_rembg_remove(_Image.fromarray(_rgb)))
            fg_mask = (_rgba[:, :, 3] >= cfg["bg_alpha_cutoff"]).astype(np.uint8) * 255
        except ImportError:
            pass

    # 6. K-means quantization (foreground only when mask available)
    color_layer = kmeans_quantization(color_layer, cfg["K"], fg_mask=fg_mask)
    if debug:
        _debug_save(color_layer, ddir, "04_kmeans")

    # 7. Remove small noisy regions (adaptive to image size)
    color_layer = remove_small_regions(color_layer, adaptive_min_area)
    if debug:
        _debug_save(color_layer, ddir, "05_region_cleanup")

    # 8. Morphological closing
    color_layer = apply_morphology(color_layer, cfg["morph_kernel_size"])

    # 9. Merge near-duplicates + re-quantize (pass fg_mask to keep clusters foreground-focused)
    color_layer = merge_similar_colors(color_layer)
    color_layer = kmeans_quantization(color_layer, cfg["K"], fg_mask=fg_mask)
    color_layer = merge_similar_colors(color_layer)

    # 10. Composite all layers
    result = composite_final(color_layer, outline_mask, white_mask)
    if debug:
        _debug_save(result, ddir, "06_final")

    # 11. Background removal via rembg (reuse fg_mask if already computed)
    if cfg["remove_bg"]:
        if fg_mask is not None:
            # Reuse the mask we already computed before K-means
            bgr = result.copy()
            alpha = fg_mask.copy()
            opaque = alpha == 255
            bgr[(white_mask > 0) & opaque] = [255, 255, 255]
            bgr[(outline_mask > 0) & opaque] = [0, 0, 0]
            result = np.dstack([bgr, alpha])
        else:
            result = remove_background(result, original_resized, outline_mask, white_mask,
                                       cfg["bg_alpha_cutoff"])
        if debug:
            _debug_save(result, ddir, "07_bg_removed")

    # 12. Save
    save_output(result, output_path)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess LLM-generated portraits for embroidery digitizing."
    )
    parser.add_argument("input", help="Image name (looked up in images/ folder) or full path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (default: generated/<name>_embroidery.png)")
    parser.add_argument("-K", type=int, default=6, help="Fill colors (default: 6)")
    parser.add_argument("-w", "--width", type=int, default=512, help="Resize width")
    parser.add_argument("--dark-threshold", type=int, default=80,
                        help="Outline extraction threshold (0-255)")
    parser.add_argument("--white-threshold", type=int, default=200,
                        help="White highlight threshold (0-255)")
    parser.add_argument("--no-remove-bg", action="store_true",
                        help="Skip background removal")
    parser.add_argument("--debug", action="store_true", help="Save intermediate outputs")
    parser.add_argument("--debug-dir", default="debug_output", help="Debug output dir")

    args = parser.parse_args()

    user_config = {
        "width": args.width,
        "K": args.K,
        "dark_threshold": args.dark_threshold,
        "white_threshold": args.white_threshold,
        "remove_bg": not args.no_remove_bg,
        "debug": args.debug,
        "debug_dir": args.debug_dir,
    }

    input_path = _resolve_input_path(args.input)
    out = args.output or _default_output_path(input_path)
    process_image(input_path, out, user_config)
    print(f"Done → {out}")
