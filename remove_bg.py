"""
Remove background from an image using rembg.
No color processing, no quantization — just background removal.

Reads from images/ folder, saves to generated/ folder.
Requires: pip install "rembg[cpu]"
"""

import os
import sys
from rembg import remove
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, "images")
_GENERATED_DIR = os.path.join(_SCRIPT_DIR, "generated")


def _resolve_input(name: str) -> str:
    if os.path.isfile(name):
        return name
    candidate = os.path.join(_IMAGES_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    base, ext = os.path.splitext(name)
    if not ext:
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(_IMAGES_DIR, base + ext)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"Image not found: {name}")


def remove_bg(input_path: str, output_path: str | None = None) -> None:
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(_GENERATED_DIR, exist_ok=True)
        output_path = os.path.join(_GENERATED_DIR, f"{base}_nobg.png")

    img = Image.open(input_path)
    result = remove(img)
    result.save(output_path)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_bg.py <image_name> [output]")
        sys.exit(1)

    input_path = _resolve_input(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    remove_bg(input_path, output_path)
