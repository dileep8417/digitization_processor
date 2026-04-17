"""Flask frontend for the Embroidery Preprocessing Pipeline."""

import base64
import os
import tempfile

# Fix SSL on corporate networks (Zscaler etc.) by using OS trust store
try:
    import ssl as _ssl
    import truststore
    _orig_ctx = _ssl.create_default_context
    def _truststore_ctx(*args, **kwargs):
        return truststore.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
    _ssl.create_default_context = _truststore_ctx
except ImportError:
    pass

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from pipeline import process_image, DEFAULT_CONFIG

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_PROMPT = (
    """
    Convert the given human portrait into a clean, simplified illustration suitable for embroidery digitizing, while strictly preserving the identity of each person.

    PRIMARY REQUIREMENT (MOST IMPORTANT):
    Every person in the image must remain clearly recognizable
    Preserve exact facial geometry for each person:
    eye shape and spacing
    nose shape and structure
    mouth shape and expression
    jawline and face proportions
    Do NOT alter or stylize facial proportions

    COLOR REQUIREMENTS (CRITICAL FOR EMBROIDERY):
    Use warm, natural skin tones (beige, tan, or brown depending on the person) — NEVER gray or white for skin
    Skin must have a visible warm undertone (yellow/orange), not neutral gray
    Use exactly 5–7 clearly distinct, saturated fill colors across the entire portrait
    Each color region must be visibly different from its neighbors — avoid subtle gradients
    Lips must be a clearly visible pink, red, or brown tone, distinctly different from surrounding skin
    Cheeks should have a slight warm tint different from the rest of the skin
    Hair color must be clearly distinct from skin color
    Clothing should use 1–2 solid, saturated colors that contrast with skin
    If clothes are white colour then replace it to some other lighter colour

    SIMPLIFICATION REQUIREMENTS:
    Simplify skin into 2–3 flat, solid tone regions with hard edges between them (one light tone, one shadow tone)
    Skin tones should be natural and uniform within each region — NO blush, NO rosy cheeks, NO circular color spots on the face
    Each skin tone region should be a clean, flat fill — like cutting colored paper
    Simplify hair and beard into clean, solid color blocks
    Remove fine details such as skin texture, pores, and noise
    NO gradients — every region should be a single flat color
    Think of it as a paint-by-numbers image: each region is one solid color with a clear edge

    FACIAL FEATURE CLARITY (CRITICAL):
    Ensure eyes, eyebrows, nose, and lips are clearly visible for EVERY face
    Eyes must have visible dark pupils, colored iris, and white sclera — each as distinct regions
    Eyebrows must be noticeably darker than surrounding skin
    Nose must have a visible contour line or a distinct shadow tone
    Lips must be a different color from skin — not just slightly darker, but a clearly different hue
    Facial features must have strong contrast against the skin — avoid low-contrast rendering

    CONSISTENCY ACROSS FACES:
    Apply the same level of detail and contrast to all faces
    Do not leave any face with weaker or missing features compared to others

    EDGES AND STRUCTURE:
    Use clear, solid dark outlines (near-black) around all major features: eyes, nose, mouth, face shape, hair boundary
    Outlines should be consistent thickness, bold enough to be visible at small sizes
    Maintain clean boundaries between every color region — no soft blending

    BACKGROUND:
    Use a solid, uniform light grey background (approximately RGB 200, 200, 200)
    No gradients, no textures
    Background must be clearly different from skin tones and from pure white

    AVOID:
    Beautifying
    Blushes 
    Gray or desaturated skin tones
    Soft gradients or blended color transitions
    Photorealistic textures or lighting
    Subtle color differences between adjacent regions
    Over-smoothing that removes facial features
    Extremely low contrast where features blend into skin
    Near-white or washed-out colors anywhere on the subject
    Blush effects, rosy cheeks, or any circular/radial color on the face
    Anime or manga-style face coloring

    OUTPUT GOAL: The result should look like a clean, bold, flat-color illustration — similar to a vector art portrait or a paint-by-numbers template. Every region is one solid color. All faces have clear, high-contrast features. The image should contain exactly 5–7 distinct colors (excluding black outlines and white highlights) that are easily separable by automated color quantization.
    """
)


@app.route("/")
def index():
    return render_template("index.html", defaults=DEFAULT_CONFIG,
                           has_api_key=bool(GEMINI_API_KEY),
                           default_prompt=DEFAULT_PROMPT)


@app.route("/generate", methods=["POST"])
def generate():
    """Step 1 — Generate cartoon portrait via Gemini."""
    if not GEMINI_API_KEY:
        return jsonify(error="GEMINI_API_KEY not set. Export it as an environment variable."), 400

    file = request.files.get("image")
    base_prompt = request.form.get("base_prompt", DEFAULT_PROMPT).strip()
    edit_prompt = request.form.get("edit_prompt", "").strip()
    previous_file = request.files.get("previous_image")
    if not file:
        return jsonify(error="No image uploaded"), 400

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        if previous_file and edit_prompt:
            # Case 3: edit the previous cartoon
            img_bytes = previous_file.read()
            mime = "image/png"
            prompt = (f"Edit this cartoon illustration with the following changes: {edit_prompt}\n\n"
                      "Keep everything else exactly the same — preserve the identity, style, colors, and outlines.")
        else:
            # Case 1 & 2: fresh generation from original photo
            img_bytes = file.read()
            mime = file.content_type or "image/png"
            prompt = base_prompt

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    types.Part.from_text(text=prompt),
                ]),
            ],
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return jsonify(image=b64)

        return jsonify(error="Gemini did not return an image. Try a different prompt."), 500
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/process", methods=["POST"])
def process():
    """Step 2 — Run embroidery pipeline."""
    file = request.files.get("image")
    image_b64 = request.form.get("image_b64")

    if file:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif image_b64:
        raw = base64.b64decode(image_b64)
        file_bytes = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        return jsonify(error="No image provided"), 400

    if image is None:
        return jsonify(error="Invalid image file"), 400

    cfg = {}
    for key, default in DEFAULT_CONFIG.items():
        val = request.form.get(key)
        if val is None:
            continue
        if isinstance(default, bool):
            cfg[key] = val.lower() in ("true", "1", "on")
        elif isinstance(default, int):
            cfg[key] = int(val)
        elif isinstance(default, float):
            cfg[key] = float(val)
        else:
            cfg[key] = val

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_in_path, tmp_out_path = tmp_in.name, tmp_out.name
        cv2.imwrite(tmp_in_path, image)

    try:
        process_image(tmp_in_path, tmp_out_path, cfg)
        result = cv2.imread(tmp_out_path, cv2.IMREAD_UNCHANGED)
        _, buf = cv2.imencode(".png", result)
        b64 = base64.b64encode(buf).decode("utf-8")
        return jsonify(image=b64)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
