# 🧵 Embroidery Preprocessor

Converts cartoon-style portraits into embroidery-friendly images suitable for auto-digitizing tools (e.g., Wilcom).

The pipeline quantizes colors, extracts outlines, preserves white highlights (teeth, eye whites), and optionally removes the background — producing clean, flat images ready for stitch digitizing.

Optionally uses **Gemini AI** to generate the cartoon portrait from a regular photo before processing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## Project Structure

```
embroidery_preprocess/
├── pipeline.py                       # Core image processing pipeline
├── app.py                            # Flask web frontend
├── launcher.py                       # Cross-platform auto-launch script
├── remove_bg.py                      # Standalone background removal utility
├── requirements.txt                  # Python dependencies
├── .env.example                      # API key template
├── EmbroideryPreprocessor.command    # macOS double-click launcher
├── EmbroideryPreprocessor.bat        # Windows double-click launcher
├── templates/
│   └── index.html                    # Web UI template
├── images/                           # Input images (place your files here)
└── generated/                        # Output images (created automatically)
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd embroidery_preprocess
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install background removal (optional but recommended)

```bash
pip install "rembg[cpu]"
```

> **Note:** The first run with background removal will download the model (~170 MB). This is a one-time download.

### 5. Set up Gemini API key (optional — for AI cartoon generation)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

3. Edit `.env` and paste your key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

> Without the API key, the app still works — you just skip Step 1 and upload a cartoon image directly.

## Usage

### Web UI (recommended)

```bash
source venv/bin/activate
export GEMINI_API_KEY=your_key_here   # or use .env file
python app.py
```

Open **http://localhost:5001** in your browser.

**Step 1 — Generate Cartoon (optional)**
1. Upload a photo (PNG/JPEG)
2. (Optional) Edit the pre-filled generation prompt to change the style
3. Click **Generate Cartoon** to create a cartoon portrait via Gemini AI
4. If you're not happy with the result, you can:
   - Click **Regenerate** with an empty edit field → generates a completely new cartoon from the original photo
   - Type edit instructions (e.g., "make hair darker") and click **Regenerate** → edits the current cartoon while keeping everything else the same
5. Click **Use This → Step 2** when satisfied

Or click **Skip to Embroidery →** if you already have a cartoon image.

**Step 2 — Embroidery Pipeline**
1. Preview the input image
2. (Optional) Expand **Advanced Settings** to tweak parameters — hover over the **?** icon for guidance
3. Click **Process Image**
4. Preview the result and click **Download Result** to save

### Quick Launch (Double-Click to Run)

Instead of using the terminal, double-click a shortcut to start the app and open the browser automatically:

- **macOS** — double-click `EmbroideryPreprocessor.command`
- **Windows** — double-click `EmbroideryPreprocessor.bat`

> On macOS, if you get a "not identified developer" warning, right-click → Open → Open. You only need to do this once.

Both launchers automatically load the API key from `.env` if present.

To stop the server, close the terminal window or press `Ctrl+C`.

### Command Line

```bash
# Basic usage (reads from images/ folder, saves to generated/)
python pipeline.py <image_name>

# Examples
python pipeline.py yovan                          # looks up images/yovan.png
python pipeline.py couple.jpg                     # looks up images/couple.jpg
python pipeline.py /path/to/photo.png             # absolute path

# With options
python pipeline.py yovan -K 8 -w 768              # 8 fill colors, 768px wide
python pipeline.py yovan --dark-threshold 100      # detect more outlines
python pipeline.py yovan --no-remove-bg            # skip background removal
python pipeline.py yovan -o output.png             # custom output path
python pipeline.py yovan --debug                   # save intermediate stages to debug_output/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `-K` | 6 | Number of fill colors |
| `-w`, `--width` | 512 | Resize width in pixels |
| `--dark-threshold` | 80 | Outline detection sensitivity (0–255) |
| `--white-threshold` | 200 | White highlight detection (0–255) |
| `--no-remove-bg` | — | Skip background removal |
| `-o`, `--output` | auto | Custom output file path |
| `--debug` | — | Save intermediate pipeline stages |
| `--debug-dir` | `debug_output` | Directory for debug output |

## Advanced Settings Reference

| Setting | Default | ↑ Increase | ↓ Decrease |
|---------|---------|------------|------------|
| **Resize Width** | 512 | Sharper, more detailed output | Faster processing, smaller files |
| **Fill Colors** | 6 | More subtle details (blush, shadows) | Simpler, bolder look |
| **Smoothing Strength** | 3 | Cleaner color patches, less noise | More original texture preserved |
| **Color Blending** | 30 | Similar shades merge together | Subtle color differences kept |
| **Blend Reach** | 30 | Broader, more uniform areas | Blending stays local |
| **Gap Filling** | 3 | Fills more gaps in color regions | Preserves thin lines |
| **Outline Sensitivity** | 80 | Detects fainter lines (may catch shadows) | Only the darkest outlines |
| **White Detection** | 200 | Only the brightest whites detected | Includes slightly off-white areas |
| **BG Removal Strictness** | 128 | Removes more aggressively (may clip edges) | Keeps more edges (may leave remnants) |

## Standalone Background Removal

To remove the background from an image without any color processing:

```bash
python remove_bg.py <image_name>
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'rembg'`** — Install it with `pip install "rembg[cpu]"`, or disable background removal with `--no-remove-bg` (CLI) or uncheck the toggle (Web UI).
- **`ModuleNotFoundError: No module named 'google'`** — Install it with `pip install google-genai`. Only needed for Step 1 (AI cartoon generation).
- **"GEMINI_API_KEY not set"** — Create a `.env` file with your key (see Setup step 5), or export it: `export GEMINI_API_KEY=your_key`.
- **Gemini returns no image** — Try a different edit prompt, or use a clearer photo. The model occasionally declines certain images.
- **Processing is slow** — Background removal is the heaviest step. Reduce the resize width or disable background removal for faster results.
- **Background removal cuts into the subject** — Lower the **BG Removal Strictness** value, or disable it entirely if your image already has a clean/transparent background.
- **Too few colors / details lost** — Increase the **Fill Colors (K)** value.
- **Result looks noisy** — Increase **Smoothing Strength** and **Color Blending**.
