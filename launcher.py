"""Cross-platform launcher — starts the server and opens the browser."""

import subprocess
import sys
import threading
import time
import webbrowser

URL = "http://localhost:5001"


def open_browser():
    """Wait for the server to be ready, then open the browser."""
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(URL, timeout=1)
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run([sys.executable, "app.py"])
