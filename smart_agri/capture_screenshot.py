"""
capture_screenshot.py — Headless-friendly GUI screenshot capture.
Launches the app, pre-fills rice-field values, waits for full render, saves PNG.
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import tkinter as tk
from src.gui import SmartAgriApp, FEATURE_COLS
from src.utils import FEATURE_RANGES

OUT_PATH = os.path.join(_ROOT, 'results', 'gui_screenshot.png')

# Rice-field demo values for the screenshot
DEMO_VALUES = {
    'N': 80, 'P': 48, 'K': 40,
    'temperature': 23, 'humidity': 82,
    'ph': 6.0, 'rainfall': 200,
}


def main():
    root = tk.Tk()
    app  = SmartAgriApp(root)

    def _setup_and_capture():
        # Pre-fill demo values
        for feat, val in DEMO_VALUES.items():
            if feat in app.feature_vars:
                app.feature_vars[feat].set(val)
                app._sync_label(feat)

        root.update_idletasks()
        root.update()

        # Run analysis so result cards are populated
        app._run_analysis()
        root.update_idletasks()
        root.update()

        # Wait one extra frame for canvas draws to settle
        root.after(800, _shoot)

    def _shoot():
        try:
            from PIL import ImageGrab
            root.update_idletasks()
            root.update()
            x = root.winfo_rootx()
            y = root.winfo_rooty()
            w = root.winfo_width()
            h = root.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(OUT_PATH)
            print(f'Screenshot saved -> {OUT_PATH}')
        except Exception as exc:
            print(f'Screenshot failed: {exc}')
        finally:
            root.destroy()

    root.after(1200, _setup_and_capture)
    root.mainloop()


if __name__ == '__main__':
    main()
