"""
run_ui.py — Launch the ADAS Pipeline web UI.

Usage (from anywhere):
    python run_ui.py

Then open http://localhost:8000 in your browser.
"""

import os
import sys

# Ensure the adas_pipeline root is always on the path regardless of where
# this script is invoked from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import uvicorn

if __name__ == "__main__":
    print("=" * 55)
    print("  ADAS Pipeline UI")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 55)
    uvicorn.run(
        "app.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
