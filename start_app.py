"""Starts the Pic Aliver app and writes ready status to a file."""
import sys
import os
from pathlib import Path

# Mark as starting
Path("./app_starting.txt").write_text("starting")

# Import and launch
sys.path.insert(0, str(Path(__file__).parent / "src"))
from picture_aliver.app import build_app

app = build_app()
Path("./app_starting.txt").write_text("ready")
app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True,
    theme="soft",
)
