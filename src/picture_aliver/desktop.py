"""
Pic Aliver - Desktop Application

Native Windows application. No browser needed.

Usage:  python -m src.picture_aliver.desktop
        python -m src.picture_aliver.desktop --model "DreamShaper XL"
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue
import tkinter as tk
from tkinter import (
    Toplevel, Frame, Label, Entry, Button, Text, Canvas,
    filedialog, messagebox, ttk, PhotoImage, DISABLED, NORMAL,
    END, WORD, LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH, NW,
    StringVar, IntVar, DoubleVar, BooleanVar,
)
from tkinterdnd2 import TkinterDnD, DND_FILES
from typing import Optional, Dict, Any, Callable, List

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from picture_aliver.experiences import ModelExperienceLauncher, GenerationResult
from core.model_registry import ModelCategory
from core.debug import debug as _debug


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

PAD = {"padx": 6, "pady": 4}
LPAD = {"padx": 10, "pady": 6}
HISTORY_FILE = Path("./outputs/prompt_history.json")


# ──────────────────────────────────────────────
# PROMPT HISTORY
# ──────────────────────────────────────────────

class PromptHistory:
    def __init__(self, path: Path = HISTORY_FILE):
        self.path = path
        self.entries: List[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []
        else:
            self.entries = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def add(self, model: str, mode: str, prompt: str, neg: str,
            seed: Optional[int], width: int, height: int, steps: int,
            cfg: float, status: str, error: Optional[str] = None,
            output_path: Optional[str] = None, duration: float = 0.0,
            vram_mb: float = 0.0):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": neg,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "status": status,
            "error": error,
            "output_path": output_path,
            "duration": round(duration, 2),
            "vram_mb": round(vram_mb, 0),
        }
        self.entries.append(entry)
        if len(self.entries) > 500:
            self.entries = self.entries[-500:]
        self.save()

    def clear(self):
        self.entries.clear()
        self.save()

    def build_report(self, last_n: int = 10) -> str:
        lines = []
        lines.append("## Pic Aliver Error Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("### Environment")
        lines.append(f"- Platform: {sys.platform}")
        lines.append(f"- Python: {sys.version.split()[0]}")
        lines.append("")
        lines.append("### Recent Generations")
        recent = self.entries[-last_n:] if len(self.entries) > last_n else self.entries
        for e in recent:
            icon = "OK" if e["status"] == "success" else "FAIL"
            err = f"  Error: {e['error']}" if e["error"] else ""
            lines.append(f"- [{icon}] {e['model']} {e['mode']} {e['duration']}s{err}")
        lines.append("")
        lines.append("### Last Failure Details")
        for e in reversed(self.entries):
            if e["status"] != "success":
                lines.append(f"- **Model:** {e['model']}")
                lines.append(f"- **Mode:** {e['mode']}")
                lines.append(f"- **Prompt:** {e['prompt']}")
                lines.append(f"- **Duration:** {e['duration']}s")
                lines.append(f"- **Error:** {e['error']}")
                lines.append("")
                break
        return "\n".join(lines)


# ──────────────────────────────────────────────
# HELPER: run in thread and update GUI via queue
# ──────────────────────────────────────────────

class GenWorker:
    """Runs generation in a background thread."""

    def __init__(self, app: "PicAliverApp"):
        self.app = app
        self.queue: Queue = Queue()
        self._running = False

    def generate(self, model_name: str, mode: str, kwargs: dict):
        if self._running:
            return
        self._running = True
        self.app.set_status("Starting...")
        self.app.show_progress(0, 1, "Initializing...")
        kwargs["_progress_callback"] = self._on_progress
        t = threading.Thread(target=self._work, args=(model_name, mode, kwargs), daemon=True)
        t.start()
        self.app.after(100, self._poll)

    def _on_progress(self, current: int, total: int, stage: str = ""):
        self.queue.put(("progress", current, total, stage))

    def _work(self, model_name: str, mode: str, kwargs: dict):
        try:
            launcher = self.app.launcher
            trace = _debug.begin_trace(mode, model_name, kwargs.get("prompt", ""))

            self._on_progress(0, 1, f"Preparing {model_name}...")
            exp = launcher.get_experience(model_name)
            if exp is None:
                self.queue.put(("error", f"Unknown model: {model_name}", kwargs))
                _debug.end_trace(False, "Unknown model")
                return

            t0 = time.time()
            result = exp.run_mode(mode, **kwargs)
            elapsed = time.time() - t0
            result.processing_time = elapsed

            self._on_progress(1, 1, "Complete")
            if result.success and result.output_path:
                _debug.end_trace(True)
                self.queue.put(("result", result, kwargs))
            else:
                _debug.end_trace(False, result.error)
                self.queue.put(("error", result.error or "Generation failed", kwargs))
        except Exception as e:
            _debug.end_trace(False, str(e))
            self.queue.put(("error", str(e), kwargs))

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, current, total, stage = msg
                    self.app.show_progress(current, total, stage)
                    continue
                if kind == "error" or kind == "result":
                    self.app.hide_progress()
                gen_kwargs = msg[2] if len(msg) > 2 else {}
                self._running = False
                if kind == "result":
                    self.app.show_result(msg[1])
                    self.app.set_status(f"Done in {msg[1].processing_time:.1f}s")
                    self.app._log_event("OK", msg[1].model_name, msg[1].mode,
                                        msg[1].processing_time, error=None)
                    self.app._history_add(msg[1], gen_kwargs)
                elif kind == "error":
                    self.app.set_status(f"Error: {msg[1]}")
                    self.app._log_event("FAIL", gen_kwargs.get("_model", "?"),
                                        gen_kwargs.get("_mode", "?"), 0, error=str(msg[1]))
                    self.app._history_add_fail(msg[1], gen_kwargs)
                    messagebox.showerror("Error", msg[1])
        except Exception:
            pass
        if self._running:
            self.app.after(100, self._poll)

    def done(self):
        self._running = False


# ──────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────

class PicAliverApp(ttk.Frame):
    """Main Pic Aliver desktop application."""

    def __init__(self, master: Tk, preload_model: Optional[str] = None):
        super().__init__(master)
        self.master = master
        master.title("Pic Aliver")
        master.geometry("860x780")
        master.minsize(700, 650)

        self.history = PromptHistory()
        self.launcher = ModelExperienceLauncher()
        self.worker = GenWorker(self)
        self._preload_model = preload_model

        self._build_ui()
        self._populate_models()
        self._refresh_history()

        if preload_model:
            self._do_preload(preload_model)

    # ── UI BUILDING ──

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        BG = "#1e1e1e"
        BG2 = "#2d2d2d"
        BG3 = "#3c3c3c"
        FG = "#e0e0e0"
        FG2 = "#a0a0a0"
        BLUE = "#0078d4"
        BLUE_HOVER = "#1a9eff"
        BLUE_PRESSED = "#005a9e"
        SELECT = "#094771"
        style.configure('.', background=BG, foreground=FG, fieldbackground=BG2,
                        troughcolor=BG3, selectbackground=SELECT, selectforeground=FG)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TButton', background=BLUE, foreground=FG, borderwidth=1, focusthickness=2, focuscolor='')
        style.map('TButton', background=[('active', BLUE_HOVER), ('pressed', BLUE_PRESSED)],
                  foreground=[('disabled', '#666666')])
        style.configure('TEntry', fieldbackground=BG2, foreground=FG, insertcolor=FG, padding=4)
        style.configure('TCombobox', fieldbackground=BG2, foreground=FG, arrowcolor=FG, padding=2)
        style.map('TCombobox', fieldbackground=[('readonly', BG2)])
        style.configure('TSpinbox', fieldbackground=BG2, foreground=FG)
        style.configure('TNotebook', background=BG, tabmargins=[0, 2, 0, 0])
        style.configure('TNotebook.Tab', background=BG3, foreground=FG2, padding=[10, 3], borderwidth=1)
        style.map('TNotebook.Tab', background=[('selected', BLUE)], foreground=[('selected', FG)])
        style.configure('TLabelframe', background=BG2, foreground=FG, bordercolor=BG3, lightcolor=BG2, darkcolor=BG3)
        style.configure('TLabelframe.Label', background=BG2, foreground=FG)
        style.configure('Treeview', background=BG2, foreground=FG, fieldbackground=BG2, rowheight=24)
        style.configure('Treeview.Heading', background=BG3, foreground=FG, borderwidth=1)
        style.map('Treeview.Heading', background=[('active', BLUE)])
        style.configure('Horizontal.TProgressbar', background=BLUE, troughcolor=BG3, bordercolor=BG3, lightcolor=BLUE, darkcolor=BLUE)
        style.configure('Vertical.TScrollbar', background=BG3, troughcolor=BG, bordercolor=BG3, arrowcolor=FG)
        style.map('Vertical.TScrollbar', background=[('active', BG2)])
        style.configure('Red.TButton', background='#c42b1c', foreground=FG, borderwidth=1)
        style.map('Red.TButton', background=[('active', '#e84535'), ('pressed', '#a02010')])
        style.configure('Small.TButton', background=BLUE, foreground=FG, borderwidth=1, padding=[6, 1], font=('Arial', 8))
        style.map('Small.TButton', background=[('active', BLUE_HOVER), ('pressed', BLUE_PRESSED)])

    def _build_ui(self):
        self.master.configure(bg="#1e1e1e")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self._setup_styles()
        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew", **LPAD)

        # ── Generate Tab ──
        gen_container = ttk.Frame(nb)
        nb.add(gen_container, text="Generate")
        gen_container.columnconfigure(0, weight=1)
        gen_container.rowconfigure(0, weight=1)

        gen_canvas = Canvas(gen_container, borderwidth=0, highlightthickness=0, bg="#1e1e1e")
        gen_scrollbar = ttk.Scrollbar(gen_container, orient="vertical", command=gen_canvas.yview)
        gen_canvas.configure(yscrollcommand=gen_scrollbar.set)
        gen_scrollbar.grid(row=0, column=1, sticky="ns")
        gen_canvas.grid(row=0, column=0, sticky="nsew")

        gen_frame = ttk.Frame(gen_canvas)
        gen_frame.bind("<Configure>", lambda e: gen_canvas.configure(scrollregion=gen_canvas.bbox("all")))
        gen_frame_window = gen_canvas.create_window((0, 0), window=gen_frame, anchor="nw")
        def _resize_gen_frame(event):
            gen_canvas.itemconfig(gen_frame_window, width=event.width)
        gen_canvas.bind("<Configure>", _resize_gen_frame)
        gen_frame.columnconfigure(1, weight=1)

        def _on_mousewheel(event):
            gen_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        gen_canvas.bind("<Enter>", lambda e: gen_canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+"))
        gen_canvas.bind("<Leave>", lambda e: gen_canvas.unbind_all("<MouseWheel>"))

        row = 0

        # Model + Mode row
        f = ttk.Frame(gen_frame)
        f.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)
        ttk.Label(f, text="Model:").grid(row=0, column=0, **PAD)
        self.model_var = StringVar()
        self.model_combo = ttk.Combobox(f, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.grid(row=0, column=1, sticky="ew", **PAD)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Label(f, text="Mode:").grid(row=0, column=2, **PAD)
        self.mode_var = StringVar(value="txt2img")
        self.mode_combo = ttk.Combobox(f, textvariable=self.mode_var, state="readonly", width=14)
        self.mode_combo.grid(row=0, column=3, sticky="ew", **PAD)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)
        row += 1

        # ── Prompt box ──
        prompt_box = ttk.LabelFrame(gen_frame, text="Prompt", padding=6)
        prompt_box.grid(row=row, column=0, columnspan=3, sticky="ew", **PAD)
        prompt_box.columnconfigure(1, weight=1)

        ttk.Label(prompt_box, text="Prompt:").grid(row=0, column=0, sticky="w", **PAD)
        self.prompt_entry = Text(prompt_box, height=3, width=60)
        self.prompt_entry.grid(row=0, column=1, columnspan=2, sticky="ew", **PAD)

        ttk.Label(prompt_box, text="Negative:").grid(row=1, column=0, sticky="w", **PAD)
        self.neg_entry = Text(prompt_box, height=2, width=60)
        self.neg_entry.grid(row=1, column=1, columnspan=2, sticky="ew", **PAD)
        row += 1

        # ── Settings row ──
        sf = ttk.LabelFrame(gen_frame, text="Settings", padding=6)
        sf.grid(row=row, column=0, columnspan=3, sticky="ew", **PAD)
        sf.columnconfigure(1, weight=1)
        sf.columnconfigure(3, weight=1)
        sf.columnconfigure(5, weight=1)

        RES_PRESETS = [
            "512x512", "512x768", "576x1024", "640x640", "768x768",
            "768x1024", "1024x576", "1024x1024", "1024x1280",
            "1280x720", "1280x1280", "1920x1080",
        ]
        ttk.Label(sf, text="Width:").grid(row=0, column=0, **PAD)
        self.width_var = StringVar(value="1024")
        ttk.Entry(sf, textvariable=self.width_var, width=7).grid(row=0, column=1, **PAD)
        ttk.Label(sf, text="Height:").grid(row=0, column=2, **PAD)
        self.height_var = StringVar(value="1024")
        ttk.Entry(sf, textvariable=self.height_var, width=7).grid(row=0, column=3, **PAD)

        ttk.Label(sf, text="Steps:").grid(row=0, column=4, **PAD)
        self.steps_var = StringVar(value="25")
        ttk.Entry(sf, textvariable=self.steps_var, width=7).grid(row=0, column=5, **PAD)

        ttk.Label(sf, text="CFG:").grid(row=1, column=0, **PAD)
        self.cfg_var = StringVar(value="7.5")
        ttk.Entry(sf, textvariable=self.cfg_var, width=7).grid(row=1, column=1, **PAD)
        ttk.Label(sf, text="Seed:").grid(row=1, column=2, **PAD)
        self.seed_var = StringVar(value="-1")
        ttk.Entry(sf, textvariable=self.seed_var, width=7).grid(row=1, column=3, **PAD)

        # Resolution preset
        ttk.Label(sf, text="Res:").grid(row=1, column=4, **PAD)
        self.res_var = StringVar()
        res_combo = ttk.Combobox(sf, textvariable=self.res_var, values=RES_PRESETS, state="readonly", width=12)
        res_combo.grid(row=1, column=5, sticky="ew", **PAD)
        def _on_res_preset(_event=None):
            val = self.res_var.get()
            if "x" in val:
                w, h = val.split("x")
                self.width_var.set(w.strip())
                self.height_var.set(h.strip())
        res_combo.bind("<<ComboboxSelected>>", _on_res_preset)

        # VRAM info
        ttk.Label(sf, text="VRAM:").grid(row=2, column=0, **PAD)
        self.vram_var = StringVar(value="N/A")
        ttk.Label(sf, textvariable=self.vram_var, width=10, anchor="w").grid(row=2, column=1, **PAD)
        row += 1

        # ── Image Input box ──
        self.img_frame = ttk.LabelFrame(gen_frame, text="Image Input", padding=6)
        self.img_frame.grid(row=row, column=0, columnspan=3, sticky="ew", **PAD)
        self.img_frame.columnconfigure(1, weight=1)
        self.img_frame.columnconfigure(3, weight=1)

        # img2img sub-section
        ttk.Label(self.img_frame, text="Input Image:").grid(row=0, column=0, **PAD)
        self.i2i_path_var = StringVar()
        self.i2i_picker = self._make_image_picker(self.img_frame, self.i2i_path_var, text="Click to select input image")
        self.i2i_picker.grid(row=0, column=1, columnspan=3, **PAD)
        ttk.Label(self.img_frame, text="Strength:").grid(row=1, column=0, **PAD)
        self.strength_var = StringVar(value="0.75")
        ttk.Entry(self.img_frame, textvariable=self.strength_var, width=6).grid(row=1, column=1, sticky="w", **PAD)

        # img2video sub-section
        ttk.Separator(self.img_frame, orient="horizontal").grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Label(self.img_frame, text="Start Frame:").grid(row=3, column=0, **PAD)
        self.i2v_start_var = StringVar()
        self.i2v_start_picker = self._make_image_picker(self.img_frame, self.i2v_start_var, text="Click to select start frame")
        self.i2v_start_picker.grid(row=3, column=1, columnspan=2, **PAD)
        ttk.Label(self.img_frame, text="End Frame:").grid(row=4, column=0, **PAD)
        self.i2v_end_var = StringVar()
        self.i2v_end_picker = self._make_image_picker(self.img_frame, self.i2v_end_var, text="Click to select end frame (optional)")
        self.i2v_end_picker.grid(row=4, column=1, columnspan=2, **PAD)
        ttk.Label(self.img_frame, text="Frames:").grid(row=3, column=3, **PAD)
        self.frames_var = StringVar(value="24")
        ttk.Entry(self.img_frame, textvariable=self.frames_var, width=6).grid(row=3, column=4, **PAD)
        ttk.Label(self.img_frame, text="FPS:").grid(row=4, column=3, **PAD)
        self.fps_var = StringVar(value="8")
        ttk.Entry(self.img_frame, textvariable=self.fps_var, width=6).grid(row=4, column=4, **PAD)
        self.img_frame.grid_remove()
        row += 1

        # Generate button
        btn_frame = ttk.Frame(gen_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=8)
        self.gen_btn = ttk.Button(btn_frame, text="Generate", command=self._on_generate)
        self.gen_btn.pack(side=LEFT, **PAD)
        ttk.Button(btn_frame, text="Copy Error Report", command=self._copy_error_report).pack(side=LEFT, **PAD)
        row += 1

        # Progress bar
        prog_frame = ttk.Frame(gen_frame)
        prog_frame.grid(row=row, column=0, columnspan=3, sticky="ew", **PAD)
        prog_frame.columnconfigure(1, weight=1)
        self.prog_label = ttk.Label(prog_frame, text="", font=("Arial", 9))
        self.prog_label.grid(row=0, column=0, **PAD)
        self.prog_bar = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", length=200)
        self.prog_bar.grid(row=0, column=1, sticky="ew", **PAD)
        self.prog_bar.grid_remove()
        self.prog_label.grid_remove()
        row += 1

        # Output area (accumulates persistent results)
        self.output_frame = ttk.LabelFrame(gen_frame, text="Output")
        self.output_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", **PAD)
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_outer = ttk.Frame(self.output_frame)
        self.output_outer.grid(row=0, column=0, sticky="nsew")
        self.output_outer.columnconfigure(0, weight=1)
        self.output_outer.rowconfigure(0, weight=1)
        self.output_canvas = Canvas(self.output_outer, bg="#2d2d2d", highlightthickness=0, borderwidth=0)
        self.output_scrollbar = ttk.Scrollbar(self.output_outer, orient="vertical", command=self.output_canvas.yview)
        self.output_canvas.configure(yscrollcommand=self.output_scrollbar.set)
        self.output_scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_canvas.grid(row=0, column=0, sticky="nsew")
        self.output_inner = ttk.Frame(self.output_canvas)
        self.output_inner.bind("<Configure>",
            lambda e: self.output_canvas.configure(scrollregion=self.output_canvas.bbox("all")))
        self.output_inner_window = self.output_canvas.create_window((0, 0), window=self.output_inner, anchor="nw")
        def _resize_output(event):
            self.output_canvas.itemconfig(self.output_inner_window, width=event.width)
        self.output_canvas.bind("<Configure>", _resize_output)
        self.output_items: list[dict] = []
        self._output_thumb_refs: list = []
        self._output_placeholder = ttk.Label(self.output_inner, text="No results yet.", foreground="#888")
        self._output_placeholder.pack(pady=60)
        row += 1

        # Error log (real-time)
        log_frame = ttk.LabelFrame(gen_frame, text="Event Log")
        log_frame.grid(row=row, column=0, columnspan=3, sticky="ew", **PAD)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = Text(log_frame, height=5, wrap=WORD, font=("Consolas", 9),
                             bg="#1a1a1a", fg="#cccccc", state=DISABLED, insertbackground="#cccccc",
                             borderwidth=0, highlightthickness=0)
        self.log_text.grid(row=0, column=0, sticky="ew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Status bar
        self.status_var = StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w",
                               background="#007acc", foreground="#ffffff")
        status_bar.grid(row=1, column=0, sticky="ew", **LPAD)

        # ── History Tab ──
        hist_frame = ttk.Frame(nb)
        nb.add(hist_frame, text="History")
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)

        tree_frame = ttk.Frame(hist_frame)
        tree_frame.grid(row=0, column=0, sticky="nsew", **LPAD)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        columns = ("time", "model", "mode", "prompt", "status", "duration")
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                         height=12, selectmode="browse")
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("model", text="Model")
        self.history_tree.heading("mode", text="Mode")
        self.history_tree.heading("prompt", text="Prompt")
        self.history_tree.heading("status", text="Status")
        self.history_tree.heading("duration", text="Dur (s)")
        self.history_tree.column("time", width=140, minwidth=100)
        self.history_tree.column("model", width=140, minwidth=100)
        self.history_tree.column("mode", width=70, minwidth=60)
        self.history_tree.column("prompt", width=200, minwidth=100)
        self.history_tree.column("status", width=60, minwidth=50)
        self.history_tree.column("duration", width=60, minwidth=50)
        self.history_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.history_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=tree_scroll.set)
        self.history_tree.bind("<Double-1>", self._on_history_doubleclick)

        btn_f = ttk.Frame(hist_frame)
        btn_f.grid(row=1, column=0, pady=4)
        ttk.Button(btn_f, text="Refresh", command=self._refresh_history).pack(side=LEFT, **PAD)
        ttk.Button(btn_f, text="Clear History", command=self._clear_history).pack(side=LEFT, **PAD)
        ttk.Button(btn_f, text="Copy Error Report", command=self._copy_error_report).pack(side=LEFT, **PAD)

        # ── Debug Tab ──
        debug_frame = ttk.Frame(nb)
        nb.add(debug_frame, text="Debug")
        debug_frame.columnconfigure(0, weight=1)
        debug_frame.rowconfigure(0, weight=1)

        debug_text_frame = ttk.Frame(debug_frame)
        debug_text_frame.grid(row=0, column=0, sticky="nsew", **LPAD)
        debug_text_frame.columnconfigure(0, weight=1)
        debug_text_frame.rowconfigure(0, weight=1)

        self.debug_text = Text(debug_text_frame, wrap=WORD, font=("Consolas", 10), bg="#1e1e1e", fg="#00ff00",
                                insertbackground="#00ff00", borderwidth=0, highlightthickness=0)
        self.debug_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(debug_text_frame, orient="vertical", command=self.debug_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.debug_text.configure(yscrollcommand=scroll.set)

        btn_f2 = ttk.Frame(debug_frame)
        btn_f2.grid(row=1, column=0, pady=4)
        ttk.Button(btn_f2, text="Refresh", command=self._refresh_debug).pack(side=LEFT, **PAD)
        ttk.Button(btn_f2, text="Save Log", command=self._save_debug_log).pack(side=LEFT, **PAD)
        ttk.Button(btn_f2, text="Clear", command=self._clear_debug).pack(side=LEFT, **PAD)

        # ── Gallery Tab ──
        gallery_frame = ttk.Frame(nb)
        nb.add(gallery_frame, text="Gallery")
        gallery_frame.columnconfigure(0, weight=1)
        gallery_frame.rowconfigure(0, weight=1)

        gal_container = ttk.Frame(gallery_frame)
        gal_container.grid(row=0, column=0, sticky="nsew", **LPAD)
        gal_container.columnconfigure(0, weight=1)
        gal_container.rowconfigure(1, weight=1)

        gal_toolbar = ttk.Frame(gal_container)
        gal_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(gal_toolbar, text="Refresh", command=self._refresh_gallery).pack(side=LEFT, **PAD)
        self.gal_filter_var = StringVar(value="All")
        gal_filter = ttk.Combobox(gal_toolbar, textvariable=self.gal_filter_var,
                                  values=["All", "txt2img", "img2img", "txt2video", "img2video"],
                                  state="readonly", width=14)
        gal_filter.pack(side=LEFT, **PAD)
        gal_filter.bind("<<ComboboxSelected>>", lambda _: self._refresh_gallery())

        gal_canvas_frame = ttk.Frame(gal_container)
        gal_canvas_frame.grid(row=1, column=0, sticky="nsew")
        gal_canvas_frame.columnconfigure(0, weight=1)
        gal_canvas_frame.rowconfigure(0, weight=1)

        self.gal_canvas = Canvas(gal_canvas_frame, borderwidth=0, highlightthickness=0, bg="#1e1e1e")
        gal_scrollbar = ttk.Scrollbar(gal_canvas_frame, orient="vertical", command=self.gal_canvas.yview)
        self.gal_canvas.configure(yscrollcommand=gal_scrollbar.set)
        gal_scrollbar.grid(row=0, column=1, sticky="ns")
        self.gal_canvas.grid(row=0, column=0, sticky="nsew")

        self.gal_inner = ttk.Frame(self.gal_canvas)
        self.gal_inner.bind("<Configure>",
                            lambda e: self.gal_canvas.configure(scrollregion=self.gal_canvas.bbox("all")))
        self.gal_window = self.gal_canvas.create_window((0, 0), window=self.gal_inner, anchor="nw")
        def _resize_gal(event):
            self.gal_canvas.itemconfig(self.gal_window, width=event.width)
        self.gal_canvas.bind("<Configure>", _resize_gal)

        self._gallery_items: list[dict] = []
        self._gal_thumb_refs: list = []

        self._update_vram_display()

    def _populate_models(self):
        image_models = self.launcher.manager.available_image_models
        motion_models = self.launcher.manager.available_motion_models

        choices = []
        choices.append("--- TEXT/IMAGE MODELS ---")
        by_style = self.launcher.manager.list_models_by_style()
        for style, names in by_style.items():
            if style == "Motion":
                continue
            for name in names:
                if name in image_models:
                    info = self.launcher.manager.get_image_model_info(name)
                    if info and info.category in (ModelCategory.TEXT2VIDEO, ModelCategory.I2V):
                        continue
                    label = f"  {name} ({info.pipeline_type})" if info else f"  {name}"
                    choices.append(label)

        choices.append("--- VIDEO DIFFUSION ---")
        for name in image_models:
            info = self.launcher.manager.get_image_model_info(name)
            if info and info.category in (ModelCategory.TEXT2VIDEO, ModelCategory.I2V):
                label = f"  {name} ({info.pipeline_type})" if info else f"  {name}"
                choices.append(label)

        choices.append("--- MOTION / ANIMATION ---")
        for name in motion_models:
            info = self.launcher.manager.get_motion_model_info(name)
            label = f"  {name} ({info.pipeline_type})" if info else f"  {name}"
            choices.append(label)

        self.model_combo["values"] = choices
        if choices:
            self.model_combo.current(0)

    def _on_model_change(self, event=None):
        model = self._selected_model()
        if not model:
            return
        is_motion = self.launcher.manager.get_motion_model_info(model) is not None
        info = self.launcher.manager.get_image_model_info(model)
        if info and info.category in (ModelCategory.TEXT2VIDEO, ModelCategory.I2V):
            self.mode_combo["values"] = ["txt2video", "img2video"]
        elif is_motion:
            self.mode_combo["values"] = ["txt2img", "img2img", "img2video"]
        else:
            self.mode_combo["values"] = ["txt2img", "img2img"]
        if self.mode_var.get() not in self.mode_combo["values"]:
            self.mode_var.set(self.mode_combo["values"][0])
        self._on_mode_change()

    def _on_mode_change(self, event=None):
        mode = self.mode_var.get()
        self.img_frame.grid_remove()
        if mode in ("img2img", "img2video", "txt2video"):
            self.img_frame.grid()

    def _selected_model(self) -> Optional[str]:
        raw = self.model_var.get()
        if not raw or raw.startswith("---"):
            return None
        return raw.strip().split(" (")[0]

    def _make_image_picker(self, parent, path_var: StringVar, text: str = "Click or drag image here", size: int = 180):
        frame = ttk.Frame(parent, relief="groove", borderwidth=2)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        label = ttk.Label(frame, text=text, anchor="center", background="#3c3c3c", foreground="#aaaaaa")
        label.grid(row=0, column=0, sticky="nsew")
        frame.configure(width=size, height=int(size * 0.75))

        def _set_image(path: str):
            path_var.set(path)
            try:
                from PIL import Image, ImageTk
                img = Image.open(path)
                img.thumbnail((size, int(size * 0.75)))
                tk_img = ImageTk.PhotoImage(img)
                label.config(image=tk_img, text="", background="#1e1e1e")
                label.image = tk_img
            except Exception:
                label.config(image="", text="(preview failed)", background="#2d2d2d")

        def _on_click(event=None):
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")]
            )
            if path:
                _set_image(path)

        def _on_drop(event):
            raw = event.data.strip()
            if raw.startswith("{") and raw.endswith("}"):
                raw = raw[1:-1]
            if raw.startswith("file:///"):
                raw = raw[8:]
            parts = raw.replace("\n", " ").replace("\r", " ").split()
            for p in parts:
                p = p.strip()
                if p and os.path.isfile(p):
                    _set_image(p)
                    break

        label.bind("<Button-1>", _on_click)
        frame.drop_target_register(DND_FILES)
        frame.dnd_bind("<<Drop>>", _on_drop)
        path_var.trace_add("write", lambda *_: None)
        return frame

    def _do_preload(self, model_name: str):
        def load():
            try:
                self.master.after(0, lambda: self.set_status(f"Loading {model_name}..."))
                self.master.after(0, lambda: self._log_event("...", model_name, "preload", 0))
                exp = self.launcher.get_experience(model_name)
                if exp:
                    if self.launcher.manager.get_motion_model_info(model_name):
                        exp.manager.load_motion_model(model_name)
                    else:
                        exp.manager.load_image_model(model_name)
                self.master.after(0, lambda: self.set_status(f"{model_name} loaded and ready"))
                self.master.after(0, lambda: self._log_event("OK", model_name, "preload", 0))
            except Exception as e:
                self.master.after(0, lambda: self.set_status(f"Pre-load failed: {e}"))
                self.master.after(0, lambda: self._log_event("FAIL", model_name, "preload", 0, error=str(e)))
        t = threading.Thread(target=load, daemon=True)
        t.start()

    # ── GENERATION ──

    def _on_generate(self):
        model = self._selected_model()
        if not model:
            messagebox.showwarning("Pic Aliver", "Select a model.")
            return

        mode = self.mode_var.get()
        exp = self.launcher.get_experience(model)
        if exp:
            valid = [m["id"] for m in exp.available_modes()]
            if mode not in valid:
                self.set_status(f"Mode '{mode}' not allowed for {model}")
                return
        prompt = self.prompt_entry.get("1.0", END).strip()
        if not prompt:
            messagebox.showwarning("Pic Aliver", "Enter a prompt.")
            return

        neg = self.neg_entry.get("1.0", END).strip()

        kwargs = {
            "prompt": prompt,
            "negative_prompt": neg,
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "steps": int(self.steps_var.get()),
            "guidance_scale": float(self.cfg_var.get()),
            "seed": int(self.seed_var.get()) if self.seed_var.get() != "-1" else None,
            "_model": model,
            "_mode": mode,
        }

        if mode == "img2img":
            if not self.i2i_path_var.get():
                messagebox.showwarning("Pic Aliver", "Select an input image.")
                return
            kwargs["input_image"] = self.i2i_path_var.get()
            kwargs["strength"] = float(self.strength_var.get())
        elif mode in ("img2video", "txt2video"):
            if mode == "img2video" and not self.i2v_start_var.get():
                messagebox.showwarning("Pic Aliver", "Select a start frame.")
                return
            if self.i2v_start_var.get():
                kwargs["input_image"] = self.i2v_start_var.get()
            if self.i2v_end_var.get():
                kwargs["end_image"] = self.i2v_end_var.get()
            kwargs["num_frames"] = int(self.frames_var.get())
            kwargs["fps"] = int(self.fps_var.get())

        self.worker.generate(model, mode, kwargs)

    def show_result(self, result):
        path = result.output_path
        if not path:
            return
        if self._output_placeholder:
            self._output_placeholder.destroy()
            self._output_placeholder = None
        self._add_output_tile(result)
        self.set_status(f"Done: {path.name}")
        self._refresh_gallery()

    def _add_output_tile(self, result):
        from PIL import Image, ImageTk
        path = result.output_path
        frame = ttk.Frame(self.output_inner, relief="solid", borderwidth=1)
        frame.pack(fill="x", padx=4, pady=2)
        ext = path.suffix.lower() if path else ""

        if ext == ".mp4":
            thumb_frame = tk.Frame(frame, width=120, height=68, bg="#1a1a1a", highlightthickness=0)
            thumb_frame.pack(side=tk.LEFT, padx=2, pady=2)
            thumb_frame.pack_propagate(False)
            tk.Label(thumb_frame, text="▶ VIDEO", fg="#0078d4", bg="#1a1a1a",
                     font=("Arial", 9, "bold")).pack(expand=True)
        else:
            try:
                img = Image.open(path)
                img.thumbnail((120, 68))
                photo = ImageTk.PhotoImage(img)
                self._output_thumb_refs.append(photo)
                lbl = ttk.Label(frame, image=photo)
                lbl.image = photo
                lbl.pack(side=tk.LEFT, padx=2, pady=2)
            except Exception:
                ttk.Label(frame, text="(no preview)", width=18).pack(side=tk.LEFT, padx=2, pady=2)

        info_f = ttk.Frame(frame)
        info_f.pack(side=tk.LEFT, fill="x", expand=True, padx=4)
        ttk.Label(info_f, text=path.name, font=("Arial", 8, "bold")).pack(anchor="w")
        ttk.Label(info_f,
                  text=f"[{getattr(result, 'mode', '?')}]  {getattr(result, 'model_name', '?')}",
                  font=("Arial", 7), foreground="#888").pack(anchor="w")

        def _on_menu(ev, p=path):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Save As...", command=lambda: self._output_save(p))
            menu.add_command(label="Open in Explorer", command=lambda: self._open_in_explorer(p))
            menu.add_separator()
            menu.add_command(label="Copy Path", command=lambda: self._copy_path(p))
            menu.add_command(label="Delete from List", command=lambda: self._delete_output_tile(frame, p))
            menu.tk_popup(ev.x_root, ev.y_root)

        frame.bind("<Button-3>", _on_menu)

    def _output_save(self, path):
        save_path = filedialog.asksaveasfilename(
            title="Save Output",
            initialfile=path.name,
            defaultextension=path.suffix,
            filetypes=[("All Files", "*.*")]
        )
        if save_path:
            import shutil
            try:
                shutil.copy2(str(path), str(save_path))
                self.set_status(f"Saved: {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _open_in_explorer(self, path):
        import subprocess
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
        except Exception:
            pass

    def _copy_path(self, path):
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self.set_status("Path copied to clipboard")

    def _delete_output_tile(self, frame, path):
        frame.destroy()
        for i, item in enumerate(self.output_items):
            if item.get("path") == path:
                self.output_items.pop(i)
                break
        if not self.output_inner.winfo_children():
            self._output_placeholder = ttk.Label(self.output_inner, text="No results yet.", foreground="#888")
            self._output_placeholder.pack(pady=60)
            self.output_items.clear()
        self.set_status(f"Removed: {path.name}")

    # ── GALLERY ──

    def _refresh_gallery(self):
        for w in self.gal_inner.winfo_children():
            w.destroy()
        self._gal_thumb_refs.clear()
        self._gallery_items.clear()

        outputs_dir = Path("./outputs").resolve()
        if not outputs_dir.exists():
            return

        filt = self.gal_filter_var.get()
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4")

        items = []
        for mode_dir in outputs_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            if filt != "All" and mode_dir.name != filt:
                continue
            for f in mode_dir.iterdir():
                if f.suffix.lower() in exts:
                    items.append({"path": f, "mode": mode_dir.name})

        items.sort(key=lambda x: x["path"].stat().st_mtime, reverse=True)
        self._gallery_items = items

        if not items:
            ttk.Label(self.gal_inner, text="No generations yet.", foreground="#888").pack(pady=40)
            return

        cols = max(1, (self.gal_canvas.winfo_width() or 600) // 160)
        for idx, item in enumerate(items):
            row = idx // cols
            col = idx % cols
            self._make_gallery_tile(item, row, col)
        self.gal_inner.update_idletasks()

    def _make_gallery_tile(self, item: dict, row: int, col: int):
        from PIL import Image, ImageTk
        frame = ttk.Frame(self.gal_inner, relief="solid", borderwidth=1)
        frame.grid(row=row, column=col, padx=3, pady=3, sticky="nw")

        thumb_size = (140, 105)
        try:
            img = Image.open(item["path"])
            img.thumbnail(thumb_size)
            photo = ImageTk.PhotoImage(img)
            self._gal_thumb_refs.append(photo)
            lbl = ttk.Label(frame, image=photo)
            lbl.image = photo
            lbl.pack(padx=2, pady=(2, 0))
        except Exception:
            ttk.Label(frame, text="(no preview)", width=18).pack(padx=2, pady=(2, 0))

        name = item["path"].stem[:24]
        ttk.Label(frame, text=name, font=("Arial", 7), anchor="center").pack(fill="x", padx=2)
        ttk.Label(frame, text=f"[{item['mode']}]", font=("Arial", 6), foreground="#888",
                  anchor="center").pack(fill="x", padx=2)

        def _on_click(ev, p=item["path"]):
            try:
                import subprocess
                subprocess.Popen(["explorer", "/select,", str(p)])
            except Exception:
                pass

        def _on_menu(ev, p=item["path"]):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Open in Explorer", command=lambda: _on_click(None, p))
            menu.add_command(label="Save to Saved/", command=lambda: self._gallery_save(p))
            menu.add_command(label="Save As...", command=lambda: self._output_save(p))
            menu.add_separator()
            menu.add_command(label="Copy Path", command=lambda: self._copy_path(p))
            menu.add_command(label="Delete", command=lambda: self._gallery_delete(p))
            menu.tk_popup(ev.x_root, ev.y_root)

        frame.bind("<Button-1>", _on_click)
        frame.bind("<Button-3>", _on_menu)

    def _gallery_save(self, path: Path):
        saved_dir = Path("./saved").resolve()
        saved_dir.mkdir(exist_ok=True)
        import shutil
        dest = saved_dir / path.name
        try:
            shutil.copy2(str(path), str(dest))
            self.set_status(f"Saved: {dest}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _gallery_delete(self, path: Path):
        if messagebox.askyesno("Delete", f"Delete {path.name}?"):
            try:
                path.unlink()
                self._refresh_gallery()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete:\n{e}")

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.master.update_idletasks()

    def show_progress(self, current: int, total: int, stage: str = ""):
        if total > 0:
            pct = min(int(current / total * 100), 100)
            self.prog_bar["value"] = pct
            self.prog_bar.grid()
        if stage:
            self.prog_label["text"] = stage
            self.prog_label.grid()
            self.set_status(stage)
        self.master.update_idletasks()

    def hide_progress(self):
        self.prog_bar.grid_remove()
        self.prog_label.grid_remove()
        self.master.update_idletasks()

    # ── EVENT LOG (real-time) ──

    def _log_event(self, icon: str, model: str, mode: str, duration: float,
                   error: Optional[str] = None):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = "ok" if icon == "OK" else "fail" if icon == "FAIL" else "info"
        line = f"[{ts}] [{icon}] {model} {mode}"
        if duration:
            line += f" {duration:.1f}s"
        if error:
            line += f"  {error}"
        line += "\n"

        self.log_text.configure(state=NORMAL)
        self.log_text.insert(END, line)
        self.log_text.see(END)
        self.log_text.configure(state=DISABLED)
        self._update_vram_display()

    def _update_vram_display(self):
        try:
            info = self.launcher.manager.loader.get_vram_info()
            used = info.get("used_mb", 0)
            total = info.get("total_mb", 0)
            self.vram_var.set(f"{used}MB / {total}MB" if total else "N/A")
        except Exception:
            self.vram_var.set("N/A")

    # ── HISTORY ──

    def _history_add(self, result: GenerationResult, kwargs: dict):
        model = kwargs.get("_model", result.model_name)
        mode = kwargs.get("_mode", result.mode)
        self.history.add(
            model=model, mode=mode,
            prompt=kwargs.get("prompt", result.prompt or ""),
            neg=kwargs.get("negative_prompt", ""),
            seed=kwargs.get("seed"),
            width=int(kwargs.get("width", 0)),
            height=int(kwargs.get("height", 0)),
            steps=int(kwargs.get("steps", 0)),
            cfg=float(kwargs.get("guidance_scale", 0)),
            status="success",
            output_path=str(result.output_path) if result.output_path else None,
            duration=result.processing_time,
            vram_mb=0,
        )
        self._refresh_history()

    def _history_add_fail(self, error: str, kwargs: dict):
        self.history.add(
            model=kwargs.get("_model", "?"),
            mode=kwargs.get("_mode", "?"),
            prompt=kwargs.get("prompt", ""),
            neg=kwargs.get("negative_prompt", ""),
            seed=kwargs.get("seed"),
            width=int(kwargs.get("width", 0)),
            height=int(kwargs.get("height", 0)),
            steps=int(kwargs.get("steps", 0)),
            cfg=float(kwargs.get("guidance_scale", 0)),
            status="failed",
            error=error,
            duration=0,
            vram_mb=0,
        )
        self._refresh_history()

    def _refresh_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for e in reversed(self.history.entries[-200:]):
            status = e["status"]
            tag = "ok" if status == "success" else "fail"
            dur = f"{e['duration']:.1f}" if e["duration"] else ""
            prompt = e["prompt"][:60] + "..." if len(e["prompt"]) > 60 else e["prompt"]
            self.history_tree.insert("", END, values=(
                e["timestamp"], e["model"], e["mode"], prompt, status, dur
            ), tags=(tag,))
        self.history_tree.tag_configure("ok", foreground="#4caf50", background="#1e1e1e")
        self.history_tree.tag_configure("fail", foreground="#f44336", background="#1e1e1e")

    def _on_history_doubleclick(self, event):
        sel = self.history_tree.selection()
        if not sel:
            return
        item = sel[0]
        values = self.history_tree.item(item, "values")
        if not values:
            return
        ts = values[0]
        entry = None
        for e in self.history.entries:
            if e["timestamp"] == ts:
                entry = e
                break
        if not entry:
            return
        msg = (
            f"Time: {entry['timestamp']}\n"
            f"Model: {entry['model']}\n"
            f"Mode: {entry['mode']}\n"
            f"Prompt: {entry['prompt']}\n"
            f"Negative: {entry['negative_prompt']}\n"
            f"Seed: {entry['seed']}\n"
            f"Size: {entry['width']}x{entry['height']}\n"
            f"Steps: {entry['steps']}  CFG: {entry['cfg']}\n"
            f"Duration: {entry['duration']}s\n"
            f"Status: {entry['status']}\n"
        )
        if entry.get("error"):
            msg += f"Error: {entry['error']}\n"
        if entry.get("output_path"):
            msg += f"Output: {entry['output_path']}\n"
        messagebox.showinfo("Generation Details", msg)

    def _clear_history(self):
        if messagebox.askyesno("Clear History", "Delete all history entries?"):
            self.history.clear()
            self._refresh_history()

    def _copy_error_report(self):
        report = self.history.build_report()
        self.clipboard_clear()
        self.clipboard_append(report)
        self.set_status("Error report copied to clipboard")

    # ── DEBUG ──

    def _refresh_debug(self):
        diag = _debug.get_diagnostics()
        self.debug_text.delete("1.0", END)
        self.debug_text.insert(END, "─── SYSTEM ───\n")
        d = diag.get("system", {})
        self.debug_text.insert(END, f"Platform: {d.get('platform', 'N/A')}\n")
        self.debug_text.insert(END, f"Python: {d.get('python', 'N/A')}\n")
        self.debug_text.insert(END, f"RAM: {d.get('ram', 'N/A')}\n")
        self.debug_text.insert(END, f"CPU: {d.get('cpu_count', 'N/A')} cores\n")
        c = diag.get("cuda", {})
        if c.get("available"):
            self.debug_text.insert(END, f"GPU: {c.get('device', 'N/A')}\n")
            self.debug_text.insert(END, f"VRAM: {c.get('total_vram', 'N/A')}\n")
        else:
            self.debug_text.insert(END, "GPU: None (CPU mode)\n")
        self.debug_text.insert(END, f"\n─── GENERATIONS ───\n")
        for t in diag.get("recent_traces", []):
            icon = "OK" if t.get("success") else "FAIL"
            self.debug_text.insert(END, f"[{icon}] {t.get('model_name','?')} {t.get('mode','?')} {t.get('duration',0):.1f}s {t.get('peak_vram_mb',0):.0f}MB\n")
        self.debug_text.insert(END, f"\n─── MODEL LOADS ───\n")
        for r in diag.get("recent_loads", []):
            s = "OK" if r.get("success") else "FAIL"
            self.debug_text.insert(END, f"[{s}] {r.get('model','?')} {r.get('time',0):.1f}s +{r.get('vram_mb',0):.0f}MB\n")
        self.debug_text.insert(END, f"\n─── PERFORMANCE ───\n")
        p = diag.get("performance", {})
        if p:
            self.debug_text.insert(END, f"Total: {p.get('total_generations', 0)}\n")
            self.debug_text.insert(END, f"Avg: {p.get('avg_duration', 0)}s  Max: {p.get('max_duration', 0)}s\n")
            self.debug_text.insert(END, f"Success: {p.get('success_rate', 'N/A')}\n")

    def _save_debug_log(self):
        path = _debug.save_log()
        self.debug_text.insert(END, f"\nLog saved: {path}\n")

    def _clear_debug(self):
        _debug._traces.clear()
        _debug._model_loads.clear()
        _debug._performance_log.clear()
        self.debug_text.delete("1.0", END)
        self.debug_text.insert(END, "Debug logs cleared.\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def _setup_ssl():
    """Ensure SSL certificates are found in frozen environments."""
    try:
        import certifi
        cacert = certifi.where()
        if os.path.isfile(cacert):
            if "SSL_CERT_FILE" not in os.environ:
                os.environ["SSL_CERT_FILE"] = cacert
            if "REQUESTS_CA_BUNDLE" not in os.environ:
                os.environ["REQUESTS_CA_BUNDLE"] = cacert
    except Exception:
        pass


def _setup_hf_env():
    """Set HuggingFace environment for frozen app."""
    cache_root = Path("E:\\Picture-Aliver\\models\\cache")
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    _setup_ssl()
    _setup_hf_env()

    parser = argparse.ArgumentParser(description="Pic Aliver Desktop App")
    parser.add_argument("--model", "-m", type=str, default=None, help="Pre-load a model")
    args, _ = parser.parse_known_args()

    root = TkinterDnD.Tk()
    app = PicAliverApp(root, preload_model=args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
