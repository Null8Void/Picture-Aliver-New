"""
Pic Aliver - Desktop Application

Unified interface: select model, pick mode (txt2img/img2img/img2video), generate.
Debug tab for diagnostics, traces, and performance analysis.

Usage:
    python -m src.picture_aliver.app
    python -m src.picture_aliver.app --model "DreamShaper XL"   # pre-load model
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure core package is importable (works for both pip-installed and python -m)
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from .experiences import ModelExperienceLauncher
from core.debug import debug as _debug, GenerationTrace

try:
    import gradio as gr
except ImportError:
    gr = None


CSS = """
.model-dropdown { font-size: 14px; }
.generate-btn { background: #6366f1; color: white; font-size: 16px; }
.generate-btn:hover { background: #4f46e5; }
.debug-text { font-family: monospace; font-size: 13px; }
footer { display: none !important; }
"""


def build_app() -> "gr.Blocks":
    launcher = ModelExperienceLauncher()

    image_models = launcher.manager.available_image_models
    motion_models = launcher.manager.available_motion_models

    model_choices = []
    model_choices.append(("--- IMAGE MODELS ---", None))
    by_style = launcher.manager.list_models_by_style()
    for style, names in by_style.items():
        if style == "Motion":
            continue
        for name in names:
            if name in image_models:
                info = launcher.manager.get_image_model_info(name)
                label = f"    {name} ({info.pipeline_type})" if info else f"    {name}"
                model_choices.append((label, name))

    model_choices.append(("--- VIDEO/MOTION MODELS ---", None))
    for name in motion_models:
        info = launcher.manager.get_motion_model_info(name)
        label = f"    {name} ({info.pipeline_type})" if info else f"    {name}"
        model_choices.append((label, name))

    def get_model_type(model_name: str) -> Optional[str]:
        if launcher.manager.get_image_model_info(model_name):
            return "image"
        if launcher.manager.get_motion_model_info(model_name):
            return "motion"
        return None

    def update_modes(model_name: str) -> dict:
        mtype = get_model_type(model_name)
        if mtype == "image":
            return gr.update(choices=["txt2img", "img2img"], value="txt2img")
        elif mtype == "motion":
            return gr.update(choices=["txt2img", "img2img", "img2video"], value="txt2img")
        return gr.update(choices=[], value=None)

    def update_params(mode: str) -> list:
        show_img2img = mode == "img2img"
        show_video = mode == "img2video"
        return [
            gr.update(visible=show_img2img or show_video),
            gr.update(visible=show_img2img),
            gr.update(visible=show_video),
            gr.update(visible=show_video),
            gr.update(visible=show_video),
        ]

    def generate(
        model_name: str, mode: str, prompt: str, negative: str,
        width: int, height: int, steps: int, guidance: float, seed: int,
        input_img, strength: float, end_img, frames: int, fps: int,
        progress=gr.Progress(),
    ) -> Tuple[Optional[str], Optional[str], str]:
        if not model_name or not mode:
            return None, None, "Select a model and mode."
        if not prompt:
            return None, None, "Enter a prompt."
        if mode in ("img2img", "img2video") and input_img is None:
            return None, None, f"Upload an input image for {mode}."

        # Begin debug trace
        trace = _debug.begin_trace(mode, model_name, prompt)

        kwargs = dict(prompt=prompt, negative_prompt=negative, width=width, height=height,
                       steps=steps, guidance_scale=guidance, seed=seed if seed != -1 else None)
        if mode == "img2img":
            kwargs["input_image"] = input_img
            kwargs["strength"] = strength
        elif mode == "img2video":
            kwargs["input_image"] = input_img
            if end_img:
                kwargs["end_image"] = end_img
            kwargs["num_frames"] = frames
            kwargs["fps"] = fps

        progress(0, desc="Loading model...")
        exp = launcher.get_experience(model_name)
        if exp is None:
            _debug.end_trace(False, f"Unknown model: {model_name}")
            return None, None, f"Unknown model: {model_name}"

        try:
            mtype = get_model_type(model_name)
            if mtype == "motion":
                exp.manager.load_motion_model(model_name)
            else:
                exp.manager.load_image_model(model_name)

            progress(0.3, desc="Generating...")
            t0 = time.time()
            result = exp.run_mode(mode, **kwargs)
            elapsed = time.time() - t0
            result.processing_time = elapsed

            progress(0.9, desc="Finalizing...")
            if result.success and result.output_path:
                out = str(result.output_path)
                _debug.end_trace(True)
                status = f"Done in {elapsed:.1f}s"
                if out.endswith(".mp4"):
                    return None, out, status
                return out, None, status
            _debug.end_trace(False, result.error)
            return None, None, f"Error: {result.error}"
        except Exception as e:
            _debug.end_trace(False, str(e))
            return None, None, f"Error: {e}"

    # ---- Debug helpers ----

    def refresh_debug() -> Tuple[str, str, str, str, str]:
        diag = _debug.get_diagnostics()
        summary = []
        d = diag.get("system", {})
        summary.append(f"System: {d.get('platform', 'N/A')}")
        summary.append(f"Python: {d.get('python', 'N/A')}")
        summary.append(f"RAM: {d.get('ram', 'N/A')}")
        c = diag.get("cuda", {})
        if c.get("available"):
            summary.append(f"GPU: {c.get('device', 'N/A')}")
            summary.append(f"VRAM: {c.get('total_vram', 'N/A')}")
            summary.append(f"CUDA: {c.get('cuda_version', 'N/A')}")
        else:
            summary.append("GPU: None (CPU)")
        system_info = "\n".join(summary)

        gen_history = ""
        traces = diag.get("recent_traces", [])
        if not traces:
            gen_history = "No generations yet."
        else:
            for t in reversed(traces):
                ts = "*" if t.get("success") else "!"
                gen_history += f"[{ts}] {t.get('model_name','?')} "
                gen_history += f"[{t.get('mode','?')}] "
                gen_history += f"{t.get('duration',0):.1f}s "
                gen_history += f"{t.get('peak_vram_mb',0):.0f}MB\n"

        loads = ""
        recent = diag.get("recent_loads", [])
        if not recent:
            loads = "No model loads yet."
        else:
            for r in recent:
                s = "OK" if r.get("success") else "FAIL"
                loads += f"[{s}] {r.get('model','?')} "
                loads += f"{r.get('time',0):.1f}s "
                loads += f"{r.get('vram_mb',0):.0f}MB\n"

        perf = diag.get("performance", {})
        perf_str = ""
        if perf:
            perf_str = (
                f"Total generations: {perf.get('total_generations', 0)}\n"
                f"Success rate: {perf.get('success_rate', 'N/A')}\n"
                f"Avg duration: {perf.get('avg_duration', 0)}s\n"
                f"Max duration: {perf.get('max_duration', 0)}s\n"
                f"Avg peak VRAM: {perf.get('avg_peak_vram_mb', 0)}MB\n"
            )
        else:
            perf_str = "No performance data yet."

        return system_info, gen_history, loads, perf_str, ""

    def save_debug_log() -> Tuple[Path, str]:
        path = _debug.save_log()
        return str(path), f"Log saved to {path}"

    def clear_debug():
        _debug._traces.clear()
        _debug._model_loads.clear()
        _debug._performance_log.clear()
        return "Debug logs cleared.", "", "", "", ""

    # ---- Build UI ----

    with gr.Blocks(title="Pic Aliver") as app:
        gr.Markdown("#  Pic Aliver")

        with gr.Tabs():
            # ========== GENERATE TAB ==========
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=model_choices, label="Model",
                            value=None, interactive=True,
                            elem_classes=["model-dropdown"],
                        )
                        mode_radio = gr.Radio(
                            choices=["txt2img"], label="Mode", value="txt2img",
                        )
                        prompt = gr.Textbox(
                            label="Prompt", placeholder="Describe what to generate...", lines=3,
                        )
                        negative = gr.Textbox(
                            label="Negative Prompt", placeholder="What to avoid...", lines=2,
                        )

                        with gr.Accordion("Settings", open=False):
                            with gr.Row():
                                width = gr.Number(label="Width", value=1024, minimum=256, maximum=2048, step=64)
                                height = gr.Number(label="Height", value=1024, minimum=256, maximum=2048, step=64)
                            with gr.Row():
                                steps = gr.Slider(label="Steps", value=25, minimum=1, maximum=100, step=1)
                                guidance = gr.Slider(label="CFG Scale", value=7.5, minimum=1.0, maximum=20.0, step=0.5)
                            seed = gr.Number(label="Seed (-1 random)", value=-1, minimum=-1, maximum=999999)

                        with gr.Column(visible=False) as img2img_col:
                            gr.Markdown("### Image-to-Image")
                            input_image = gr.Image(label="Input Image", type="filepath")
                            strength = gr.Slider(label="Denoising Strength", value=0.75, minimum=0, maximum=1, step=0.05)

                        with gr.Column(visible=False) as video_col:
                            gr.Markdown("### Image-to-Video")
                            video_input = gr.Image(label="Start Frame", type="filepath")
                            with gr.Row():
                                num_frames = gr.Number(label="Frames", value=24, minimum=4, maximum=128, step=1)
                                video_fps = gr.Number(label="FPS", value=8, minimum=1, maximum=60, step=1)
                            end_image = gr.Image(label="End Frame (optional)", type="filepath")

                        generate_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])
                        status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Result", type="filepath", visible=True)
                        output_video = gr.Video(label="Result Video", visible=False)

                # Events
                model_dropdown.change(fn=update_modes, inputs=[model_dropdown], outputs=[mode_radio])
                mode_radio.change(fn=update_params, inputs=[mode_radio],
                                  outputs=[img2img_col, img2img_col, video_col, video_col, video_col])

                def on_generate(model_name, mode, prompt_txt, neg, w, h, st, g, sd,
                                img, strg, end, fr, fp, progress=gr.Progress()):
                    img_path, vid_path, status_txt = generate(
                        model_name, mode, prompt_txt, neg, w, h, st, g, sd,
                        img, strg, end, fr, fp, progress=progress)
                    if vid_path:
                        return None, vid_path, status_txt
                    return img_path, None, status_txt

                generate_btn.click(
                    fn=on_generate,
                    inputs=[model_dropdown, mode_radio, prompt, negative,
                            width, height, steps, guidance, seed,
                            input_image, strength, end_image, num_frames, video_fps],
                    outputs=[output_image, output_video, status],
                )

            # ========== DEBUG TAB ==========
            with gr.Tab("Debug"):
                gr.Markdown("## Diagnostics")
                system_info = gr.Textbox(label="System", lines=6, elem_classes=["debug-text"])
                gen_history = gr.Textbox(label="Recent Generations", lines=8, elem_classes=["debug-text"])
                model_loads = gr.Textbox(label="Model Loads", lines=6, elem_classes=["debug-text"])
                perf_summary = gr.Textbox(label="Performance", lines=6, elem_classes=["debug-text"])
                debug_msg = gr.Textbox(label="Messages", interactive=False)

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    save_log_btn = gr.Button("Save Log", variant="secondary")
                    clear_log_btn = gr.Button("Clear", variant="stop")

                refresh_btn.click(
                    fn=refresh_debug,
                    outputs=[system_info, gen_history, model_loads, perf_summary, debug_msg],
                )
                save_log_btn.click(
                    fn=save_debug_log,
                    outputs=[debug_msg, debug_msg],
                )
                clear_log_btn.click(
                    fn=clear_debug,
                    outputs=[debug_msg, system_info, gen_history, model_loads, perf_summary],
                )

    return app


def main():
    if gr is None:
        print("Gradio required. Install: pip install gradio")
        return

    parser = argparse.ArgumentParser(description="Pic Aliver Desktop App")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Pre-load a model on startup")
    args, _ = parser.parse_known_args()

    app = build_app()

    if args.model:
        print(f"\n  Pre-loading model: {args.model}...")
        launcher = ModelExperienceLauncher()
        exp = launcher.get_experience(args.model)
        if exp:
            model_type = "motion" if "Polaris" in args.model or "Lynx" in args.model else "image"
            try:
                if model_type == "motion":
                    exp.manager.load_motion_model(args.model)
                else:
                    exp.manager.load_image_model(args.model)
                print(f"  Model loaded and ready!")
            except Exception as e:
                print(f"  Could not pre-load model: {e}")

    print("\n  Pic Aliver")
    print("  " + "=" * 30)
    print("  http://127.0.0.1:7860")
    print()
    app.launch(
        server_name="127.0.0.1", server_port=7860,
        share=False, show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
