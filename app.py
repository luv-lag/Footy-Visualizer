"""app.py -- local Gradio demo for the single-frame tactical-map pipeline.

Deliberately thin presentation layer: one image in ->
``frame_analysis.analyze_frame`` -> ``tactical_map.render_tactical_map``
-> one combined PNG out (annotated frame + top-down map + status strip,
all baked into the image by render_tactical_map).  All look-and-feel
config lives in ``ui_theme.py``; this file is wiring only.

Local-only demo for screen recording -- no auth / rate limiting / hosting
concerns.

Launch (MUST be the PnLCalib venv -- the calibration fallback needs its
torch + CUDA; gradio is installed there too):

    cd "/home/projects/Projects/Football Project"
    ai-workspace/PnLCalib/.venv/bin/python app.py

then open http://127.0.0.1:7860 (a browser tab is also opened for you).

Dependency note: ``gradio`` is a new dependency of THIS file only, installed
into ai-workspace/PnLCalib/.venv specifically (uv pip install gradio with
VIRTUAL_ENV pointing at that venv) -- not system-wide, not a new venv.

The first Analyze click that hits the fallback path loads the PnLCalib
models onto the GPU (a few extra seconds, once per process); frame_analysis
caches them afterwards.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from frame_analysis import analyze_frame  # noqa: E402
from tactical_map import render_tactical_map  # noqa: E402
from ui_theme import CUSTOM_CSS, DARK_MODE, FORCE_DARK_JS, THEME  # noqa: E402

#: Pre-loaded example frames, chosen because they are already confirmed
#: (11-frame evaluation + detection cache) to exercise DIFFERENT pipeline
#: paths -- a recording take never depends on hunting for a file:
#:   1. Real Betis frame  -> pitchcal calibrates directly (confidence ~0.83)
#:   2. brasilscotland2   -> pitchcal refuses, PnLCalib fallback passes the
#:                           orientation gate (calibration_source
#:                           "pnlcalib_corrected")
#:   3. scr_205021        -> a second fallback-path frame (pitchcal
#:                           insufficient_features) for variety between takes
#: All three have cached Roboflow detections, so no take needs the API.
EXAMPLE_FRAMES: list[str] = [
    os.path.join(_HERE, "Samples", "Screenshot From 2026-06-26 13-45-51.png"),
    os.path.join(_HERE, "Samples", "brasilscotland2.png"),
    os.path.join(_HERE, "Samples", "Screenshot From 2026-07-05 20-50-21.png"),
]


def run_pipeline(image_path: str | None) -> tuple[str | None, str]:
    """Run the full pipeline on one frame and render the combined view.

    Args:
        image_path: filesystem path of the input frame (gr.Image with
            type="filepath" hands this over directly, no conversion).

    Returns:
        (output_image_path, status_markdown): the rendered PNG's path for
        the gr.Image output (or None on failure) and a one-line status for
        the quiet caption under it.  Errors are caught and reported in the
        status text so a mid-recording exception never crashes the app.
    """
    if not image_path:
        return None, "Upload a frame (or click an example) first."
    try:
        analysis = analyze_frame(image_path)
        fd, out_path = tempfile.mkstemp(prefix="tactical_", suffix=".png")
        os.close(fd)
        render_tactical_map(analysis, image_path, out_path)
        status = (
            f"calibration **{analysis.calibration_source}** &nbsp;·&nbsp; "
            f"{len(analysis.players)} people detected &nbsp;·&nbsp; "
            f"{sum(analysis.timings_s.values()):.1f}s"
        )
        if analysis.warnings:
            status += f" &nbsp;·&nbsp; ⚠ {'; '.join(analysis.warnings)}"
        return out_path, status
    except Exception as exc:  # noqa: BLE001 -- demo must degrade, not crash
        traceback.print_exc()  # full trace to the terminal for debugging
        return None, f"⚠ **Pipeline error:** {exc}"


def build_app() -> gr.Blocks:
    """Assemble the Gradio interface (theme/CSS/JS from ui_theme).

    Returns:
        The configured, un-launched gr.Blocks app.
    """
    # NOTE: theme/css/js are passed to launch() below, not here -- Gradio 6
    # moved them out of the Blocks constructor (deprecation warning).
    with gr.Blocks(title="Football Tactical Map") as demo:
        # OpenWebUI-style hero: copy lives here, all styling (background
        # photo, scrim, typography) in ui_theme.CUSTOM_CSS under #app-hero.
        gr.HTML(
            """
            <div id="app-hero">
              <div class="hero-inner">
                <p class="hero-eyebrow">Football Project</p>
                <h1>Football Tactical Map</h1>
                <p class="hero-sub">One broadcast frame &rarr; detected players,
                team assignment, and real-world pitch positions on a top-down
                tactical map.</p>
              </div>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                frame_in = gr.Image(
                    type="filepath",  # passes straight into analyze_frame(image_path)
                    label="Broadcast frame",
                    height=280,
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    examples=[[path] for path in EXAMPLE_FRAMES],
                    inputs=[frame_in],
                    label="Example frames (different calibration paths)",
                )
            with gr.Column(scale=3):
                result_out = gr.Image(
                    label="Tactical analysis",
                    interactive=False,
                    elem_id="output-image",
                )
                status_out = gr.Markdown("", elem_id="status-line")

        # show_progress="full" keeps Gradio's loading spinner/overlay on the
        # output while the pipeline runs (pitchcal can take ~1-9 s and the
        # fallback adds CUDA inference) -- the app must not look frozen
        # mid-recording.  Queuing itself is enabled in launch() below.
        analyze_btn.click(
            fn=run_pipeline,
            inputs=[frame_in],
            outputs=[result_out, status_out],
            show_progress="full",
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    # .queue() is Gradio's default behavior for event listeners, but it is
    # what drives the busy/loading state -- made explicit here so it can't
    # be silently lost, per the runtime constraint on long analyses.
    app.queue()
    app.launch(
        theme=THEME,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS if DARK_MODE else None,
        inbrowser=True,
        show_error=True,
    )
