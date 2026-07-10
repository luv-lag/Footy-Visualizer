"""ui_theme.py -- aesthetic config for the tactical-map demo app.

Everything about the app's LOOK lives here, isolated from app.py's wiring,
in two clearly separated places:

  1. The named constants just below + the ``THEME`` object they drive
     ("change a color / opacity / blur" edits happen ONLY here).
  2. ``CUSTOM_CSS`` at the bottom, for layout details Gradio's theme API
     can't reach ("adjust layout CSS" edits happen ONLY there; every
     tunable value inside it is injected from the constants).

Design direction (v2, after seeing v1 run): the colorized historical photo
runs FIXED behind the whole page (like the OpenWebUI reference), heavily
knocked back so it reads as atmosphere; content sits on translucent dark
"glass" cards; the accent is white/off-white (equally neutral against the
tactical map's own blue/magenta/red/yellow as the old teal was); chrome is
minimal and contextual -- borders and labels whisper, and only strengthen
in response to actual interaction.
"""

from __future__ import annotations

import base64
import os

import gradio as gr

# ══════════════════════════════════════════════════════════════════════════
# EDIT THESE to restyle the whole app
# ══════════════════════════════════════════════════════════════════════════

#: Primary accent (hex): buttons, hover/active states, focus rings.
#: Off-white -- deliberately neutral against the rendered tactical map's
#: own team/referee/ball colors (blue, magenta, red, orange, yellow), the
#: same constraint that originally ruled out red/orange/blue/magenta hues.
ACCENT_COLOR: str = "#f2f3f1"

#: Accent hover state (slightly dimmed so hover is perceptible on white).
ACCENT_COLOR_HOVER: str = "rgba(242, 243, 241, 0.82)"

#: Text color used ON accent-filled surfaces (near-black on off-white).
ON_ACCENT_COLOR: str = "#10151a"

#: Neutral (background/border) palette name from ``gr.themes.colors``.
NEUTRAL_COLOR: str = "slate"

#: Force the app into dark mode (the sports-broadcast look this demo wants).
DARK_MODE: bool = True

#: Font family.  Loaded as a Google Font with system-ui fallbacks, so the
#: app still renders fine (just less pretty) with no internet.
FONT: str = "Inter"

#: Corner roundness preset exposed by Gradio themes: "none" | "sm" | "md" | "lg".
RADIUS_SIZE: str = "lg"

#: Spacing density preset exposed by Gradio themes: "sm" | "md" | "lg".
SPACING_SIZE: str = "md"

#: Base page color, visible only where the photo hasn't painted yet.
BASE_BG: str = "#0c1116"

# ── full-page background photo ────────────────────────────────────────────
#: The colorized (AI-restored) historical photograph, shipped alongside the
#: app. Busy and warm-toned, so it gets a double knock-back (filter + scrim
#: below) to read as a subtle atmospheric backdrop behind the WHOLE page.
HERO_BACKGROUND_IMAGE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "background_image.png"
)

#: Photo layer filter: desaturate + darken + soften so detail recedes.
BG_SATURATE: float = 0.45
BG_BRIGHTNESS: float = 0.60
BG_BLUR_PX: float = 1.5

#: Full-page scrim over the photo (RGB + opacity).  More aggressive than a
#: standard hero gradient because text/cards sit over the image everywhere.
OVERLAY_RGB: str = "9, 12, 16"
OVERLAY_OPACITY_TOP: float = 0.78     # behind the hero text
OVERLAY_OPACITY_BOTTOM: float = 0.86  # behind the working UI further down

# ── card "glass" surfaces ─────────────────────────────────────────────────
#: Semi-transparent dark fill for card-like surfaces (upload card, output
#: panel, examples) so they read clearly without fully hiding the photo.
CARD_BG_RGBA: str = "rgba(15, 20, 26, 0.68)"

#: backdrop-filter blur behind cards.  Needs a Chromium/WebKit-family
#: browser; harmlessly ignored elsewhere (the rgba fill still carries
#: legibility on its own).
CARD_BLUR_PX: int = 10

# ── contextual chrome ─────────────────────────────────────────────────────
#: Border alpha for cards at rest (whisper) vs during interaction
#: (hover / focus / drag) -- the border is a cue, not permanent framing.
BORDER_IDLE_ALPHA: float = 0.10
BORDER_ACTIVE_ALPHA: float = 0.45

#: Quiet placeholder shown in the empty output panel so the empty state
#: reads as intentional, not broken.  Disappears once a result renders.
EMPTY_STATE_TEXT: str = "Results will appear here"
EMPTY_STATE_ALPHA: float = 0.35


# ══════════════════════════════════════════════════════════════════════════
# Theme object (built from a stock base + .set overrides, not hand-rolled)
# ══════════════════════════════════════════════════════════════════════════
def _palette(name: str) -> gr.themes.Color:
    """Resolve a palette name from the constants above to a theme Color.

    Args:
        name: attribute name inside ``gr.themes.colors``.

    Returns:
        The corresponding :class:`gr.themes.Color` palette.

    Raises:
        AttributeError: if the name is not a real Gradio palette (fail loud
            at import time rather than silently theming wrong).
    """
    return getattr(gr.themes.colors, name)


#: The app theme: stock Soft base; hue stays neutral and the visible accent
#: (buttons, focus, active states) is driven explicitly by ACCENT_COLOR.
#: Card surfaces are translucent AT THE THEME LEVEL so every Gradio-drawn
#: panel participates in the glass look, not just the ones CSS names.
THEME: gr.themes.Base = gr.themes.Soft(
    primary_hue=_palette(NEUTRAL_COLOR),
    neutral_hue=_palette(NEUTRAL_COLOR),
    radius_size=RADIUS_SIZE,
    spacing_size=SPACING_SIZE,
    font=[gr.themes.GoogleFont(FONT), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill_dark=BASE_BG,
    # translucent glass cards (backdrop blur is added in CUSTOM_CSS):
    block_background_fill_dark=CARD_BG_RGBA,
    panel_background_fill_dark=CARD_BG_RGBA,
    block_border_width="1px",
    block_border_color_dark=f"rgba(255, 255, 255, {BORDER_IDLE_ALPHA})",
    # white accent: buttons, active/hover, focus rings:
    color_accent=ACCENT_COLOR,
    color_accent_soft_dark="rgba(255, 255, 255, 0.10)",
    border_color_accent_dark=ACCENT_COLOR,
    input_border_color_focus_dark=ACCENT_COLOR,
    button_primary_background_fill_dark=ACCENT_COLOR,
    button_primary_background_fill_hover_dark=ACCENT_COLOR_HOVER,
    button_primary_text_color_dark=ON_ACCENT_COLOR,
    # section labels: quiet text, no filled pill:
    block_label_background_fill_dark="transparent",
    block_label_text_color_dark="rgba(255, 255, 255, 0.55)",
)

#: JS snippet app.py attaches when DARK_MODE is True: rewrites the URL once
#: with ``?__theme=dark`` so Gradio serves the dark variant unconditionally.
FORCE_DARK_JS: str = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url);
  }
}
"""


def _hero_bg_css_url(image_path: str) -> str:
    """Encode the background image as a CSS ``url(data:...)`` value.

    A data URI keeps the app fully self-contained -- no Gradio static-path
    allowlisting, no URL-escaping of the space in the filename.

    Args:
        image_path: path to the background image on disk.

    Returns:
        A string usable directly inside a CSS ``background-image`` rule.
    """
    with open(image_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"url(data:image/png;base64,{encoded})"


# ══════════════════════════════════════════════════════════════════════════
# Layout CSS the theme object can't express (kept separate on purpose)
# ══════════════════════════════════════════════════════════════════════════
# Every ``__TOKEN__`` below is injected from the named constants at the
# bottom of this file -- tune the constants, not the literals here.
CUSTOM_CSS: str = """
/* Let the combined output image dominate: wide page, thin margins. */
.gradio-container { max-width: 1780px !important; margin: 0 auto; }

/* ── full-page fixed background (OpenWebUI-style) ─────────────────────
   Two fixed pseudo-element layers on <body>: the photo (filtered so its
   busy warm detail recedes) and a strong scrim above it.  Gradio's own
   containers are made transparent so the layers show through everywhere
   -- hero, cards, examples -- staying anchored while the page scrolls. */
gradio-app, .gradio-container { background: transparent !important; }
body { background: __BASE_BG__; }
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -2;
  background-image: __HERO_BG_URL__;
  background-size: cover;
  background-position: center 32%;   /* keep the players, crop the film border */
  filter: saturate(__BG_SATURATE__) brightness(__BG_BRIGHTNESS__) blur(__BG_BLUR_PX__px);
  transform: scale(1.04);            /* hide blur-softened edges */
}
body::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  background: linear-gradient(to bottom,
    rgba(__OVERLAY_RGB__, __OVERLAY_TOP__) 0%,
    rgba(__OVERLAY_RGB__, __OVERLAY_BOTTOM__) 100%);
}

/* ── hero: typography only, fully transparent over the page layers ──── */
#hero-block { background: transparent !important; border: none !important; box-shadow: none !important; }
#app-hero {
  min-height: 290px;
  display: flex;
  align-items: flex-end;
}
#app-hero .hero-inner { padding: 2.2rem 2.4rem 1.6rem; }
#app-hero .hero-eyebrow {
  font-size: 0.85rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.6);
  margin: 0 0 0.35rem;
}
#app-hero h1 {
  font-size: 2.3rem;
  font-weight: 650;
  color: #fff;
  margin: 0 0 0.45rem;
  letter-spacing: 0.01em;
  text-shadow: 0 1px 12px rgba(0, 0, 0, 0.45);
}
#app-hero .hero-sub {
  font-size: 1.0rem;
  color: rgba(255, 255, 255, 0.78);
  margin: 0;
  max-width: 46rem;
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.45);
}

/* ── glass cards: translucent fill comes from the theme; the blur that
      makes them read against the photo is added here. ─────────────────── */
#frame-input, #output-image, #example-frames {
  backdrop-filter: blur(__CARD_BLUR__px);
  -webkit-backdrop-filter: blur(__CARD_BLUR__px);
}

/* ── minimal, contextual chrome ───────────────────────────────────────
   Section labels: plain quiet text, no pill, no icon. */
.block-label {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.block-label svg { display: none; }

/* Upload card: whisper border at rest; strengthens only as an
   interaction cue (hover / keyboard focus / file drag). */
#frame-input {
  border: 1px dashed rgba(255, 255, 255, __BORDER_IDLE__) !important;
  transition: border-color 120ms ease;
}
#frame-input:hover,
#frame-input:focus-within,
#frame-input:has(.dragging) {
  border-color: rgba(255, 255, 255, __BORDER_ACTIVE__) !important;
}
/* Gradio draws its own dashed box inside the dropzone; one border is enough. */
#frame-input .upload-container, #frame-input button { border: none !important; }

/* Output panel, EMPTY state: open space, not a bordered box -- but with a
   quiet, deliberate placeholder so it can't be mistaken for a rendering
   error.  :has(.empty) matches only while Gradio shows its empty shell,
   so everything here vanishes automatically once a result renders. */
#output-image:has(.empty) {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}
#output-image .empty svg { display: none; }   /* stock placeholder icon */
#output-image .empty::after {
  content: "__EMPTY_TEXT__";
  color: rgba(255, 255, 255, __EMPTY_ALPHA__);
  font-size: 0.95rem;
}

/* Example thumbnails: no border/label chrome; hover glow is the
   affordance that they're clickable. */
#example-frames { border: none !important; }
#example-frames .gallery-item, #example-frames button {
  border: none !important;
  background: transparent;
  transition: box-shadow 120ms ease, transform 120ms ease;
}
#example-frames .gallery-item:hover, #example-frames button:hover {
  box-shadow: 0 0 0 1px rgba(255, 255, 255, __BORDER_ACTIVE__);
}

/* Keep the status line quiet -- the PNG's own status strip is the loud one. */
#status-line { opacity: 0.75; font-size: 0.9rem; min-height: 1.4em; background: transparent !important; border: none !important; }

/* Hide Gradio's footer chrome for clean screen recordings. */
footer { visibility: hidden; }
"""

# Inject every tunable value (kept out of the literal above so the CSS
# stays readable and the constants stay the single source of truth).
_CSS_TOKENS: dict[str, str] = {
    "__HERO_BG_URL__": _hero_bg_css_url(HERO_BACKGROUND_IMAGE),
    "__BASE_BG__": BASE_BG,
    "__BG_SATURATE__": str(BG_SATURATE),
    "__BG_BRIGHTNESS__": str(BG_BRIGHTNESS),
    "__BG_BLUR_PX__": str(BG_BLUR_PX),
    "__OVERLAY_RGB__": OVERLAY_RGB,
    "__OVERLAY_TOP__": str(OVERLAY_OPACITY_TOP),
    "__OVERLAY_BOTTOM__": str(OVERLAY_OPACITY_BOTTOM),
    "__CARD_BLUR__": str(CARD_BLUR_PX),
    "__BORDER_IDLE__": str(BORDER_IDLE_ALPHA),
    "__BORDER_ACTIVE__": str(BORDER_ACTIVE_ALPHA),
    "__EMPTY_TEXT__": EMPTY_STATE_TEXT,
    "__EMPTY_ALPHA__": str(EMPTY_STATE_ALPHA),
}
for _token, _value in _CSS_TOKENS.items():
    CUSTOM_CSS = CUSTOM_CSS.replace(_token, _value)
