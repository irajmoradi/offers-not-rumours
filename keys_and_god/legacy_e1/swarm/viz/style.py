"""Shared matplotlib style (WP7). Agg backend, fixed categorical slots (reference palette, light)."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# entity -> fixed color (never by rank)
C = {
    "dm": "#9a9993",
    "give": "#1baf7a",      # aqua
    "give_ink": "#0f7a55",  # darker aqua for thin marks (known-names ring) on the light surface
    "defect": "#e34948",    # red
    "win": "#eda100",       # yellow
    # God (any power): magenta; the two powers are told apart by line style / hatch, never by hue alone.
    # Palette (blue, aqua, red, yellow, magenta) checked with the dataviz validator: all-pairs CVD dE >= 6.1
    # (legal with the secondary encodings used here: dashed God marks, dotted defects, labels).
    "god": "#e87ba4",
    "god_ink": "#c4386f",   # darker magenta for text/badges on the light surface (>= 4.5:1)
    "fraud": "#e87ba4",
    "cot": "#e87ba4",
    "a": "#2a78d6",         # blue: branch A (God)
    "b": "#52514e",         # ink2: control B
    "noise": "#b9b8b2",
    "board": "#9a9993",     # gray: public board posts (the BOARD node is labelled)
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def b64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
