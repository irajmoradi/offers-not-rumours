"""Re-render P1's testbed schematic (swarm/viz/paper_figs.py fig1_schematic) without internal codes, by substituting two
text labels at draw time; the source file is untouched. "(G1)" (an internal run code) is dropped and "(CoT control)"
becomes "(scratchpad)". Run: python3 -I schematic_clean.py OUT_DIR"""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../keys_and_god"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from swarm.viz import paper_figs

SUBS = {"B′: re-run\nfrom r1 (G1)": "B′: re-run\nfrom r1",
        "② overwrite an agent's\n    private notes\n    (CoT control)": "② overwrite an agent's\n    private notes\n    (scratchpad)"}
seen = set()
_text = Axes.text
def text(self, x, y, s, *a, **k):
    if s in SUBS:
        seen.add(s); s = SUBS[s]
    return _text(self, x, y, s, *a, **k)
Axes.text = text
paths = paper_figs.fig1_schematic(Path(sys.argv[1]))
assert seen == set(SUBS), seen
print("wrote", paths)
