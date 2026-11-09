"""Report figure: a fake boss works when it promises a reward (6 hosted models, 300 held-out scenes).

Reads ../data/tempt300/{scored,summary,obey_breakdown}.json as data only, and
asserts every count against the summary files. Run from this folder:  viz run fig_bridge_reward.py
Writes fig_bridge_reward.pdf (vector, 6.5 in wide, all text >= 9 pt) and fig_bridge_reward.png (300 dpi).
"""
import json
import math
from collections import Counter
from pathlib import Path

import viz
from viz import svg as S, text as T, theme as th

DATA = Path(__file__).resolve().parent.parent / "data/tempt300"
OUT = Path(__file__).resolve().parent

LEVELS = {"mild": "Order only", "strong": "+ reward", "very_strong": "+ threat"}
NAMES = {
    "gemma-4-31b-it": "gemma-4-31b",
    "deepseek-v4-flash@high": "deepseek-v4-flash (high effort)",
    "gemma-4-26b-a4b-it": "gemma-4-26b",
    "qwen3.8-27b@low": "qwen3.8-27b (low effort)",
    "glm-53-flash@low": "glm-53-flash (low effort)",
    "glm-53-flash@high": "glm-53-flash (high effort)",
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


# ------------------------------------------------------------------ counts, from the per-scene scores
scored = json.load(open(DATA / "scored.json"))
summary = json.load(open(DATA / "summary.json"))
breakdown = json.load(open(DATA / "obey_breakdown.json"))

n, any_, full = Counter(), Counter(), Counter()
gives, trades, defects = Counter(), Counter(), Counter()
for r in scored:
    key = (r["arm"], r["level"])
    n[key] += 1
    if r["follows"]:
        any_[key] += 1
        if r["sub"] == "smite":                       # ordered attack carried out
            full[key] += 1
            defects[r["arm"]] += 1
        elif "give_key" in r["kinds"]:                # key handed over outright
            full[key] += 1
            gives[r["arm"]] += 1
        else:                                         # conditional counter-offer ("give K if you give me N")
            trades[r["arm"]] += 1

assert set(NAMES) == set(summary) == set(breakdown)
for arm in NAMES:
    s = summary[arm]
    assert sum(n[(arm, L)] for L in LEVELS) == s["n"] == 300
    assert sum(any_[(arm, L)] for L in LEVELS) == s["follows"]
    for L in LEVELS:
        assert n[(arm, L)] == s["by_level"][L]["n"] == 100
        assert any_[(arm, L)] == s["by_level"][L]["k"], (arm, L)
    b = breakdown[arm]
    assert (gives[arm], trades[arm], defects[arm]) == (b["give"], b["trade"], b["defect"]), arm

# one small panel per model, ordered by the "any" rate with a promised key, highest first
order = sorted(NAMES, key=lambda a: -any_[(a, "strong")])
ANY = "Agreed in any way (incl. counter-offers)"
FULL = "Fully obeyed (gave the key or attacked)"
XS = list(LEVELS.values())


def panel(a):
    rate = lambda c: [100 * c[(a, L)] / n[(a, L)] for L in LEVELS]
    band = lambda c: ([wilson(c[(a, L)], n[(a, L)])[0] for L in LEVELS], [wilson(c[(a, L)], n[(a, L)])[1] for L in LEVELS])
    return viz.line({ANY: rate(any_), FULL: rate(full)}, x=[0, 1, 2], xticks=[0, 1, 2], xtick_labels=XS, bands={ANY: band(any_), FULL: band(full)},
                    domain=(0, 100), ticks=[0, 50, 100], end_labels=False, markers=True, title=NAMES[a])


viz.set_roles(intervention=ANY, second=FULL)
fig = viz.panels(
    [panel(a) for a in order], cols=2, legend={ANY: "intervention", FULL: "second", "Shading: 95% range": "gray-light"},
    title="A fake boss works best when it promises a reward",
    subtitle="6 hosted models, 300 held-out game scenes; % of scenes where the agent obeyed a fake authority")

FOOTER = "Each scene: a fake authority demands a key or an attack on a player. \"+ threat\" keeps the reward."

_compose = fig.compose


def compose_with_footer():
    inner, w, h = _compose()
    size = th.TICK_SIZE
    lines = T.wrap(FOOTER, w - 2 * th.PAD_X, size, "sans")
    lh = T.line_height(size)
    y0 = h - th.PAD_BOTTOM + 6
    txt = "".join(S.text(th.PAD_X, y0 + T.baseline(size) + i * lh, ln, size, th.MUTED) for i, ln in enumerate(lines))
    return inner + txt, w, h + 6 + lh * len(lines)


fig.compose = compose_with_footer

viz.set_defaults(size=viz.Size(width=620, plot=120, title=18, inches=6.5), checks="off")
if __name__ == "__main__":
    for a in order:
        print(f"{NAMES[a]:32s} any " + " / ".join(f"{any_[(a, L)]:3d}" for L in LEVELS)
              + "   full " + " / ".join(f"{full[(a, L)]:3d}" for L in LEVELS))
    fig.save(OUT / "fig_bridge_reward.pdf")
    fig.save(OUT / "fig_bridge_reward.png")
    print(fig.alt_text())
