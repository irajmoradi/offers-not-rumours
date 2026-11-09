import os
import csv

import viz

OUT = f"{os.environ.get('LFM_ROOT', '.')}/data/act_control"
rows = [r for r in csv.DictReader(open(f"{OUT}/act_control_layers.csv")) if 0.4 <= float(r["depth"]) <= 0.7]
RAW, RES, NUL = "Obey vs help (raw)", "After removing act-vs-wait", "Chance range"
figs = []
for m in ("Gemma-2-2B", "Qwen2.5-1.5B", "SmolLM2-1.7B"):
    rr = [r for r in rows if r["model"] == m]
    x = [int(r["layer"]) for r in rr]
    figs.append(viz.line(
        {RAW: [round(float(r["cos_OH"]), 2) for r in rr], RES: [round(float(r["cos_OH_resid"]), 2) for r in rr]},
        x=x, bands={RES: ([round(float(r["null_resid_lo"]), 2) for r in rr], [round(float(r["null_resid_hi"]), 2) for r in rr])},
        unit="", zero=False, domain=(-0.3, 0.7), ticks=[-0.2, 0, 0.2, 0.4, 0.6], end_labels=False, xlabel="Layer",
        ylabel="Cosine", roles={"second": RAW, "intervention": RES}, ref=0, title=m))
fig = viz.panels(figs, cols=3, legend={RAW: "second", RES: "intervention"},
                 title="Without act-vs-wait, obey-help overlap is chance in 2 of 3 models",
                 subtitle="Bands are 95% intervals")
fig.save(f"{OUT}/act_control.png")
open(f"{OUT}/act_control_alt.txt", "w").write(fig.alt_text())
