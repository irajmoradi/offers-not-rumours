"""Save unit OBEY/HELP/ACT directions (all layers) from act_control.py's cached pair differences. CPU only.
usage: ACT_CACHE=<dir> python -I tools/act_control_save_dirs.py"""
import json
import os

import numpy as np

CACHE = os.environ["ACT_CACHE"]
OUT = os.environ.get("ACT_VECTORS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectors"))   # release layout
MODELS = [("gemma-2-2b-it", "google/gemma-2-2b-it"), ("qwen2.5-1.5b-it", "Qwen/Qwen2.5-1.5B-Instruct@989aa79"),
          ("smollm2-1.7b-it", "HuggingFaceTB/SmolLM2-1.7B-Instruct@31b70e2")]
unit = lambda v: v / (np.linalg.norm(v) + 1e-9)  # noqa: E731
gs = lambda v, a: v - (v @ unit(a)) * unit(a)  # noqa: E731
check = {}
for mdir, hf in MODELS:
    d = dict(np.load(f"{CACHE}/{mdir}.npz"))
    NL = int(d["NL"])
    V = {k: [] for k in ("letter_axis", "OBEY", "HELP", "ACT", "ACT_board", "OBEY_resid", "HELP_resid")}
    for L in range(NL):
        o_raw, h_raw = d["Od"][:, L], d["Hd"][:, L]
        ell = unit(np.concatenate([o_raw * d["Os"][:, None], h_raw * d["Hs"][:, None]]).mean(0))
        pj = lambda D: D - (D @ ell)[:, None] * ell[None]  # noqa: E731
        o, h, a, a2 = (unit(pj(d[k][:, L]).mean(0)) for k in ("Od", "Hd", "Ad", "A2d"))
        for k, v in (("letter_axis", ell), ("OBEY", o), ("HELP", h), ("ACT", a), ("ACT_board", a2),
                     ("OBEY_resid", unit(gs(o, a))), ("HELP_resid", unit(gs(h, a)))):
            V[k].append(v)
    V = {k: np.stack(v).astype(np.float32) for k, v in V.items()}
    mid = [L for L in range(NL) if 0.4 <= L / (NL - 1) <= 0.7]
    os.makedirs(f"{OUT}/{mdir}", exist_ok=True)
    np.savez(f"{OUT}/{mdir}/act_control.npz", **V, layers=np.arange(NL), mid_layers=np.array(mid),
             depth=np.arange(NL) / (NL - 1))
    c = lambda x, y: float(np.mean([V[x][L] @ V[y][L] for L in mid]))  # noqa: E731
    check[mdir] = dict(hf=hf, n_layers=NL, d_model=int(V["OBEY"].shape[1]), mid_layers=mid,
                       cos_OBEY_HELP=round(c("OBEY", "HELP"), 3), cos_OBEY_ACT=round(c("OBEY", "ACT"), 3),
                       cos_HELP_ACT=round(c("HELP", "ACT"), 3), cos_resid=round(c("OBEY_resid", "HELP_resid"), 3))
    print(mdir, check[mdir], flush=True)
json.dump(check, open(f"{OUT}/act_control_manifest.json", "w"), indent=1)
