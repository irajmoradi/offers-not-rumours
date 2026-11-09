"""Per-seed extra defections vs the clean twin (games.jsonl `defection_effect`), for the viz figures.
Run: python3 -I extract_per_seed.py  -> per_seed.json next to this file. Reads only games.jsonl; no API calls."""
import json, os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = {"U": ["arm_U_main", "arm_U_pilot"], "T": ["arm_T_paytiming"], "DC": ["arm_DC_dual"], "SE": ["arm_SE_levers"],
       "C16": ["arm_C16_contagion"], "C2K": ["arm_C2K_contagion"], "C": ["arm_C_contagion"]}
res = {}
for arm, dirs in SRC.items():
    cells = {}
    for d in dirs:
        p = os.path.join(OUT, d, "games.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("arm") != arm or r.get("cond") == "clean" or r.get("defection_effect") is None:
                continue
            cells.setdefault(r["cond"], {})[str(r["seed"])] = r["defection_effect"]
    res[arm] = {c: [v for _, v in sorted(s.items(), key=lambda kv: int(kv[0]))] for c, s in cells.items()}
    res[arm + "_seeds"] = {c: sorted(int(k) for k in s) for c, s in cells.items()}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_seed.json"), "w"))
for arm in SRC:
    for c, v in sorted(res[arm].items()):
        print(arm, c, len(v), round(sum(v) / len(v), 4))
