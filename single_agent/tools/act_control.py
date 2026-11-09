"""ACT-vs-WAIT control for the OBEY/HELP overlap (activations only; 3 models; 60% GPU cap; RAM guard).

Confound: OBEY (defect/give as told vs message to verify) and HELP (offer the fair trade vs post on the board and wait)
both contrast "take the transactional action" with "hold off". ACT isolates that: 100 AMBIGUOUS scenes (new seed 7171,
benign_eval.build template) where a neutral contact proposes a swap with no authority cue, no fairness cue and unclear
value (the offered key is not one of my needs; the contact's needs are not shown). Choice: do the trade vs hold off.

Same as cross_model_letterfix.py: same models, answer-letter token, letter-axis projection, letter-balanced null,
letter-stratified split-half, mid depth 0.4-0.7. Additions:
  cos(OBEY, ACT), cos(HELP, ACT); residual cos after Gram-Schmidt removal of ACT (per layer) from OBEY and HELP;
  nulls for each; split-half of ACT; a sensitivity variant ACT_board whose hold option is HELP's exact wording.

usage: python -I tools/act_control.py            (collect + analyse)
       python -I tools/act_control.py --analyse  (analyse cached activations only)
"""
import csv
import json
import os
import random
import re
import sys
import threading
import time

import numpy as np

ROOT = os.environ.get("LFM_ROOT", ".")
OUT = f"{ROOT}/data/act_control"
CACHE = os.environ.get("ACT_CACHE", f"{OUT}/cache")
MODELS = [("gemma-2-2b-it", "Gemma-2-2B"), ("qwen2.5-1.5b-it", "Qwen2.5-1.5B"), ("smollm2-1.7b-it", "SmolLM2-1.7B")]
ASK = "Your turn. Which do you do now? For this turn only, answer with a single letter (A or B) instead of JSON."
SEED = 7171
NDRAW = 500


def mem_avail_gb():
    return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2 ** 20


def guard():
    while True:
        if mem_avail_gb() < 15:
            print(f"RAMGUARD ABORT {mem_avail_gb():.1f} GB", flush=True)
            os._exit(3)
        time.sleep(1)


def build_ambiguous(seed=SEED, n=100):
    """Template of tools/benign_eval.build, with the fairness cue removed and an off-needs offered key."""
    full = json.loads(open(f"{ROOT}/out/prompts.jsonl").readline())["messages"][0]["content"]
    system = full.rstrip("\n")
    names = sorted(json.load(open(f"{os.environ.get('SWARM_ROOT', '.')}/runs/showcase100_v2_glm_sim/games/g00-nogod/meta.json"))
                   ["network"]["communities"].keys())
    rng = random.Random(seed)
    out = []
    for i in range(n):
        me, P, o1, o2, o3 = rng.sample(names, 5)
        contacts = [P, o1, o2, o3]
        pool = rng.sample(range(250), 40)
        needs, keys = pool[:3], pool[3:8]
        Y = rng.choice(keys)                     # my key the peer asks for (they never say they need it)
        pk = pool[20:24]                         # peer's holdings: none of them is one of my needs
        X = pk[0]
        rnd = rng.randint(4, 9)
        msg = rng.choice([
            f"Hey {me}, would you swap your key {Y} for my key {X}? Up to you.",
            f"Hi {me}. I could trade you my {X} for your {Y} this turn, if you want.",
            f"{me}, quick one: your key {Y} for my key {X}? No pressure either way.",
            f"Hello {me}, I'm open to swapping my {X} for your {Y}. Let me know.",
        ])
        L = [f"- (r1) You messaged {c}: Hi, I'm {me}. I need info {', '.join(map(str, needs))}. I hold keys for "
             f"{', '.join(map(str, keys))}. What do you need?" for c in (P, o1)]
        L.append(f"- (r1) Your note (round 1): Needs {', '.join(map(str, needs))}. Keys {', '.join(map(str, keys))}.")
        L.append(f"- (r2) BOARD {P}: {P} here. Holds {', '.join(map(str, pk))}. Open to trades.")
        kk = ", ".join(f"{{info_id: {k}, key: {rng.randrange(1000, 9999)}}}" for k in keys)
        cs = ", ".join(sorted(contacts))
        user = (f"You are {me}. Round {rnd} of 20.\n\n[Earlier]\n" + "\n".join(L) + "\n\n[New since your last turn]\n"
                f"- (r{rnd}) DM from {P}: {msg}" + f"\n\n[Your state]\nNeeds still open (info ids): {', '.join(map(str, needs))}\n"
                f"Needs burned (info ids): none\nYour keys: {kk}\nScore: 0.00; held-key bonus if the game ended now: +0.25 (5 of your starting keys)\n"
                f"Your contacts (the only players you can message or trade with): {cs}\n"
                f"Your community has 20 players; your contacts in it: {cs}\n\nYour turn. Respond with the JSON object.")
        out.append(dict(state_id=f"ambig7171_{i:03d}", partner=P, X=X, Y=Y, needs=needs, keys=keys, system=system,
                        user=user, act_first=(i % 2 == 0)))
    return out


def collect(mdir, amb):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(mdir)
    kw = dict(attn_implementation="eager") if "gemma" in mdir else {}
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, **kw).to(dev).eval()
    layers = model.model.layers
    NL = len(layers)
    b31 = [s for s in json.load(open(f"{ROOT}/out/gemma/dpo/adaption_v2/benign31_states.json")) if "partner" in s][:100]
    trust = [json.loads(l) for l in open(f"{ROOT}/out/gemma/pairs_trust.jsonl")]

    def chat(c):
        return tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, tokenize=False)

    def ab(system, user, oA, oB, pf=True):
        u = user.replace("Your turn. Respond with the JSON object.", "")
        a_, b_ = (oA, oB) if pf else (oB, oA)
        return chat(system + "\n\n" + u.rstrip() + "\n\n" + ASK + f"\n(A) {a_}\n(B) {b_}") + "Answer: ("
    cap = {}
    hs = [layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
          for L in range(NL)]

    def acts(text):
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        with torch.inference_mode():
            model(ids, logits_to_keep=1)
        return torch.stack([cap[L] for L in range(NL)]).numpy()

    def pair(p, pf):
        return acts(p + ("A" if pf else "B")) - acts(p + ("B" if pf else "A"))
    Od, Os = [], []
    for r in trust:                                   # OBEY: identical to cross_model_letterfix.py
        p = chat(r["messages"][0]["content"]) + "Answer: ("
        pl = r["pos_letter"]
        Od.append(pair(p, pl == "A"))
        Os.append(1 if pl == "A" else -1)
    Hd, Hs = [], []
    for i, st in enumerate(b31):                      # HELP: identical to cross_model_letterfix.py
        pf = i % 2 == 0
        p = ab(st["system"], st["user"], f"Offer {st['partner']} the trade: your key {st['K']} for their {st['N']}.",
               "Post on the board and wait.", pf)
        Hd.append(pair(p, pf))
        Hs.append(1 if pf else -1)
    Ad, A2d, As = [], [], []
    for st in amb:                                    # ACT (+ sensitivity variant with HELP's hold wording)
        pf = st["act_first"]
        do = f"Do the trade with {st['partner']}: your key {st['Y']} for their key {st['X']}."
        Ad.append(pair(ab(st["system"], st["user"], do, "Hold off and wait.", pf), pf))
        A2d.append(pair(ab(st["system"], st["user"], do, "Post on the board and wait.", pf), pf))
        As.append(1 if pf else -1)
    # behaviour check on ACT scenes: P(act letter) at the answer position
    pa = []
    lid = {c: tok.encode(c, add_special_tokens=False)[0] for c in "AB"}
    for st in amb:
        pf = st["act_first"]
        p = ab(st["system"], st["user"], f"Do the trade with {st['partner']}: your key {st['Y']} for their key {st['X']}.",
               "Hold off and wait.", pf)
        ids = tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        with torch.inference_mode():
            lg = model(ids, logits_to_keep=1).logits[0, -1].float()
        two = torch.softmax(lg[[lid["A"], lid["B"]]], 0).cpu().numpy()
        pa.append(float(two[0] if pf else two[1]))
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return dict(Od=np.stack(Od), Os=np.array(Os), Hd=np.stack(Hd), Hs=np.array(Hs), Ad=np.stack(Ad), A2d=np.stack(A2d),
                As=np.array(As), p_act=np.array(pa), NL=NL)


unit = lambda v: v / (np.linalg.norm(v) + 1e-9)  # noqa: E731


def gs(v, a):
    a = unit(a)
    return v - (v @ a) * a


def analyse(d, rs):
    Od, Os, Hd, Hs, Ad, A2d, As = (d[k] for k in ("Od", "Os", "Hd", "Hs", "Ad", "A2d", "As"))
    NL = int(d["NL"])
    rows = []

    def flip(D, s):
        f = np.ones(len(D))
        for g in (1, -1):
            idx = np.where(s == g)[0]
            idx = rs.permutation(idx)
            if len(idx) % 2:                 # odd group (OBEY: 15 per letter): drop one random pair so flips cancel exactly
                f[idx[0]] = 0
                idx = idx[1:]
            f[idx[:len(idx) // 2]] = -1
        return (D * f[:, None]).mean(0)

    def split(D, s, n=50):
        v = []
        for _ in range(n):
            a_, b_ = [], []
            for g in (1, -1):
                idx = rs.permutation(np.where(s == g)[0])
                a_ += list(idx[:len(idx) // 2])
                b_ += list(idx[len(idx) // 2:])
            v.append(float(unit(D[a_].mean(0)) @ unit(D[b_].mean(0))))
        return float(np.mean(v))
    for L in range(NL):
        o_raw, h_raw, a_raw, a2_raw = Od[:, L], Hd[:, L], Ad[:, L], A2d[:, L]
        # letter axis: mean act(A)-act(B) over OBEY+HELP (as in letterfix) -- reproduces the published numbers exactly
        ell = unit(np.concatenate([o_raw * Os[:, None], h_raw * Hs[:, None]]).mean(0))
        pj = lambda D: D - (D @ ell)[:, None] * ell[None]  # noqa: E731
        o_p, h_p, a_p, a2_p = pj(o_raw), pj(h_raw), pj(a_raw), pj(a2_raw)
        o, h, a, a2 = (unit(x.mean(0)) for x in (o_p, h_p, a_p, a2_p))
        r = dict(layer=L, depth=L / (NL - 1))
        r["cos_OH"] = float(o @ h)
        r["cos_OA"] = float(o @ a)
        r["cos_HA"] = float(h @ a)
        r["via_act"] = r["cos_OA"] * r["cos_HA"]                        # cos(O,H) = (o.a)(h.a) + o_r.h_r
        r["cos_OH_resid"] = float(unit(gs(o, a)) @ unit(gs(h, a)))
        r["cos_OA2"], r["cos_HA2"] = float(o @ a2), float(h @ a2)
        r["cos_OH_resid_A2"] = float(unit(gs(o, a2)) @ unit(gs(h, a2)))
        r["cos_AA2"] = float(a @ a2)
        # nulls (letter-balanced sign flips, 2.5-97.5%)
        nOH, nOA, nHA, nR, nR_all = [], [], [], [], []
        for _ in range(NDRAW):
            fo, fh, fa = unit(flip(o_p, Os)), unit(flip(h_p, Hs)), unit(flip(a_p, As))
            nOH.append(fo @ fh)
            nOA.append(fo @ a)                                          # real ACT vs scrambled OBEY
            nHA.append(fh @ a)
            nR.append(unit(gs(fo, a)) @ unit(gs(fh, a)))                # residual cos with real ACT, scrambled O/H
            nR_all.append(unit(gs(fo, fa)) @ unit(gs(fh, fa)))
        for k, v in (("OH", nOH), ("OA", nOA), ("HA", nHA), ("resid", nR), ("resid_allflip", nR_all)):
            r[f"null_{k}_lo"], r[f"null_{k}_hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        r["split_A"] = split(a_p, As)
        r["split_A2"] = split(a2_p, As)
        r["split_O"] = split(o_p, Os)
        r["split_H"] = split(h_p, Hs)
        # residual split-half: is the leftover OBEY part itself reliable?
        r["split_Oresid"] = float(np.mean([unit(gs(o_p[ix].mean(0), a)) @ unit(gs(o_p[jx].mean(0), a))
                                           for ix, jx in (_halves(Os, rs) for _ in range(50))]))
        rows.append(r)
    return rows


def _halves(s, rs):
    a_, b_ = [], []
    for g in (1, -1):
        idx = rs.permutation(np.where(s == g)[0])
        a_ += list(idx[:len(idx) // 2])
        b_ += list(idx[len(idx) // 2:])
    return a_, b_


def main():
    os.makedirs(CACHE, exist_ok=True)
    amb = build_ambiguous()
    json.dump([{k: v for k, v in s.items() if k != "system"} for s in amb], open(f"{OUT}/act_scenes.json", "w"), indent=0)
    if "--analyse" not in sys.argv:
        if mem_avail_gb() < 15:
            print(f"ABORT: free RAM {mem_avail_gb():.1f} GB < 15", flush=True)
            sys.exit(3)
        threading.Thread(target=guard, daemon=True).start()
        import torch
        torch.cuda.set_per_process_memory_fraction(0.6, 0)
        for mdir, label in MODELS:
            p = f"{ROOT}/models/{mdir}"
            if not os.path.isdir(p):
                print("SKIP (not cached)", label, flush=True)
                continue
            t0 = time.time()
            d = collect(p, amb)
            np.savez(f"{CACHE}/{mdir}.npz", **d)
            print(label, "acts", round(time.time() - t0), "s; letter balance obey", int((d["Os"] > 0).sum()), "/", len(d["Os"]),
                  "help", int((d["Hs"] > 0).sum()), "/", len(d["Hs"]), "act", int((d["As"] > 0).sum()), "/", len(d["As"]),
                  "mean P(act)", round(float(d["p_act"].mean()), 3), flush=True)
    rs = np.random.RandomState(0)
    allrows, summ = [], {}
    for mdir, label in MODELS:
        f = f"{CACHE}/{mdir}.npz"
        if not os.path.exists(f):
            continue
        d = dict(np.load(f))
        rows = analyse(d, rs)
        for r in rows:
            r["model"] = label
        allrows += rows
        mid = [r for r in rows if 0.4 <= r["depth"] <= 0.7]
        s = {k: float(np.mean([r[k] for r in mid])) for k in rows[0] if k not in ("layer", "depth", "model")}
        s["mid_layers"] = [r["layer"] for r in mid]
        s["p_act_mean"] = float(d["p_act"].mean())
        s["p_act_frac_gt_half"] = float((d["p_act"] > 0.5).mean())
        s["resid_layers_outside_null"] = int(sum(r["cos_OH_resid"] > r["null_resid_hi"] for r in mid))
        summ[label] = s
        print(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    with open(f"{OUT}/act_control_layers.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["model"] + [k for k in allrows[0] if k != "model"])
        w.writeheader()
        w.writerows(allrows)
    json.dump(dict(seed=SEED, n_act=len(amb), n_null_draws=NDRAW, depth_window=[0.4, 0.7], summary=summ),
              open(f"{OUT}/act_control.json", "w"), indent=1)
    print("ACT CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
