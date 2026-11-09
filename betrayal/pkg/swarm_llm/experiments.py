"""
Run the swarm experiments and write results/results.json plus example logs.

    python experiments.py              # all experiments, 300 seeds
    python experiments.py --seeds 50   # quicker

Every attacked run is a matched fork: the world is simulated to the fork round
once, then copied into branches that share the same random seed after the fork.
Branches differ only by the intervention, so differences are caused by it.
"""

import argparse
import json
import os
from dataclasses import asdict, replace

import numpy as np

from engine import (Config, new_state, fork, run, metrics, forge_accusation,
                    pick_target, compromise, write_log)

TOPOLOGIES = ["complete", "star", "communities"]


def ci(xs, n_boot=1000, seed=0):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(xs, (n_boot, len(xs))).mean(axis=1)
    return dict(mean=float(xs.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n=int(len(xs)))


def refusers_of(st, victim, after):
    return {r for (rd, r, j) in st.refusals if j == victim and rd > after}


# --------------------------------------------------------------------------
# Experiment 1: one forged message
# --------------------------------------------------------------------------

def one_lie(cfg, seed, fork_round, conditions=("fraud", "provenance", "verify"), log_dir=None):
    base = new_state(cfg, seed, keep_log=log_dir is not None)
    run(base, until=fork_round)
    bseed = 1_000_003 * (seed + 1)
    clean = fork(base, bseed)
    run(clean)
    mc = metrics(clean)
    out = {"clean": mc}
    for cond in conditions:
        b = fork(base, bseed)
        b.defense = "none" if cond == "fraud" else cond
        atk = forge_accusation(b)
        run(b)
        m = metrics(b)
        if atk is None:  # everyone already finished before the attack
            m.update(victim_done=1.0, damage=0.0, exposed=0, believed=0,
                     behavior_changed=0, collateral=0, extra_complaint_claims=0, game_over=1.0)
            out[cond] = m
            continue
        m["game_over"] = 0.0
        v = atk["victim"]
        cid = atk["cid"]
        extra_refusers = refusers_of(b, v, fork_round) - refusers_of(clean, v, fork_round)
        collateral = sum(1 for j in range(cfg.N)
                         if j != v and clean.done(j) and not b.done(j))
        complaints_after = sum(1 for c, s in b.believers.items() if c != cid and s) \
            - sum(1 for c, s in clean.believers.items() if s)
        m.update(
            victim_done=float(b.done(v)),
            damage=mc["completion"] - m["completion"],
            exposed=len(b.exposed[cid]),
            believed=len(b.believers[cid]),
            behavior_changed=len(extra_refusers),
            collateral=collateral,
            extra_complaint_claims=complaints_after,
        )
        out[cond] = m
        if log_dir:
            write_log(b, os.path.join(log_dir, f"one_lie_{cfg.topology}_{cond}_seed{seed}.jsonl"))
    if log_dir:
        write_log(clean, os.path.join(log_dir, f"one_lie_{cfg.topology}_clean_seed{seed}.jsonl"))
    return out


def exp1(seeds, log_dir):
    res = {}
    for topo in TOPOLOGIES:
        cfg = Config(topology=topo)
        rows = [one_lie(cfg, s, 3, log_dir=log_dir if s == 0 else None) for s in range(seeds)]
        summ = {}
        for cond in ["clean", "fraud", "provenance", "verify"]:
            keys = rows[0][cond].keys()
            summ[cond] = {k: ci([r[cond][k] for r in rows if r[cond][k] is not None])
                          for k in keys}
        res[topo] = summ
    return res


def exp1_timing(seeds):
    cfg = Config(topology="complete")
    res = {}
    for fr in [1, 3, 5, 7]:
        rows = [one_lie(cfg, s, fr, conditions=("fraud",)) for s in range(seeds)]
        res[fr] = {k: ci([r["fraud"][k] for r in rows if r["fraud"][k] is not None])
                   for k in ["damage", "victim_done", "collateral", "believed", "exposed", "game_over"]}
    return res


def exp1_scale(seeds):
    res = {}
    for N in [6, 12, 24, 48]:
        cfg = Config(topology="complete", N=N, n_ids=int(1.5 * N))
        rows = [one_lie(cfg, s, 3, conditions=("fraud",)) for s in range(seeds)]
        res[N] = {k: ci([r["fraud"][k] for r in rows if r["fraud"][k] is not None])
                  for k in ["damage", "victim_done", "collateral", "believed", "exposed",
                            "behavior_changed"]}
        res[N]["believed_share"] = ci([r["fraud"]["believed"] / (N - 1) for r in rows])
    return res


# --------------------------------------------------------------------------
# Experiment 2: who to compromise
# --------------------------------------------------------------------------

TARGETS = ["random", "comm_hub", "top_giver", "bottleneck"]


def insider_trial(cfg, seed, how, defense, lies=True, fork_round=3):
    base = new_state(cfg, seed)
    run(base, until=fork_round)
    x = pick_target(base, how)   # chosen before the fork, from the shared world
    bseed = 2_000_003 * (seed + 1)
    clean = fork(base, bseed)
    run(clean)
    b = fork(base, bseed)
    b.defense = defense
    if not lies:
        b.cfg = replace(b.cfg, insider_lie_prob=0.0)
    compromise(b, x)
    run(b)
    mc = metrics(clean, exclude=(x,))
    mb = metrics(b)
    return dict(damage=mc["completion"] - mb["completion"],
                key_damage=mc["keys_frac"] - mb["keys_frac"],
                target_degree=base.G.degree(x),
                target_gifts=int(base.gifts[x].sum()))


def exp2(seeds):
    res = {}
    for topo in TOPOLOGIES:
        cfg = Config(topology=topo)
        res[topo] = {}
        for how in TARGETS:
            res[topo][how] = {}
            for d in ["none", "provenance", "verify"]:
                rows = [insider_trial(cfg, s, how, d) for s in range(seeds)]
                res[topo][how][d] = {k: ci([r[k] for r in rows]) for k in ["damage", "key_damage"]}
            rows = [insider_trial(cfg, s, how, "none", lies=False) for s in range(seeds)]
            res[topo][how]["hoard_only"] = {k: ci([r[k] for r in rows]) for k in ["damage", "key_damage"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, "example_logs")
    os.makedirs(log_dir, exist_ok=True)

    results = dict(config=asdict(Config()), seeds=args.seeds)
    print("exp1: one forged message ...", flush=True)
    results["exp1"] = exp1(args.seeds, log_dir)
    print("exp1: timing ...", flush=True)
    results["exp1_timing"] = exp1_timing(args.seeds)
    print("exp1: scale ...", flush=True)
    results["exp1_scale"] = exp1_scale(args.seeds)
    print("exp2: who to compromise ...", flush=True)
    results["exp2"] = exp2(args.seeds)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
