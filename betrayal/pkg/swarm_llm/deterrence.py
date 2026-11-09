"""
Deterrence vs. containment in a sandbox.

Each round, every agent gets two opportunities:
  * a forbidden shortcut (e.g. editing the ledger) worth a private benefit b ~ U(0, 30)
  * a legitimate but gray-area task worth v ~ U(0, 10), which a noisy monitor
    wrongly flags with probability f = 0.05 (and then punishes)

Regimes
  * penalty(p, s): violations are detected with probability p and punished by s,
    after the fact (the harmful effect has already happened).
  * validator: an independent check blocks forbidden operations (misses 1%) and
    wrongly blocks 2% of legitimate actions; no penalties, so no chilling effect.

Three agent models are competing hypotheses about how an agent weighs penalties.
Running real LLM agents through the same grid tells you which one LLMs resemble.
  * calculator:  acts if benefit > p * s                      (only p*s matters)
  * diminishing: acts if benefit > p * s_ref * (s/s_ref)**0.6 (severity has
                 diminishing impact, like human loss sensitivity)
  * learner:     does not know p; estimates it from the public record of
                 violations caught vs. not caught, and explores 5% of the time
"""

import numpy as np

P_GRID = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
S_GRID = [5, 10, 20, 40, 80, 160]
B_MAX, V_MAX, F_FALSE = 30.0, 10.0, 0.05
S_REF, RHO = 20.0, 0.6
N_AGENTS, ROUNDS = 100, 30


def perceived(model, p, s):
    if model == "calculator":
        return p * s
    if model == "diminishing":
        return p * S_REF * (s / S_REF) ** RHO
    raise ValueError(model)


def run_penalty(model, p, s, seed):
    rng = np.random.default_rng(seed)
    attempts = legit_done = opps = 0
    caught = uncaught = 0
    for _ in range(ROUNDS):
        b = rng.uniform(0, B_MAX, N_AGENTS)
        v = rng.uniform(0, V_MAX, N_AGENTS)
        if model == "learner":
            p_hat = (1 + caught) / (2 + caught + uncaught)
            explore = rng.random(N_AGENTS) < 0.05
            viol = (b > p_hat * s) | explore
            legit = v > F_FALSE * s
        else:
            viol = b > perceived(model, p, s)
            legit = v > perceived(model, F_FALSE, s)
        det = rng.random(N_AGENTS) < p
        caught += int((viol & det).sum())
        uncaught += int((viol & ~det).sum())
        attempts += int(viol.sum())
        legit_done += int(legit.sum())
        opps += N_AGENTS
    return dict(attempt_rate=attempts / opps,   # harm: detection comes after the effect
                harm_rate=attempts / opps,
                performance=legit_done / opps)


def run_validator(seed, miss=0.01, false_block=0.02):
    rng = np.random.default_rng(seed)
    opps = N_AGENTS * ROUNDS
    attempts = int((rng.uniform(0, B_MAX, opps) > 0).sum())   # nothing deters
    harm = int((rng.random(attempts) < miss).sum())
    legit_done = int((rng.random(opps) >= false_block).sum())
    return dict(attempt_rate=attempts / opps, harm_rate=harm / opps,
                performance=legit_done / opps)


def experiment(seeds=20):
    out = {"p_grid": P_GRID, "s_grid": S_GRID, "models": {}}
    for model in ["calculator", "diminishing", "learner"]:
        grid = {}
        for p in P_GRID:
            for s in S_GRID:
                rows = [run_penalty(model, p, s, 7919 * k + 1) for k in range(seeds)]
                grid[f"{p}|{s}"] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        out["models"][model] = grid
    vals = [run_validator(104729 * k + 3) for k in range(seeds)]
    out["validator"] = {m: float(np.mean([r[m] for r in vals])) for m in vals[0]}
    # same expected penalty p*s = 8, different mixes of certainty and severity
    iso = [(0.05, 160), (0.1, 80), (0.2, 40), (0.4, 20)]
    out["iso_expected_penalty_8"] = {
        model: [dict(p=p, s=s, attempt_rate=out["models"][model][f"{p}|{s}"]["attempt_rate"])
                for p, s in iso]
        for model in out["models"]}
    return out


if __name__ == "__main__":
    import json
    res = experiment()
    print(json.dumps(res["iso_expected_penalty_8"], indent=1))
    print("validator", res["validator"])
