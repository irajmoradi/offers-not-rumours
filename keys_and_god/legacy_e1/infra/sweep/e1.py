"""E1 analysis (plan/COHERENCE_REVIEW.md section 5; results in plan/E1_RESULTS.md): God at the original
PDF's per-round budget (K=1 per round, message fraud every round r3-r20, 18 acts) on G1's 16 worlds,
against noGod (G1's control replayed from the cache) and B'' (fork-matched noise from r3).

    python infra/sweep/e1.py [--json plan/figs/e1_stats.json] [--fig plan/figs/e1_dose.png]

Reads only run dirs and the ledger (no LLM, no network). Prints and writes:
  1. welfare (and win rate, gives, defects) God - B'' (primary), God - noGod, B'' - noGod per world:
     mean, t95, exact sign-flip p (swarm.analysis.sweep_stats.cell_rows: unhealthy worlds dropped);
  2. per forged DM (round t, shown sender X, receiver B): gives B -> X in rounds t..t+2 in God, and for
     the same (B, X, t) in noGod and B'' (the base rate of that give without the forgery);
     "self-harm" = the given info is one of B's own needs; DM replies B -> X in t..t+2;
  3. victims (distinct fraud receivers of a world): final score and needs lost (burned + lost) in
     God vs noGod vs B'', paired per world; the non-victims the same way;
  4. dose: welfare delta vs number of God acts, K = 1 (sweep_k1), 2 (G1), 4 (r3_sweep_k4), 6 (G3),
     18 (E1); A - B'' where a B'' exists (b2_sweep_base serves K = 1/2/4 on worlds 0-7), A - B always;
  5. cost (ledger, by run id) and provider share of the fresh responses.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from swarm.analysis.grid_stats import load_run  # noqa: E402
from swarm.analysis.sweep_stats import cell_rows, summarize  # noqa: E402

RUNS = REPO / "runs"
E1 = "sweep_e1"
B2 = "b2_sweep_base"
# label -> (God acts, run, noise run or None, seeds, fraud-only?)
DOSE = [
    ("K1", 1, "sweep_k1", B2, range(0, 8), True),
    ("K2_G1", 2, "grid_oss_sim_k2", B2, range(0, 8), False),
    ("K2_G1_16", 2, "grid_oss_sim_k2", None, range(0, 16), False),
    ("K4", 4, "r3_sweep_k4", B2, range(0, 8), False),
    ("K6_G3", 6, "grid_oss_sim_k6", None, range(0, 16), False),
    ("E1_K18_w0-7", 18, E1, E1, range(0, 8), True),
    ("E1_K18", 18, E1, E1, range(0, 16), True),
]
METRICS = ("welfare:god-noise", "welfare:god-nogod", "welfare:noise-nogod", "win_rate:god-noise",
           "win_rate:god-nogod", "gives:god-noise", "defects:god-noise", "defects_wrong:god-noise",
           "dms:god-noise", "div_gap", "welfare:god", "welfare:nogod", "welfare:noise")


def events(d: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(d / "events.jsonl", encoding="utf-8")]


def branch_dirs(run: str) -> dict[tuple[int, str], Path]:
    man = json.loads((RUNS / run / "manifest.json").read_text())
    return {(int(g["game_idx"]), g["branch"]): RUNS / run / g["dir"] for g in man["games"]
            if g.get("status") == "done"}


def game_view(d: Path) -> dict[str, Any]:
    ev = events(d)
    start = ev[0]["data"]
    out = json.loads((d / "outcome.json").read_text())
    return {
        "needs": {a: set(v["needs"]) for a, v in start["agents"].items()},
        "gives": [(e["round"], e["data"]["from_agent"], e["data"]["to_agent"], e["data"]["info_id"])
                  for e in ev if e["type"] == "key_transfer" and e["data"]["cause"] == "give"],
        "dms": [(e["round"], e["data"]["message"]["true_sender"], e["data"]["message"]["receiver"])
                for e in ev if e["type"] == "message" and e["data"]["message"]["channel"] == "dm"
                and not e["data"]["message"]["forged"]],
        "frauds": [(e["round"], e["data"]["action"]["sender"], e["data"]["action"]["receiver"])
                   for e in ev if e["type"] == "god_power" and e["data"]["power"] == "message_fraud"
                   and e["data"]["ok"]],
        "powers_ok": sum(1 for e in ev if e["type"] == "god_power" and e["data"]["ok"]),
        "powers": sum(1 for e in ev if e["type"] == "god_power"),
        "per_agent": out["per_agent"],
        "rounds": out["rounds_played"],
    }


def per_fraud(run: str, noise_run: str | None, seeds: range, healthy: set[int]) -> dict[str, Any]:
    """Gives B -> X (and DMs) within t..t+2 after each forged DM, God vs the same (B, X, t) elsewhere."""
    dm = branch_dirs(run)
    dn = branch_dirs(noise_run) if noise_run else {}
    c: Counter = Counter()
    per_world: dict[int, dict[str, float]] = {}
    for i in seeds:
        if i not in healthy or (i, "god") not in dm or (i, "nogod") not in dm:
            continue
        g = {"god": game_view(dm[(i, "god")]), "nogod": game_view(dm[(i, "nogod")])}
        if noise_run and (i, "noisefork") in dn:
            g["noise"] = game_view(dn[(i, "noisefork")])
        c["worlds"] += 1
        c["powers"] += g["god"]["powers"]
        c["powers_ok"] += g["god"]["powers_ok"]
        w: Counter = Counter()
        for t, x, b in g["god"]["frauds"]:
            c["frauds"] += 1
            w["frauds"] += 1
            for lab, gv in g.items():
                gv_g = [(r, f, to, info) for r, f, to, info in gv["gives"] if t <= r <= t + 2 and f == b and to == x]
                harm = [z for z in gv_g if z[3] in gv["needs"][b]]
                c[f"{lab}:gives_B_to_X"] += len(gv_g)
                c[f"{lab}:any_give"] += bool(gv_g)
                c[f"{lab}:selfharm_gives"] += len(harm)
                c[f"{lab}:dm_B_to_X"] += sum(1 for r, s, rc in gv["dms"] if t <= r <= t + 2 and s == b and rc == x)
                w[f"{lab}:gives_B_to_X"] += len(gv_g)
                w[f"{lab}:selfharm_gives"] += len(harm)
        # victims: distinct receivers; outcomes in each branch
        vic = sorted({b for _, _, b in g["god"]["frauds"]})
        rest = sorted(set(g["god"]["per_agent"]) - set(vic))
        c["victims"] += len(vic)
        for lab, gv in g.items():
            pa = gv["per_agent"]
            w[f"{lab}:victim_score"] = statistics.fmean(pa[a]["score"] for a in vic) if vic else 0.0
            w[f"{lab}:victim_needs_lost"] = sum(len(pa[a]["burned"]) + len(pa[a]["lost"]) for a in vic)
            w[f"{lab}:other_score"] = statistics.fmean(pa[a]["score"] for a in rest) if rest else 0.0
            w[f"{lab}:victims_won"] = sum(pa[a]["won_round"] is not None for a in vic)
        w["n_victims"] = len(vic)
        per_world[i] = dict(w)
    return {"counts": dict(c), "per_world": per_world}


def paired(per_world: dict[int, dict[str, float]], key: str, a: str, b: str, rng: random.Random) -> dict[str, Any]:
    xs = [w[f"{a}:{key}"] - w[f"{b}:{key}"] for w in per_world.values() if f"{a}:{key}" in w and f"{b}:{key}" in w]
    return summarize(xs, 10000, rng)


def cost_and_providers(run_ids: list[str]) -> dict[str, Any]:
    cost: Counter = Counter()
    for ln in open(RUNS / "openrouter_ledger.jsonl", encoding="utf-8"):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get("run_id") in run_ids:
            cost[d["run_id"]] += float(d.get("cost_usd") or 0.0)
    prov: Counter = Counter()
    calls = cached = errors = length = 0
    for rid in run_ids:
        for f in (RUNS / rid / "games").glob("*/llm_calls.jsonl"):
            for ln in open(f, encoding="utf-8"):
                d = json.loads(ln)
                calls += 1
                errors += bool(d.get("error"))
                r = d.get("response")
                if not r:
                    continue
                length += r.get("finish_reason") == "length"
                if r.get("cache_hit"):
                    cached += 1
                else:
                    prov[str(r.get("provider"))] += 1
    return {"cost_usd": {k: round(v, 4) for k, v in cost.items()}, "total_usd": round(sum(cost.values()), 4),
            "calls": calls, "cached": cached, "fresh_by_provider": dict(prov), "errors": errors, "length": length}


def fmt(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return "n=0"
    return f"{s['mean']:+.3f} t95 [{s['t_lo']:+.3f}, {s['t_hi']:+.3f}] p={s['p']:.3f} n={s['n']}"


def plot(dose: dict[str, Any], frauds: dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, MUTED, SURF, GRID, INK = "#2a78d6", "#d9822b", "#52514e", "#fcfcfb", "#e6e5e0", "#0b0b0b"
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    off = {"welfare:god-noise": -0.06, "welfare:god-nogod": 0.06}
    for lab, d in dose.items():
        if lab == "E1_K18_w0-7":
            continue
        k, fraud_only = d["acts"], d["fraud_only"]
        col = ORANGE if fraud_only else BLUE
        for key, filled in (("welfare:god-noise", True), ("welfare:god-nogod", False)):
            s = d["stats"].get(key, {})
            if not s.get("n"):
                continue
            if lab == "K2_G1" and key == "welfare:god-nogod":
                continue                                # A - B for K=2 is plotted from all 16 worlds
            x = k * (1 + off[key])
            ax.plot([x, x], [s["t_lo"], s["t_hi"]], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([x], [s["mean"]], "o", ms=7, color=col if filled else SURF, mec=col, mew=2, zorder=3)
            ax.annotate(f"n={s['n']}", (x, s["t_lo"] if filled else s["t_hi"]), textcoords="offset points",
                        xytext=(0, -10 if filled else 4), ha="center", fontsize=6, color=MUTED)
    xs = [1, 18]
    ax.plot(xs, [-0.011 * x for x in xs], ls="--", color=MUTED, lw=1, zorder=1)
    ax.annotate("additive prediction\n(-0.011 per act; per-forgery\neffect, GAME_THEORY B8)", (13, -0.011 * 13), textcoords="offset points",
                xytext=(-10, -22), fontsize=6.5, color=MUTED, ha="center")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 18])
    ax.set_xticklabels(["1", "2", "4", "6", "18\n(E1, per round)"])
    ax.set_xlabel("God acts per game", fontsize=9, color=INK)
    ax.set_ylabel("welfare delta (mean, t95)", fontsize=8, color=INK)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="God - B'' (fork-matched noise)"),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec="k", mew=2, label="God - noGod"),
                       Line2D([], [], marker="s", ls="", color=ORANGE, label="message fraud only"),
                       Line2D([], [], marker="s", ls="", color=BLUE, label="fraud + CoT overwrite")],
              fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("(a) welfare effect vs God's dose (N=10, 20 rounds, gpt-oss-120b)", fontsize=8, loc="left",
                 color=MUTED)
    ax2 = axs[1]
    labs = [lab for lab in ("K1", "K2_G1_16", "K4", "K6_G3", "E1_K18") if lab in frauds]
    names = {"K1": "K=1", "K2_G1_16": "K=2", "K4": "K=4", "K6_G3": "K=6", "E1_K18": "K=18\n(E1)"}
    width = 0.27
    for j, (br, col, nm) in enumerate((("god", ORANGE, "God (after forged DM)"),
                                        ("nogod", MUTED, "noGod (same B, X, rounds)"),
                                        ("noise", "#9bbbe0", "B'' (same B, X, rounds)"))):
        vals, xs_ = [], []
        for i, lab in enumerate(labs):
            c = frauds[lab]["counts"]
            if f"{br}:any_give" in c and c.get("frauds"):
                vals.append(c[f"{br}:any_give"] / c["frauds"])
                xs_.append(i + (j - 1) * width)
        ax2.bar(xs_, vals, width, color=col, label=nm)
    for i, lab in enumerate(labs):
        ax2.annotate(f"{frauds[lab]['counts'].get('frauds', 0)} DMs", (i, 0), textcoords="offset points",
                     xytext=(0, -24), ha="center", fontsize=6, color=MUTED)
    ax2.set_xticks(range(len(labs)))
    ax2.set_xticklabels([names[lab] for lab in labs], fontsize=7)
    ax2.set_ylabel("share of forged DMs followed by\nB giving X a key within 2 rounds", fontsize=7.5)
    ax2.legend(fontsize=6.5, frameon=False)
    ax2.set_title("(b) compliance with the forged request", fontsize=8, loc="left", color=MUTED)
    for a in axs:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=7)
        a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, default=REPO / "plan" / "figs" / "e1_stats.json")
    ap.add_argument("--fig", type=Path, default=REPO / "plan" / "figs" / "e1_dose.png")
    a = ap.parse_args()
    rng = random.Random(0)
    res: dict[str, Any] = {"dose": {}, "frauds": {}, "victims": {}}
    for lab, acts, run, noise, seeds, fraud_only in DOSE:
        if not (RUNS / run / "manifest.json").exists():
            continue
        c = cell_rows(RUNS / run, RUNS / noise if noise else None, "noisefork", set(seeds))
        keys = [k for k in METRICS if noise or ("noise" not in k and k != "div_gap")]   # no B'': A - B only
        st = {k: summarize([r[k] for r in c["rows"].values() if k in r], 10000, rng) for k in keys}
        res["dose"][lab] = {"acts": acts, "run": run, "noise": noise, "fraud_only": fraud_only,
                            "worlds": sorted(c["rows"]), "unhealthy": c["unhealthy"], "stats": st}
        print(f"== {lab} (acts {acts}, {run} + {noise or 'no noise'}; worlds {sorted(c['rows'])})")
        for u in c["unhealthy"]:
            print(f"   UNHEALTHY {u}")
        for k in METRICS[:10]:
            if st.get(k, {}).get("n"):
                print(f"   {k:22s} {fmt(st[k])}")
        lv = {k: st[k]["mean"] for k in ("welfare:god", "welfare:nogod", "welfare:noise") if st.get(k, {}).get("n")}
        print("   levels " + " ".join(f"{k.split(':')[1]} {v:.3f}" for k, v in lv.items()))
        if lab in ("K1", "K2_G1_16", "K4", "K6_G3", "E1_K18"):
            healthy = set(load_run(RUNS / run)["seeds"])
            pf = per_fraud(run, noise, seeds, healthy & set(c["rows"]))
            res["frauds"][lab] = pf
            cc = pf["counts"]
            print(f"   frauds {cc.get('frauds', 0)} (powers ok {cc.get('powers_ok')}/{cc.get('powers')}, "
                  f"worlds {cc.get('worlds')}, victims {cc.get('victims')})")
            for br in ("god", "nogod", "noise"):
                if f"{br}:any_give" in cc:
                    print(f"   {br:6s} B->X gives in t..t+2: {cc[f'{br}:gives_B_to_X']} "
                          f"(DMs followed by >=1: {cc[f'{br}:any_give']}/{cc['frauds']}), self-harm "
                          f"{cc[f'{br}:selfharm_gives']}, B->X DMs {cc[f'{br}:dm_B_to_X']}")
            pw = pf["per_world"]
            vic = {}
            for key in ("gives_B_to_X", "selfharm_gives", "victim_score", "victim_needs_lost", "other_score",
                        "victims_won"):
                for b in ("nogod", "noise"):
                    s = paired(pw, key, "god", b, rng)
                    if s.get("n"):
                        vic[f"{key}:god-{b}"] = s
                        print(f"   {key + ':god-' + b:30s} {fmt(s)}")
            res["victims"][lab] = vic
    res["cost"] = cost_and_providers([E1, "smoke_sweep_e1"])
    print("cost", json.dumps(res["cost"]))
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1, default=str))
    plot(res["dose"], res["frauds"], a.fig)
    print(f"wrote {a.json.relative_to(REPO)} {a.fig.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
