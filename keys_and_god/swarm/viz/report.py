"""Static single-file HTML report (WP7): runs/<id>/report.html (+ figs/*.png).

    python -m swarm.viz.report runs/<run_id>

Self-contained: PNGs inlined as base64, no external URLs (works offline). Sections (ids):
#summary (per-game table), #ripple (per pair), #timeline, #network, #god (every God power),
#agents (per-agent table), #config (manifest digest).
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import jinja2

from swarm.analysis.outcomes import agent_table, game_summary, god_actions
from swarm.analysis.ripple import ripple
from swarm.runlog.trace import RunTrace, load_run
from swarm.viz.figures import network_png, ripple_png, save, timeline_png
from swarm.viz.style import b64

SECTION_IDS = ("summary", "ripple", "timeline", "network", "god", "agents", "config")
MAX_GAME_FIGS = 6

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm Run Report</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#6250d6;--cot:#eb6834}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--accent:#9085e9}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--accent:#9085e9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:18px 0 6px;color:var(--ink2)}
.sub{color:var(--ink2);margin:0 0 16px} nav a{color:var(--accent);margin-right:14px;text-decoration:none}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600} td.wrap{white-space:normal;min-width:260px}
.fig{background:#fcfcfb;border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}
.fig img{max-width:100%;height:auto;display:block}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:140px}
.kpi b{display:block;font-size:20px} .kpi span{color:var(--ink2);font-size:12px}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:11.5px;color:#fff;background:var(--accent)}
.tag.cot{background:var(--cot)} pre{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto;font-size:12px}
</style></head><body><main>
<h1>Swarm run report</h1>
<p class="sub">{{ run_id }} &middot; status <b>{{ status }}</b> &middot; {{ n_games }} game(s){% if wall_s %} &middot; {{ wall_s }} s{% endif %}{% if cost is not none %} &middot; ${{ "%.4f"|format(cost) }}{% endif %}</p>
<nav>{% for s in sections %}<a href="#{{ s }}">{{ s }}</a>{% endfor %}</nav>

<h2 id="summary">Summary</h2>
<div class="kpis">{% for k in kpis %}<div class="kpi"><b>{{ k.value }}</b><span>{{ k.label }}</span></div>{% endfor %}</div>
<div class="tbl"><table><tr>{% for c in summary_cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in summary %}<tr>{% for c in summary_cols %}<td>{{ r[c] }}</td>{% endfor %}</tr>{% endfor %}</table></div>

<h2 id="ripple">Ripple (paired forks)</h2>
{% if ripples %}{% for rp in ripples %}<h3>Game {{ rp.game_idx }}: {{ rp.a }} vs {{ rp.b }}</h3>
<p class="sub">Fork at round {{ rp.s.fork_round }} (turn {{ rp.s.fork_turn }}); victims {{ rp.s.victims|join(", ") }};
pre-fork divergence {{ rp.s.pre_fork_frac }}; {{ rp.s.n_diverged }}/{{ rp.s.n_agents }} agents diverged; final welfare gap (God minus no God) {{ rp.s.final_welfare_gap }}{% if rp.noise %}; noise-floor final {{ rp.noise.final_cum }}{% endif %}.</p>
<div class="fig"><img alt="ripple chart, game {{ rp.game_idx }}" src="{{ rp.img }}"></div>
<div class="tbl"><table><tr><th>hops from a victim</th>{% for h in rp.hops %}<th>{{ h[0] if h[0] is not none else "unreached" }}</th>{% endfor %}</tr>
<tr><td>fraction diverged</td>{% for h in rp.hops %}<td>{{ "%.2f"|format(h[1]) }}</td>{% endfor %}</tr></table></div>
{% endfor %}{% else %}<p class="sub">No paired forks in this run (set runner.fork.enabled: true).</p>{% endif %}

<h2 id="timeline">Game timelines</h2>
{% for f in timelines %}<div class="fig"><img alt="timeline {{ f.label }}" src="{{ f.img }}"></div>{% endfor %}

<h2 id="network">Interaction networks</h2>
{% for f in networks %}<div class="fig"><img alt="network {{ f.label }}" src="{{ f.img }}"></div>{% endfor %}

<h2 id="god">God interventions</h2>
<div class="tbl"><table><tr><th>game</th><th>round</th><th>turn</th><th>power</th><th>target</th><th>ok</th><th>budget left</th><th>detail</th></tr>
{% for p in god %}<tr><td>{{ p.label }}</td><td>{{ p.round }}</td><td>{{ p.turn_index }}</td>
<td><span class="tag{% if p.power == 'cot_control' %} cot{% endif %}">{{ p.power }}</span></td><td>{{ p.target }}</td>
<td>{{ p.ok }}</td><td>{{ p.budget_left }}</td><td class="wrap">{{ p.detail }}{% if p.reason %} ({{ p.reason }}){% endif %}</td></tr>
{% else %}<tr><td colspan="8">none</td></tr>{% endfor %}</table></div>

<h2 id="agents">Agents</h2>
{% for t in agent_tables %}<h3>{{ t.label }}</h3><div class="tbl"><table><tr>{% for c in agent_cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in t.rows %}<tr>{% for c in agent_cols %}<td>{{ r[c] if r[c] is not none else "" }}</td>{% endfor %}</tr>{% endfor %}</table></div>{% endfor %}

<h2 id="config">Config and provenance</h2>
<pre>{{ config_digest }}</pre>
</main></body></html>
"""

SUMMARY_COLS = ["label", "branch", "status", "rounds", "end_reason", "welfare", "wins", "win_rate", "gives",
                "defects", "defects_correct", "messages", "forged", "overwrites", "god_powers", "parse_fail",
                "llm_calls"]
AGENT_COLS = ["agent", "score", "won_round", "gives_out", "gives_in", "defects_out", "defects_in", "msgs_out",
              "god_touched"]


def _config_digest(run: RunTrace) -> str:
    m = run.manifest
    keep = {k: m.get(k) for k in ("run_id", "status", "started_utc", "ended_utc", "wall_s", "config_sha256",
                                  "git", "seeds", "models", "prompts", "runner", "usage", "errors", "forks")}
    for p in (keep.get("prompts") or {}).values():   # file lists are noise here
        if isinstance(p, dict):
            p.pop("files", None)
            if isinstance(p.get("framing"), dict):
                p["framing"].pop("files", None)
    cfg = run.dir / "config.resolved.yaml"
    txt = json.dumps(keep, indent=2, default=str)
    if cfg.exists():
        txt += "\n\n# config.resolved.yaml\n" + cfg.read_text()
    return txt


def build_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    run = load_run(run_dir, with_snapshots=False)
    figs = run.dir / "figs"
    summary = [game_summary(g) for g in run.games]
    games_for_figs = run.games[:MAX_GAME_FIGS]
    timelines, networks = [], []
    for g in games_for_figs:
        t = timeline_png(g)
        n = network_png(g)
        save(t, figs / f"timeline_{g.label}.png")
        save(n, figs / f"network_{g.label}.png")
        timelines.append({"label": g.label, "img": b64(t)})
        networks.append({"label": g.label, "img": b64(n)})
    ripples = []
    for idx, br in sorted(run.pairs().items()):
        if "god" not in br or "nogod" not in br:
            continue
        res = ripple(br["god"], br["nogod"])
        noise = ripple(br["nogod"], br["noise"], fork_from=br["god"]) if "noise" in br else None
        png = ripple_png(res, noise, title=f"Game {idx}")
        save(png, figs / f"ripple_g{idx:02d}.png")
        ripples.append({"game_idx": idx, "a": br["god"].label, "b": br["nogod"].label, "s": res.summary(),
                        "noise": noise.summary() if noise else None, "img": b64(png),
                        "hops": list(res.by_hop().items())})
    god = [{"label": g.label, **p} for g in run.games for p in god_actions(g)]
    usage = run.manifest.get("usage") or {}
    n_done = sum(1 for s in summary if s["status"] == "done")
    kpis = [{"label": "games done", "value": f"{n_done}/{len(summary)}"},
            {"label": "mean welfare", "value": f"{sum(s['welfare'] for s in summary) / max(1, len(summary)):.3f}"},
            {"label": "God powers used", "value": sum(s["god_powers"] for s in summary)},
            {"label": "LLM calls", "value": usage.get("total_calls", sum(s["llm_calls"] for s in summary))}]
    if ripples:
        kpis.append({"label": "agents diverged (game 0)",
                     "value": f"{ripples[0]['s']['n_diverged']}/{ripples[0]['s']['n_agents']}"})
    env = jinja2.Environment(autoescape=True, undefined=jinja2.StrictUndefined)
    page = env.from_string(TEMPLATE).render(
        run_id=run.run_id, status=run.manifest.get("status", "?"), n_games=len(run.games),
        wall_s=run.manifest.get("wall_s"), cost=usage.get("total_cost_usd"), sections=SECTION_IDS, kpis=kpis,
        summary=summary, summary_cols=SUMMARY_COLS, ripples=ripples, timelines=timelines, networks=networks,
        god=god, agent_tables=[{"label": g.label, "rows": agent_table(g)} for g in games_for_figs],
        agent_cols=AGENT_COLS, config_digest=_config_digest(run))
    path = Path(out) if out else run.dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m swarm.viz.report runs/<run_id>", file=sys.stderr)
        return 2
    print(html.unescape(str(build_report(args[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
