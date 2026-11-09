"""Command line: python -m swarm.analysis.invariants runs/<run_id> [--base runs/<base>] ..."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from . import __doc__ as PKG_DOC
from .runs import check_run, control_replay_checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(PKG_DOC or "").split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--examples", type=int, default=5, help="violations printed per check")
    ap.add_argument("--no-prompts", action="store_true", help="skip the llm_calls prompt checks")
    ap.add_argument("--game", default=None, help="only this game label or game index prefix (e.g. g03)")
    ap.add_argument("--base", type=Path, default=None,
                    help="runner.fork.control_god cells: base run whose nogod games the controls replay")
    args = ap.parse_args(argv)
    vs = check_run(args.run, prompts=not args.no_prompts, only=args.game)
    if args.base is not None:
        rv, info = control_replay_checks(args.run, args.base)
        vs += rv
        for ln in info:
            print(f"  replay {ln}")
    by = collections.defaultdict(list)
    for v in vs:
        by[v.check].append(v)
    print(f"{args.run}: {len(vs)} violation(s)" + ("" if vs else " -- all invariants hold"))
    for chk, lst in sorted(by.items()):
        print(f"  [{chk}] {len(lst)}")
        for v in lst[: args.examples]:
            print(f"    {v.game}: {v.detail}")
    return 1 if vs else 0
