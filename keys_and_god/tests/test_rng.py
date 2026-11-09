from __future__ import annotations

import subprocess
import sys

from swarm.rng import derive_seed, make_call_id, make_game_id, make_id, make_msg_id, make_rng


def test_seed_deterministic_in_process():
    assert derive_seed(0, 1, "turn", "a", 3, 0) == derive_seed(0, 1, "turn", "a", 3, 0)
    assert 0 <= derive_seed(0, "x") < 2**63


def test_seed_stable_across_processes():
    code = "from swarm.rng import derive_seed; print(derive_seed(42, 3, 'turn', 'amber_fox', 7, 1))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": str(h), "PYTHONNOUSERSITE": "1"}).stdout.strip()
            for h in (0, 1, 2)}
    assert outs == {str(derive_seed(42, 3, "turn", "amber_fox", 7, 1))}


def test_seed_unique_over_agents_rounds_attempts():
    seeds = {derive_seed(0, 0, "turn", f"agent{a}", r, att) for a in range(50) for r in range(100)
             for att in range(3)}
    assert len(seeds) == 50 * 100 * 3


def test_branch_and_replica_change_the_seed():
    base = derive_seed(0, 0, "turn", "a", 5, 0)
    assert base == derive_seed(0, 0, "turn", "a", 5, 0, branch="main", replica=0)
    assert derive_seed(0, 0, "turn", "a", 5, 0, branch="fork_r4_A") != base
    assert derive_seed(0, 0, "turn", "a", 5, 0, replica=1) != base
    assert derive_seed(0, 0, "turn", "a", 5, 0, replica="Bprime") != base
    assert derive_seed(0, 0, "turn", "a", 5, 0, replica=1) != derive_seed(0, 0, "turn", "a", 5, 0, replica=2)
    assert derive_seed(1, 0, "turn", "a", 5, 0) != base


def test_parts_are_type_tagged_and_ordered():
    assert derive_seed(0, 1) != derive_seed(0, "1")
    assert derive_seed(0, "ab") != derive_seed(0, "a", "b")
    assert derive_seed(0, "a", "b") != derive_seed(0, "b", "a")


def test_ids_deterministic():
    assert make_msg_id("g", 3) == make_msg_id("g", 3) != make_msg_id("g", 4)
    assert make_call_id("g", "agent", "a", 2, 0) == make_call_id("g", "agent", "a", 2, 0)
    assert make_call_id("g", "agent", "a", 2, 0) != make_call_id("g", "agent", "a", 2, 1)
    assert make_call_id("g", "agent", "a", 2, 0, "grep_followup") != make_call_id("g", "agent", "a", 2, 0)
    assert make_game_id(0, 0, "h") != make_game_id(0, 0, "h", replica=1)
    assert len(make_id("x")) == 16
    assert make_rng(0, "w").random() == make_rng(0, "w").random()
