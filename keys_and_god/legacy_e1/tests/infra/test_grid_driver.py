"""infra/grid_driver.py: projection weights and the kill target (CODE_REVIEW #24, #25). No processes killed."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

from swarm.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("grid_driver", REPO_ROOT / "infra" / "grid_driver.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def test_noise_branch_pays_the_shared_prefix():
    per = sum(10 * gd.call_weight(r) for r in range(1, 21))
    shared = sum(10 * gd.call_weight(r) for r in range(1, 3))
    two, three = gd.Cell("x", 1.0, 2, 4, 20), gd.Cell("x", 1.0, 3, 4, 20)
    assert abs(two.w_total - 4 * (2 * per - shared)) < 1e-6
    assert abs(three.w_total - 4 * (3 * per - shared)) < 1e-6      # B' is not a cache replay


def test_kill_target_group_only_for_own_group_leaders():
    leader = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])   # shares our group
    try:
        assert gd.kill_target(leader.pid) == ("group", leader.pid)
        assert gd.kill_target(child.pid) == ("pid", child.pid)
        assert os.getpgid(child.pid) == os.getpgid(0)
    finally:
        for p in (leader, child):
            p.terminate()
            p.wait()
