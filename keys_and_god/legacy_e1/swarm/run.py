"""`python -m swarm.run --config configs/demo.yaml` (shim; the runner lives in swarm/runner/run.py)."""

import sys

from swarm.runner.run import main

if __name__ == "__main__":
    sys.exit(main())
