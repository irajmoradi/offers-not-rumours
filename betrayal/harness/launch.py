"""Usage: python -I launch.py KEYFILE RUN_DIR [SCRIPT [ARGS...]]
Puts the key from KEYFILE into OPENROUTER_API_KEY (never printed) and runs SCRIPT
(default RUN_DIR/run_notebook.py) with RUN_DIR as cwd and RUN_DIR + tools/ on sys.path."""
import sys, os, runpy
tools = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tools)
from orkey import load_key
keyfile, run_dir = sys.argv[1], os.path.abspath(sys.argv[2])
script = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(run_dir, "run_notebook.py")
os.environ["OPENROUTER_API_KEY"] = load_key(keyfile)
for v in ("OPENAI_API_KEY", "TOGETHER_API_KEY", "GROQ_API_KEY", "DEEPINFRA_API_KEY", "HF_TOKEN"):
    os.environ.pop(v, None)
os.chdir(run_dir)
sys.path.insert(0, run_dir)
sys.argv = [script] + sys.argv[4:]
runpy.run_path(script, run_name="__main__")
