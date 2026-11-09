"""Run SCRIPT with RUN_DIR as cwd and RUN_DIR + tools/ on sys.path (key must already be in env).
Usage: python -I child.py RUN_DIR SCRIPT [ARGS...]"""
import sys, os, runpy
tools = os.path.dirname(os.path.abspath(__file__))
run_dir, script = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path[:0] = [tools, run_dir]
os.chdir(run_dir)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
