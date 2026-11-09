#!/bin/bash
# Test (e) for arm DC: flags-off identity hash with dual_control installed (no -I so PYTHONHASHSEED=0 applies), plus
# re-scoring existing non-dual arms with the current analysis code and comparing to the shipped results. NO network.
set -e
H=/home/iraj/projects/planC_v2/harness; PY=/home/iraj/projects/jwin_planc/.venv/bin/python
S=${1:?scratch dir}; mkdir -p $S/o $S/f $S/t $S/c16
cd $H; export PYTHONHASHSEED=0; unset OPENROUTER_API_KEY
$PY -s -P child.py ../pkg/swarm_llm identity_check.py orig 8 3 $S/o/x.json >/dev/null 2>&1 &
V2P=0 GROWTH_MOD=1 $PY -s -P child.py ../pkg/swarm_llm identity_check.py fast 8 3 $S/f/x.json >/dev/null 2>&1 &
wait
$PY -I -c "import json,sys;o=json.load(open('$S/o/x.json'))['sha256'];f=json.load(open('$S/f/x.json'))['sha256'];print('identity orig',o[:8],'fast+all-modules(flags off)',f[:8]);sys.exit(0 if o==f and o.startswith('f3a70ed6') else 1)"
cd /home/iraj/projects/planC_v2/out
$PY -I $H/analyze_t.py $S/t arm_T_paytiming >/dev/null 2>&1
ARM=C16 $PY -I $H/analyze_c.py $S/c16 arm_C16_contagion >/dev/null 2>&1
$PY -I -c "
import json,sys
a=json.load(open('$S/t/results_arm_T.json')); b=json.load(open('arm_T_v2_2/results_arm_T.json'))
c=json.load(open('$S/c16/results_arm_C.json')); d=json.load(open('arm_C16_v2_2/results_arm_C.json'))
ok=(a==b) and (c==d); print('re-score T identical:',a==b,' re-score C16 identical:',c==d); sys.exit(0 if ok else 1)"
$PY -I -m py_compile $H/run_v2.py $H/dual_control.py $H/disclosure.py $H/outsider_se.py && echo "py_compile ok"
echo "TEST (e) PASSED"
