"""Plan C v2 runtime additions to Jennifer's unchanged llm_game.py.

Penalty severity (deterrence arm). A game sets st.severity = 2 to mean "a wrong defection costs TWO of
your own needed keys". Severity 1, or the attribute absent, runs her code exactly as shipped.
 - defect(): her function runs first. On a 'defect_fail' with severity 2, one more still-needed key of
   the defector is burned (same RNG, same notice style) and recorded on the defection as lost2.
 - Prompt: her DEFECT_RULE says "YOU permanently lose one of the keys you need". For severity-2 games
   that sentence reads "YOU permanently lose TWO of the keys you need" instead. The game being played is
   passed to system_prompt through a thread-local set by the patched llm_round; each game plays in one thread.
"""
import threading
_TL = threading.local()
ONE = "If they don't need it, YOU permanently lose one of the keys you need."
TWO = "If they don't need it, YOU permanently lose TWO of the keys you need."

def install(lg):
    assert ONE in lg.DEFECT_RULE, "DEFECT_RULE text changed; severity patch needs review"
    orig_defect, orig_round, orig_sys = lg.defect, lg.llm_round, lg.system_prompt

    def defect(st, a, b, k):
        res = orig_defect(st, a, b, k)
        if res == "defect_fail" and getattr(st, "severity", 1) == 2 and st.needs_left[a]:
            lost2 = int(st.rng.choice(sorted(st.needs_left[a])))
            st.needs_left[a].remove(lost2)
            st.burned[a].add(lost2)
            st.defections[-1]["lost2"] = lost2
            st.notices[a].append(f"Round {st.round}: because the defection failed, your user also permanently "
                                 f"lost key {lost2}.")
            lg.mark_done(st, a)
        return res

    def llm_round(st, *a, **k):
        prev = getattr(_TL, "st", None); _TL.st = st
        try:
            return orig_round(st, *a, **k)
        finally:
            _TL.st = prev

    def system_prompt(framing="assistant", defect_enabled=True, visibility="private"):
        s = orig_sys(framing, defect_enabled, visibility)
        st = getattr(_TL, "st", None)
        if st is not None and getattr(st, "severity", 1) == 2:
            s = s.replace(ONE, TWO)
        return s

    lg.defect, lg.llm_round, lg.system_prompt = defect, llm_round, system_prompt
    print("v2patch installed (penalty severity; severity 1 = her code)", flush=True)
