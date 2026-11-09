import llm_game as lg, v2patch, detmock
v2patch.install(lg)
bk = detmock.DetMock()
for sev in (1, 2):
    st = lg.new_game(8, "complete", 3, 12, 0.7); st.severity = sev
    lg.play(st, bk, seed=3)
    fails = [d for d in st.defections if not d["success"]]
    print("sev", sev, "fails", len(fails), "lost2", sum("lost2" in d for d in fails),
          "TWO in prompt", any("TWO of the keys" in c["system"] for c in st.calls))
for sev in (1, 2):
    st = lg.new_game(8, "complete", 5, 12, 0.7); st.severity = sev
    a, b = 0, 1
    k = next(k for k in range(st.cfg.n_ids) if k not in st.needs_left[b])
    n0 = len(st.needs_left[a]); r = lg.defect(st, a, b, k)
    print("sev", sev, r, "needs lost", n0 - len(st.needs_left[a]), "burned", sorted(st.burned[a]), st.notices[a][-1][:60])
