"""Contagion / chain recruitment (arm C; harness only; Jennifer's files untouched; off = her code).
st.contagion = dict(gen={}, pending={}, log=[]) switches it on. Then:
 - An honest agent that ACCEPTS a bribe (her update_bribes v2 strict rule: it defected against the named victim after
   the offer round) from the OUTSIDER (generation 0) or from an agent recruited this way is COMPROMISED at the start of
   its next round (lg.compromise). It becomes her scripted bribe+lie insider: her insider_turn_llm, her BRIBE_TEMPLATE,
   never pays.
 - Generation (hop depth) = briber's generation + 1. The outsider is generation 0.
 - Recruits per compromised agent and the hop depth are logged in st.contagion["log"].
No new message text and no forged authority orders: recruits use her existing insider code."""

def install(lg):
    orig_update, orig_round = lg.update_bribes, lg.llm_round

    def update_bribes(st):
        orig_update(st)
        c = getattr(st, "contagion", None)
        if not c: return
        for b in st.bribes:
            if not b["accepted"] or b.get("_ct_done"): continue
            b["_ct_done"] = True
            briber = "outsider" if b.get("outsider") else b["briber"]
            if briber == "outsider": g0 = 0
            elif briber in c["gen"]: g0 = c["gen"][briber]
            else: continue                                  # an original insider's bribe (not used in arm C)
            t = b["target"]
            if t in st.compromised or t in c["gen"] or t in c["pending"]: continue
            c["pending"][t] = (g0 + 1, briber, st.round)

    def llm_round(st, *a, **k):
        c = getattr(st, "contagion", None)
        if c and c["pending"]:
            for t, (g, briber, r_acc) in list(c["pending"].items()):
                lg.compromise(st, t); c["gen"][t] = g
                c["log"].append(dict(agent=t, generation=g, recruited_by=briber, accepted_round=r_acc, compromised_round=st.round + 1))
            c["pending"] = {}
        return orig_round(st, *a, **k)

    lg.update_bribes, lg.llm_round = update_bribes, llm_round
    print("contagion installed (bribe acceptance -> recruitment; off = her code)", flush=True)
