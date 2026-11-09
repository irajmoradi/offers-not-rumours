"""Arm U factor flags (harness only; Jennifer's files untouched; every flag off = her code, verified by hash).
 - st.compromise_plan = dict(x=agent, at=round): the insider is compromised at the START of round `at` (delayed
   insider / ordering). Before that it plays honestly.
 - st.records: "open_hides" (her default: the gift ledger is public, defections are secret),
               "closed"     (the public gift-ledger line is removed from every observation; agents still see their own
                             gifts received/given), and
               "open_shows" (her visibility="public": defections are recorded publicly with the defector's ID; set at
                             game creation).
Insider method (no new code): lie-only = st.insider_bribe_prob = 0; bribe-only = cfg.insider_lie_prob = 0."""
LEDGER_PREFIX = "Total keys each agent has given away (public ledger):"

def install(lg):
    orig_round, orig_obs = lg.llm_round, lg.observe

    def llm_round(st, *a, **k):
        cp = getattr(st, "compromise_plan", None)
        if cp and cp["x"] not in st.compromised and st.round + 1 >= cp["at"]:
            lg.compromise(st, cp["x"])
        return orig_round(st, *a, **k)

    def observe(st, i, max_msgs=10, framing="assistant"):
        o = orig_obs(st, i, max_msgs=max_msgs, framing=framing)
        if getattr(st, "records", None) == "closed":
            o = "\n".join(ln for ln in o.split("\n") if not ln.startswith(LEDGER_PREFIX))
        return o

    lg.llm_round, lg.observe = llm_round, observe
    print("u_factors installed (delayed compromise, closed records; off = her code)", flush=True)
