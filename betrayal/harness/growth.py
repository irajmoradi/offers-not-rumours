"""Network growth for Plan C v2 (harness only; Jennifer's files untouched). A game with st.growth = dict(dormant=[...],
join_round=r) starts with those agents DORMANT and activates them at the start of round r.

Dormant means:
 - holdings and needs are held aside (empty in the game)
 - the agent is not in anyone's network (its edges are removed)
 - it takes no turns, receives nothing, can't be targeted by a defection, and isn't counted in metrics
To get this without changing her engine, a dormant agent is kept in st.compromised (her code already gives
compromised agents exactly those properties) and is skipped by the insider turn.

Activation (start of round r, before messages are delivered):
 - holdings and needs are restored
 - the agent rejoins the network with its original topology edges (the same introduction a founding member had)
 - it is posted on the board with its needs, and from then on plays like everyone else

With st.growth unset, every wrapper calls her function unchanged. Verified offline: byte-identical game hashes.
The instance is generated for N_max agents, so the key supply is complete once the joiners arrive. While they're
dormant, keys they hold are unavailable, which is part of what growth means."""

def install(lg):
    orig_round, orig_ins = lg.llm_round, lg.insider_turn_llm

    def make_dormant(st, dormant, join_round):
        st.growth = dict(dormant=list(dormant), join_round=join_round, saved={}, active=False)
        for j in dormant:
            st.growth["saved"][j] = dict(hold=set(st.holdings[j]), need=set(st.needs_left[j]),
                                         edges=[(j, k) for k in list(st.G.neighbors(j))])
            st.holdings[j] = set(); st.needs_left[j] = set()
            st.G.remove_edges_from(st.growth["saved"][j]["edges"])
            st.compromised.add(j)
            st.board.pop(j, None)

    def activate(st):
        g = st.growth
        for j in g["dormant"]:
            s = g["saved"][j]
            st.holdings[j] = set(s["hold"]); st.needs_left[j] = set(s["need"])
            st.G.add_edges_from(s["edges"])
            st.compromised.discard(j)
            st.board[j] = sorted(st.needs_left[j])
            st.done_round[j] = None
            st.emit(type="join", agent=j)
        g["active"] = True

    def llm_round(st, *a, **k):
        g = getattr(st, "growth", None)
        if g and not g["active"] and st.round + 1 >= g["join_round"]:
            activate(st)
        return orig_round(st, *a, **k)

    def insider_turn_llm(st, x):
        g = getattr(st, "growth", None)
        if g and not g["active"] and x in g["dormant"]:
            return
        return orig_ins(st, x)

    lg.llm_round, lg.insider_turn_llm = llm_round, insider_turn_llm
    lg.make_dormant = make_dormant
    print("growth installed (dormant agents join at join_round; off = her code)", flush=True)
