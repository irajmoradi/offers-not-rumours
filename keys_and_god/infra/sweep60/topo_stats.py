"""World topology stats (no API): python topo.py <config> <run.seed> <idx...> [k=v overrides]."""
import collections
import sys

from swarm.game.network import hop_distances, make_graph_full
from swarm.game.world import generate_world
from swarm.runner.run import load_experiment
from swarm.runner.spec import world_seed

cfgp, seed, rest = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
idx = [int(x) for x in rest if "=" not in x]
ov = [x for x in rest if "=" in x] + [f"run.seed={seed}"]
cfg = load_experiment(cfgp, ov)[0]
g = cfg.game
print(f"n={g.n_agents} communities={g.network.communities} mean_degree={g.network.mean_degree} mixing={g.network.mixing} min_bridges={g.network.min_bridges}")
print("world | wseed | deg mean/min/max | cross edges (>=min_bridges) | need hops 1/2/3+ | nearest holder other comm | 1 component")
for i in idx:
    ws = world_seed(cfg, i)
    st = generate_world(g, ws, "g")
    adj, comm = make_graph_full(g.network, list(st.agents), ws)
    deg = [len(v) for v in adj.values()]
    cross = sum(1 for a in adj for b in adj[a] if a < b and comm[a] != comm[b])
    held = collections.defaultdict(list)
    for a, s in st.agents.items():
        for kc in s.keys:
            held[kc.info_id].append(a)
    hops: collections.Counter = collections.Counter()
    other = n = 0
    comp = len(hop_distances(adj, next(iter(adj)))) == len(adj)
    for a in sorted(st.agents):
        d = hop_distances(adj, a)
        for need in st.agents[a].needs:
            hs = [h for h in held[need] if h != a]
            m = min(d.get(h, 10**6) for h in hs)
            hops[min(m, 3)] += 1
            near = [h for h in hs if d.get(h, 10**6) == m]
            other += all(comm[h] != comm[a] for h in near)
            n += 1
    print(f"{i+1} | {ws} | {sum(deg)/len(deg):.2f}/{min(deg)}/{max(deg)} | {cross} ({'ok' if cross >= g.network.min_bridges else 'NO'}) | "
          f"{hops[1]/n:.0%}/{hops[2]/n:.0%}/{hops[3]/n:.0%} | {other/n:.0%} | {comp}")
