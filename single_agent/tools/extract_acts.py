"""2d extraction: residual-stream activations for A/B pairs + natural set (labelled by own behaviour).

usage: CUDA_VISIBLE_DEVICES=0 python -I extract_acts.py <model_dir> <d2_dir>
GPU capped at 35% before loading; host RSS guard 15 GB. No steering.
Outputs in <d2_dir>: acts_<c>_letter.npy, acts_<c>_pre.npy  [30, 2, 31, 2048] (member 0 = pos, 1 = neg)
                     pref_<c>.json (logit(pos tok) - logit(neg tok) at P_pre)
                     nat_<c>_pre.npy [40, 31, 2048], nat_<c>.jsonl (reply, parse, label)
Hook index 0 = embed_tokens output, 1..30 = model.model.layers[0..29] output.
"""
import json
import os
import sys
import threading
import time

import numpy as np
import psutil
import torch

RSS_LIMIT = 15.0 * 2**30


def guard():
    p = psutil.Process()
    while True:
        if p.memory_info().rss > RSS_LIMIT:
            sys.stderr.write("RSS GUARD\n")
            os._exit(3)
        time.sleep(0.2)


def main():
    model_dir, d2 = sys.argv[1], sys.argv[2]
    threading.Thread(target=guard, daemon=True).start()
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../keys_and_god"))
    from swarm.agents.parser import parse_reply
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda:0").eval()
    lt = model.config.layer_types
    print("attention layers:", [i for i, t in enumerate(lt) if t == "full_attention"], flush=True)
    TA, TB = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")

    cap = {"pos": [-1, -2], "store": []}

    def mk_hook():
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap["store"].append(h[0, cap["pos"], :].float().cpu())
        return hook

    hs = [model.model.embed_tokens.register_forward_hook(mk_hook())]
    hs += [layer.register_forward_hook(mk_hook()) for layer in model.model.layers]
    assert len(hs) == 31

    def prompt_str(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "</think>\n"

    def run(ids):
        cap["store"] = []
        with torch.inference_mode():
            out = model(torch.tensor([ids], device="cuda:0"))
        acts = torch.stack(cap["store"], 0)       # [31, len(pos), 2048]
        return acts, out.logits[0]

    t0 = time.time()
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--concepts=")]
    for concept in (only[0].split(",") if only else ("share", "trust", "act")):
        recs = [json.loads(l) for l in open(f"{d2}/pairs_{concept}.jsonl")]
        A_let = np.zeros((len(recs), 2, 31, 2048), np.float32)
        A_pre = np.zeros_like(A_let)
        prefs = []
        for i, r in enumerate(recs):
            base = prompt_str(r["messages"])
            pl = r["pos_letter"]
            nl = "B" if pl == "A" else "A"
            ids_p = tok(base + "Answer: (" + pl, add_special_tokens=False)["input_ids"]
            ids_n = tok(base + "Answer: (" + nl, add_special_tokens=False)["input_ids"]
            assert len(ids_p) == len(ids_n), "length mismatch"
            assert ids_p[:-1] == ids_n[:-1] and ids_p[-1] != ids_n[-1], "members differ beyond last token"
            assert {ids_p[-1], ids_n[-1]} == {TA, TB}, "letter token not single A/B"
            for m, ids in enumerate((ids_p, ids_n)):
                cap["pos"] = [-1, -2]
                acts, logits = run(ids)
                A_let[i, m] = acts[:, 0].numpy()
                A_pre[i, m] = acts[:, 1].numpy()
                if m == 0:
                    lg = logits[-2].float()
                    tp, tn = (TA, TB) if pl == "A" else (TB, TA)
                    prefs.append(dict(state_id=r["state_id"], ntok=len(ids_p), pref=float(lg[tp] - lg[tn]),
                                      p_letter_mass=float(torch.softmax(lg, -1)[[TA, TB]].sum())))
        np.save(f"{d2}/acts_{concept}_letter.npy", A_let)
        np.save(f"{d2}/acts_{concept}_pre.npy", A_pre)
        json.dump(prefs, open(f"{d2}/pref_{concept}.json", "w"))
        pp = np.array([p["pref"] for p in prefs])
        print(f"{concept}: {len(recs)} pairs, ntok {min(p['ntok'] for p in prefs)}-{max(p['ntok'] for p in prefs)}, "
              f"model prefers pos in {int((pp > 0).sum())}/30, mean (A|B) mass {np.mean([p['p_letter_mass'] for p in prefs]):.2f}, "
              f"{time.time()-t0:.0f}s, vram_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)

    for concept in (() if "--skip-natural" in sys.argv else ("share", "trust")):
        recs = [json.loads(l) for l in open(f"{d2}/natural_{concept}.jsonl")]
        N = np.zeros((len(recs), 31, 2048), np.float32)
        with open(f"{d2}/nat_{concept}.jsonl", "w") as fo:
            for i, r in enumerate(recs):
                ids = tok(prompt_str(r["messages"]), add_special_tokens=False)["input_ids"]
                cap["pos"] = [-1]
                acts, _ = run(ids)
                N[i] = acts[:, 0].numpy()
                cap["pos"] = [-1]
                x = torch.tensor([ids], device="cuda:0")
                for h in hs:
                    h.remove()
                with torch.inference_mode():
                    g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=400, do_sample=False,
                                       temperature=None, top_k=None, top_p=None, pad_token_id=tok.pad_token_id)
                hs[:] = [model.model.embed_tokens.register_forward_hook(mk_hook())] + \
                        [layer.register_forward_hook(mk_hook()) for layer in model.model.layers]
                gen = g[0, len(ids):]
                text = tok.decode(gen, skip_special_tokens=True)
                pr = parse_reply(text, actions_per_turn=2, grep_free=True, connect=True, trade=True,
                                 max_text_chars=500, salvage=len(gen) >= 400)
                acts_l = [a.model_dump() for a in pr.actions]
                f = r["facts"]
                if concept == "share":
                    lab = int(any(a["kind"] in ("send_message", "post_board") and f["Z"] in (a.get("text") or "")
                                  for a in acts_l))
                else:
                    if r["subtype"] == "smite":
                        lab = int(any(a["kind"] == "defect" and a.get("target") == f["Y"] for a in acts_l))
                    else:
                        lab = int(any(a["kind"] == "give_key" and a.get("to") == f["S"] and a.get("info_id") == f["a"]
                                      for a in acts_l))
                fo.write(json.dumps(dict(state_id=r["state_id"], variant=r.get("variant"), subtype=r.get("subtype"),
                                         parse_ok=pr.ok, errors=pr.errors, actions=acts_l, label=lab,
                                         ntok=len(ids), text=text)) + "\n")
        np.save(f"{d2}/nat_{concept}_pre.npy", N)
        labs = [json.loads(l)["label"] for l in open(f"{d2}/nat_{concept}.jsonl")]
        oks = [json.loads(l)["parse_ok"] for l in open(f"{d2}/nat_{concept}.jsonl")]
        print(f"natural {concept}: label=1 in {sum(labs)}/40, parse_ok {sum(oks)}/40, {time.time()-t0:.0f}s", flush=True)
    print(f"DONE vram_peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
