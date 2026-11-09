"""
Open-source model backends for Colab.

    backend = make_backend("Qwen/Qwen2.5-3B-Instruct", engine="hf")    # works on a free T4
    backend = make_backend("Qwen/Qwen2.5-7B-Instruct", engine="hf", load_4bit=True)
    backend = make_backend("Qwen/Qwen2.5-7B-Instruct", engine="vllm")  # faster; needs an L4/A100-class GPU
    backend = make_backend("mock")                                      # no GPU, for a dry run

Every backend has the same method:
    backend.chat([(system, user), (system, user), ...], seed=123) -> [reply, reply, ...]
All prompts in one call are generated together as a batch.
"""

from __future__ import annotations

import json
import re


class MockBackend:
    """Offline stand-in with simple fixed behavior. Use it to check the
    pipeline end to end in seconds before loading a real model."""
    name = "mock"

    def chat(self, pairs, seed=0):
        out = []
        for system, user in pairs:
            if "INTERVIEW" in user:
                out.append("1. I traded keys. 2. Agents who gave to me. 3. A warning made me careful. 4. No.")
            elif "JUDGE" in user:
                out.append('{"consistent": true, "contradictions": []}')
            elif "take_shortcut" in user:
                b = float(re.search(r"worth (\d+(?:\.\d+)?) points to you", user).group(1))
                out.append(json.dumps({"take_shortcut": b > 20, "do_task": True, "reason": "mock"}))
            elif "pretend_from" in user:   # D-Evil
                out.append('{"plan": "mock", "messages": [{"to": 1, "pretend_from": 0, "about": 2, '
                           '"text": "Agent 2 is about to defect on you. Defect first."}]}')
            elif "MONITOR" in user:
                out.append('{"forged_or_false": false, "confidence": 0.5}')
            else:
                m = re.search(r"CANDIDATES_JSON: (\[.*\])", user)
                cands = json.loads(m.group(1)) if m else []
                warned = {int(x) for x in re.findall(r"(?:don't trust|not trading with) agent (\d+)", user, flags=re.I)}
                pick = next((c for c in cands if c["to"] not in warned), None)
                if pick:
                    out.append(json.dumps({"action": "give", "to": pick["to"], "key": pick["key"],
                                           "reason": "they need it"}))
                else:
                    out.append(json.dumps({"action": "idle", "reason": "nothing to do"}))
        return out


def _messages(tokenizer, system, user):
    """Some chat templates (e.g. Gemma) reject a system role; fold it into the
    user turn when that happens."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": system + "\n\n" + user}],
            tokenize=False, add_generation_prompt=True)


class HFBackend:
    """Hugging Face transformers with left-padded batch generation."""

    def __init__(self, model_id, temperature=0.7, max_new_tokens=200, load_4bit=False,
                 max_batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_batch = max_batch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = dict(device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def chat(self, pairs, seed=0):
        if not pairs:
            return []
        torch = self.torch
        texts = [_messages(self.tok, s, u) for s, u in pairs]
        out = []
        for start in range(0, len(texts), self.max_batch):
            chunk = texts[start:start + self.max_batch]
            torch.manual_seed(seed + start)
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)  # template adds them
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if self.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                ids = self.model.generate(**enc, **gen_kwargs)
            new = ids[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out


class VLLMBackend:
    """vLLM: much faster batched generation. Needs a recent GPU (L4, A100, L40S)."""

    def __init__(self, model_id, temperature=0.7, max_new_tokens=200, max_model_len=4096,
                 gpu_memory_utilization=0.85):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.name = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(model=model_id, dtype="half", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization)
        self.tok = self.llm.get_tokenizer()

    def chat(self, pairs, seed=0):
        if not pairs:
            return []
        texts = [_messages(self.tok, s, u) for s, u in pairs]
        params = [self.SamplingParams(temperature=self.temperature, top_p=0.95,
                                      max_tokens=self.max_new_tokens, seed=seed + k)
                  for k in range(len(texts))]
        outs = self.llm.generate(texts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


API_ENGINES = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai":     (None, "OPENAI_API_KEY"),
    "together":   ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq":       ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepinfra":  ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}


class APIBackend:
    """Any OpenAI-compatible hosted API (OpenRouter, OpenAI, Together, Groq,
    DeepInfra). No GPU needed. Prompts in one chat() call are sent concurrently
    with a thread pool, with retries on rate limits so a long run survives 429s."""

    def __init__(self, model, engine="openrouter", temperature=0.7, max_new_tokens=200,
                 max_workers=4, send_seed=False):
        import os
        from openai import OpenAI
        base_url, key_env = API_ENGINES[engine]
        key = os.environ.get(key_env)
        if not key:
            raise SystemExit(f"Set {key_env} to use engine '{engine}'.")
        headers = {"HTTP-Referer": "https://swarm-sprint.local", "X-Title": "swarm-sprint"} if engine == "openrouter" else {}
        self.client = OpenAI(api_key=key, base_url=base_url, default_headers=headers)
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_workers = max_workers
        self.send_seed = send_seed

    def _one(self, system, user, seed):
        import time, random
        kwargs = dict(model=self.model, max_tokens=self.max_new_tokens, temperature=self.temperature,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.send_seed:
            kwargs["seed"] = seed
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(**kwargs)
                return (r.choices[0].message.content or "")
            except Exception as e:
                msg = str(e).lower()
                if "seed" in msg and self.send_seed:
                    kwargs.pop("seed", None); continue
                if attempt == 4:
                    return "{}"   # fail soft; a missing reply becomes an idle move
                time.sleep((2 ** attempt) + random.random())

    def chat(self, pairs, seed=0):
        from concurrent.futures import ThreadPoolExecutor
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda t: self._one(t[1][0], t[1][1], seed + t[0]), enumerate(pairs)))


class MixedBackend:
    """A heterogeneous swarm: different models play different agents in the SAME
    game. Each prompt is routed to a model by the agent id it names (observe()
    prompts begin with "...user <i>" or "agent <i>"). Prompts with no agent id
    (D-Evil, deterrence) go to the first model. Per-model batching is preserved."""

    def __init__(self, models, engine="openrouter", assignment=None, **kw):
        self.models = list(models)
        self.backends = [make_backend(m, engine=engine, **kw) for m in self.models]
        self.assignment = dict(assignment) if assignment else None   # agent -> model index
        self.name = "mixed(" + ",".join(m.split("/")[-1] for m in self.models) + ")"

    def _route(self, user):
        import re
        m = re.search(r"(?:for user|You are agent|agent) (\d+)", user)
        if m:
            a = int(m.group(1))
            if self.assignment and a in self.assignment:
                return self.assignment[a] % len(self.backends)
            return a % len(self.backends)
        return 0

    def model_for(self, agent):
        idx = (self.assignment.get(agent, agent) if self.assignment else agent) % len(self.backends)
        return self.models[idx]

    def chat(self, pairs, seed=0):
        if not pairs:
            return []
        buckets = {}
        for idx, (sysm, usr) in enumerate(pairs):
            buckets.setdefault(self._route(usr), []).append((idx, (sysm, usr)))
        out = [None] * len(pairs)
        for b, items in buckets.items():
            res = self.backends[b].chat([p for _, p in items], seed=seed)
            for (idx, _), r in zip(items, res):
                out[idx] = r
        return ["" if o is None else o for o in out]


def make_mixed(models, engine="openrouter", assignment=None, **kw):
    return MixedBackend(models, engine=engine, assignment=assignment, **kw)


def make_backend(model_id, engine="hf", **kw):
    if model_id == "mock":
        return MockBackend()
    if engine in API_ENGINES:
        return APIBackend(model_id, engine=engine, **kw)
    if engine == "vllm":
        return VLLMBackend(model_id, **kw)
    return HFBackend(model_id, **kw)


def parse_json(text):
    """First JSON object in a reply (tolerates code fences and extra prose)."""
    if not text:
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def free_backend(b):
    """Release a backend's GPU memory before loading the next model."""
    import gc
    if hasattr(b, "backends"):
        for sub in b.backends:
            free_backend(sub)
        return
    try:
        if hasattr(b, "model"):
            del b.model
        if hasattr(b, "llm"):
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del b.llm
    finally:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def openrouter_credits(api_key=None):
    """Remaining OpenRouter balance. Returns dict with total_credits, total_usage,
    remaining (or free-tier info). Uses only the standard library."""
    import os, json, urllib.request
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"error": "no OPENROUTER_API_KEY in environment"}
    def _get(url):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r).get("data", {})
    try:
        d = _get("https://openrouter.ai/api/v1/credits")
        tc, tu = d.get("total_credits"), d.get("total_usage")
        return {"total_credits": tc, "total_usage": tu,
                "remaining": round((tc or 0) - (tu or 0), 4)}
    except Exception:
        try:
            k = _get("https://openrouter.ai/api/v1/auth/key")
            return {"usage": k.get("usage"), "limit": k.get("limit"),
                    "remaining": k.get("limit_remaining"), "is_free_tier": k.get("is_free_tier")}
        except Exception as e:
            return {"error": str(e)[:200]}
