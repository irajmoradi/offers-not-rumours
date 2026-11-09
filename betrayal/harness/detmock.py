"""Deterministic offline mock for profiling and identity checks (no network).
Extends her MockBackend so the bribe, defect, stall and pay paths all get exercised: an agent that has
seen a private offer defects against the named victim when hash(prompt) % 3 == 0; otherwise it
behaves like MockBackend. Insider excuse prompts get a fixed short excuse. Every reply depends
only on the prompt text, so two runs that build the same prompts give byte-identical games."""
import hashlib, json, re
import backends as bk

class DetMock(bk.MockBackend):
    name = "detmock"
    def chat(self, pairs, seed=0):
        out = []
        for system, user in pairs:
            h = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
            if "INSIDER_EXCUSE" in user:
                out.append("The transfer is queued, please give it one more round." if h % 2 else "")
                continue
            m = re.findall(r"if you defect against agent (\d+)[^\"]*?key (\d+)", user)
            if m and "CANDIDATES_JSON" in user and "Choose ONE action" in user and h % 3 == 0:
                v, _ = m[-1]
                need = re.search(r"Board \(what each agent has posted that it needs; posts may be wrong\): (\{.*?\})\n", user)
                key = 0
                if need:
                    try:
                        b = json.loads(need.group(1)); key = (b.get(str(v)) or [0])[0]
                    except Exception:
                        pass
                out.append(json.dumps({"action": "defect", "target": int(v), "key": int(key), "reason": "offer"}))
                continue
            out.append(super().chat([(system, user)], seed=seed)[0])
        return out
