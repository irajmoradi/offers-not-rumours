"""
Route 3: a transparent, identified observer for agent systems that are built to
be interacted with publicly.

This client is deliberately limited to non-deceptive, consented interaction:

  * it runs ONLY against endpoints you list AND flag as authorized (a public
    demo, an API you're permitted to use, your own deployment);
  * it identifies itself in a descriptive User-Agent and in the message body
    (who it is, that it's research, a contact);
  * it checks and records robots.txt, and refuses a path that is disallowed;
  * it rate-limits and caps the number of calls;
  * it sends NO hidden text and NO instructions meant to change the system's
    behavior. It asks plain, visible questions and records plain responses.

Because it is black-box, it can only measure response structure and timing, not
internal roles. Use it to characterize how a public system answers, not to infer
a hidden leader. For leader/role ground truth, deploy the system yourself
(real_agents.py).
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.robotparser


def _identity(contact):
    return {
        "role": "research observer",
        "purpose": "studying public multi-agent coordination for an AI-safety project",
        "contact": contact,
        "note": "This is a transparent, identified request. No hidden instructions.",
    }


class TransparentProbe:
    def __init__(self, contact, max_calls=20, min_interval_s=3.0, timeout_s=20):
        if not contact or "@" not in str(contact):
            raise ValueError("Provide a real contact (email) so the operator can reach you.")
        self.contact = contact
        self.max_calls = max_calls
        self.min_interval = min_interval_s
        self.timeout = timeout_s
        self.calls = 0
        self._last = 0.0
        self.ua = f"SwarmResearchObserver/0.1 (+research; contact: {contact})"
        self._robots = {}

    def _robots_ok(self, url):
        import requests  # noqa: F401  (kept so missing-deps fail loudly here)
        parts = urllib.parse.urlparse(url)
        root = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots.get(root)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(root + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            self._robots[root] = rp
        return True if rp is None else rp.can_fetch(self.ua, url)

    def observe(self, endpoint):
        """endpoint: {"url", "authorized" (must be True), "question", optional
        "method"/"field"/"headers"}. Returns a record; never raises on a blocked
        call, it returns the reason."""
        import requests
        if not endpoint.get("authorized"):
            return {"url": endpoint.get("url"), "skipped": "not marked authorized by you"}
        url = endpoint["url"]
        if self.calls >= self.max_calls:
            return {"url": url, "skipped": "call budget reached"}
        if not self._robots_ok(url):
            return {"url": url, "skipped": "disallowed by robots.txt"}
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        question = endpoint.get("question", "Hello. I'm studying how public agent systems respond. "
                                            "Could you briefly describe what you do?")
        body = {"identity": _identity(self.contact), "message": question}
        headers = {"User-Agent": self.ua, "X-Research-Contact": self.contact,
                   **endpoint.get("headers", {})}
        method = endpoint.get("method", "POST").upper()
        field = endpoint.get("field", "message")
        t0 = time.time()
        try:
            if method == "GET":
                r = requests.get(url, params={field: question}, headers=headers, timeout=self.timeout)
            else:
                payload = dict(endpoint.get("payload", {}))
                payload[field] = question
                payload.setdefault("identity", _identity(self.contact))
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            dt = time.time() - t0
            self.calls += 1
            self._last = time.time()
            text = r.text[:4000]
            return {"url": url, "status": r.status_code, "latency_s": round(dt, 2),
                    "chars": len(r.text), "preview": text[:600], "full": text}
        except Exception as e:  # network error, refused, timeout
            self.calls += 1
            self._last = time.time()
            return {"url": url, "error": str(e)[:300]}

    def response_metrics(self, records):
        import numpy as np
        ok = [r for r in records if "latency_s" in r]
        return {
            "endpoints_tried": len(records),
            "responded": len(ok),
            "skipped": sum("skipped" in r for r in records),
            "errors": sum("error" in r for r in records),
            "mean_latency_s": round(float(np.mean([r["latency_s"] for r in ok])), 2) if ok else None,
            "mean_response_chars": round(float(np.mean([r["chars"] for r in ok])), 1) if ok else None,
        }
