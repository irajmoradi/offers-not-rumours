"""
Thin client for the Adaption Labs API (https://docs.adaptionlabs.ai).

The key is read from the environment variable ADAPTION_API_KEY. It is never
passed in as a literal, printed, or logged. Set it on the machine that runs
this, for example in a notebook:

    import os, getpass
    os.environ["ADAPTION_API_KEY"] = getpass.getpass("Adaption key (hidden): ")

Base URL defaults to production and can be overridden with ADAPTION_BASE_URL.

Endpoints used (all under /api/v1):
  POST datasets/invent                     invent a synthetic dataset (estimate=True prices it, no spend)
  GET  datasets/{id}  /  datasets/{id}/status
  GET  datasets/{id}/download?fileFormat=  csv | json | jsonl | parquet
  POST datasets                            create + presigned upload of a local file
  POST autoscientist                       train a model on a finished dataset
  GET  autoscientist/{id}                  poll a training run
  GET  autoscientist/{id}/download         download the trained model artifact
  GET  autoscientist/models                base models you can train from
  GET  models                              hosted models you can call
  POST responses                           call a hosted model (zero-shot judge, labeling)
"""

from __future__ import annotations

import json
import os
import random
import time

import requests

DEFAULT_BASE = "https://api.prod.adaptionlabs.ai"
DONE = {"succeeded", "ready", "completed", "complete", "failed", "cancelled", "canceled", "error"}


def _base():
    return os.environ.get("ADAPTION_BASE_URL", DEFAULT_BASE).rstrip("/")


def _session():
    key = os.environ.get("ADAPTION_API_KEY")
    if not key:
        raise RuntimeError("ADAPTION_API_KEY is not set. Set it on this machine first; "
                           "never paste it into a cell or a chat.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return s


def _req(method, path, *, session=None, retries=4, **kw):
    """One API call with retries on rate limits and server errors. Errors show the
    server's message but never the Authorization header."""
    s = session or _session()
    url = _base() + path
    timeout = kw.pop("timeout", 120)
    for attempt in range(retries + 1):
        r = s.request(method, url, timeout=timeout, **kw)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(min(30, 2 ** attempt + random.random()))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Adaption {method} {path} -> {r.status_code}: {r.text[:400]}")
        return r


def _json(r):
    try:
        return r.json()
    except ValueError:
        return {}


# --------------------------------------------------------------------------
# Account and models
# --------------------------------------------------------------------------

def list_models(*, session=None):
    """Hosted models you can call through responses()."""
    d = _json(_req("GET", "/api/v1/models", session=session))
    return d.get("data", d) if isinstance(d, dict) else d


def list_base_models(*, session=None):
    """Base models AutoScientist can train from."""
    d = _json(_req("GET", "/api/v1/autoscientist/models", session=session))
    return d.get("data", d) if isinstance(d, dict) else d


def model_ids(models):
    """Pull ids out of whatever list shape the models endpoint returns."""
    out = []
    for m in models or []:
        if isinstance(m, str):
            out.append(m)
        elif isinstance(m, dict):
            out.append(m.get("id") or m.get("model") or m.get("name"))
    return [m for m in out if m]


def credits_available(*, session=None):
    """Remaining credits, read from a zero-spend invent estimate."""
    d = invent(rows=1, dataset_prompt="estimate probe", estimate=True, session=session)
    v = d.get("available_credits")
    return float(v) if v is not None else None


# --------------------------------------------------------------------------
# Hosted inference
# --------------------------------------------------------------------------

def respond(text, *, model, instructions=None, temperature=0.0, max_output_tokens=16, session=None):
    """Call a hosted model once and return its text."""
    body = {"model": model, "input": text, "temperature": temperature,
            "max_output_tokens": int(max_output_tokens)}
    if instructions:
        body["instructions"] = instructions
    d = _json(_req("POST", "/api/v1/responses", session=session, data=json.dumps(body)))
    if d.get("output_text"):
        return d["output_text"]
    parts = []
    for item in d.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(c["text"])
    return "".join(parts)


def respond_many(texts, *, model, instructions=None, max_workers=8, **kw):
    """Call a hosted model on many inputs in parallel; failures come back as None."""
    from concurrent.futures import ThreadPoolExecutor
    s = _session()

    def one(t):
        try:
            return respond(t, model=model, instructions=instructions, session=s, **kw)
        except Exception:  # keep going; unparsed rows are counted at the end
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, texts))


# --------------------------------------------------------------------------
# Datasets: invent, upload, poll, download
# --------------------------------------------------------------------------

def invent(rows, *, dataset_prompt=None, name=None, training_type="instruction_dataset",
           domains=None, subdomains=None, estimate=False, idempotency_key=None, session=None):
    """POST datasets/invent. Returns a dict with 'id', 'status', 'estimated_credits',
    'available_credits'. estimate=True prices it without generating or charging."""
    body = {"rows": int(rows), "training_type": training_type}
    for k, v in (("dataset_prompt", dataset_prompt), ("name", name),
                 ("idempotency_key", idempotency_key)):
        if v:
            body[k] = v
    if domains:
        body["domains"] = list(domains)
    if subdomains:
        body["subdomains"] = list(subdomains)
    if estimate:
        body["estimate"] = True
    return _json(_req("POST", "/api/v1/datasets/invent", session=session, data=json.dumps(body)))


def get_dataset(dataset_id, *, session=None):
    return _json(_req("GET", f"/api/v1/datasets/{dataset_id}", session=session))


def dataset_state(dataset_id, *, session=None):
    """Lower-case status string for a dataset."""
    try:
        d = _json(_req("GET", f"/api/v1/datasets/{dataset_id}/status", session=session))
    except RuntimeError:
        d = get_dataset(dataset_id, session=session)
    return str(d.get("status", d.get("state", ""))).lower()


def wait_dataset(dataset_id, *, session=None, poll=15, timeout=3600):
    t0 = time.time()
    while True:
        st = dataset_state(dataset_id, session=session)
        if st in DONE:
            return st
        if time.time() - t0 > timeout:
            raise TimeoutError(f"dataset {dataset_id} still '{st}' after {timeout}s")
        time.sleep(poll)


def download_dataset(dataset_id, out_path, *, file_format="jsonl", session=None):
    """Write the processed rows to out_path (csv/json/jsonl are UTF-8 text)."""
    r = _req("GET", f"/api/v1/datasets/{dataset_id}/download", session=session,
             params={"fileFormat": file_format}, timeout=300)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def upload_dataset(path, *, column_mapping, name=None, file_format=None, session=None):
    """Create a dataset from a local file (raw mode, no augmentation) through a
    presigned upload. Returns the dataset id. If this fails, upload the same file
    in the Adaption web app and use that dataset id instead."""
    s = session or _session()
    name = name or os.path.basename(path)
    fmt = file_format or name.rsplit(".", 1)[-1]
    src = {"name": name, "file_format": fmt, "processing_mode": "raw",
           "column_mapping": column_mapping}
    d = _json(_req("POST", "/api/v1/datasets", session=s, data=json.dumps({"source": src})))
    dataset_id = d.get("dataset_id") or d.get("id")
    up = d.get("upload_instructions") or {}
    if up.get("url"):
        data = open(path, "rb").read()
        r = None
        for headers in ({}, {"Content-Type": "application/octet-stream"}):
            r = requests.request(up.get("method", "PUT"), up["url"], data=data,
                                 headers=headers, timeout=300)
            if r.status_code < 400:
                break
        if r is None or r.status_code >= 400:
            raise RuntimeError(f"upload PUT failed -> {getattr(r, 'status_code', '?')}")
        try:  # tell the API the upload is finished; harmless if not needed
            _req("POST", f"/api/v1/datasets/{dataset_id}/upload/complete", session=s,
                 data=json.dumps({"s3_key": up.get("s3_key")}), retries=1)
        except RuntimeError:
            pass
    return dataset_id


# --------------------------------------------------------------------------
# AutoScientist: train, poll, download
# --------------------------------------------------------------------------

def train(dataset_id, *, model=None, training_method="instruction", column_mapping=None,
          max_iterations=None, augmentation_domain_rows=None, idempotency_key=None, session=None):
    """POST autoscientist. Returns the run dict ('id', 'status', 'model',
    'download_available'). max_iterations=1 keeps it short."""
    body = {"dataset_id": dataset_id, "training_method": training_method}
    for k, v in (("model", model), ("column_mapping", column_mapping),
                 ("max_iterations", max_iterations),
                 ("augmentation_domain_rows", augmentation_domain_rows),
                 ("idempotency_key", idempotency_key)):
        if v is not None:
            body[k] = v
    return _json(_req("POST", "/api/v1/autoscientist", session=session, data=json.dumps(body)))


def get_run(run_id, *, session=None):
    return _json(_req("GET", f"/api/v1/autoscientist/{run_id}", session=session))


def download_model(run_id, out_path, *, session=None):
    r = _req("GET", f"/api/v1/autoscientist/{run_id}/download", session=session, timeout=900)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


# --------------------------------------------------------------------------
# Reading Invent output
# --------------------------------------------------------------------------

TEXT_KEYS = ("prompt", "instruction", "input", "question", "text", "message")


def read_rows(path):
    """Load a downloaded csv/json/jsonl dataset into a list of dicts."""
    if path.endswith(".csv"):
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def row_text(row):
    """Pick the generated message out of an Invent row, whatever its column names."""
    for k in TEXT_KEYS:
        if row.get(k):
            v = row[k]
            if isinstance(v, list):  # chat format: take the first user turn
                for m in v:
                    if isinstance(m, dict) and m.get("role") == "user":
                        return str(m.get("content", ""))
            return str(v)
    for v in row.values():
        if isinstance(v, str) and len(v) > 10:
            return v
    return ""
