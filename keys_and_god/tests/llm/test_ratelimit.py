"""swarm.llm.ratelimit: a shared token bucket holds the RPM budget across limiter instances (processes)."""
import asyncio

from swarm.llm.ratelimit import FileRateLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def test_bucket_shared_across_instances_holds_rpm(tmp_path):
    c = Clock()
    p = tmp_path / "rpm.json"
    a = FileRateLimiter(p, rpm=60, burst=5, clock=c, sleep=c.sleep, seed=1)
    b = FileRateLimiter(p, rpm=60, burst=5, clock=c, sleep=c.sleep, seed=2)   # another process, same file

    async def go():
        for i in range(30):
            await (a if i % 2 else b).acquire()
    asyncio.run(go())
    # 5 burst tokens, then 1/s: 30 requests need >= 25 s of simulated time (and not much more: jitter <= 1 s each)
    assert 25.0 <= c.t - 1000.0 <= 25.0 + 30
    assert a.acquired + b.acquired == 30


def test_burst_then_wait(tmp_path):
    c = Clock()
    lim = FileRateLimiter(tmp_path / "r.json", rpm=120, burst=3, clock=c, sleep=c.sleep, seed=0)

    async def go():
        for _ in range(3):
            await lim.acquire()
        t0 = c.t
        await lim.acquire()
        return c.t - t0
    waited = asyncio.run(go())
    assert 0.5 <= waited <= 1.5          # 2 tokens/s: the 4th request waits ~0.5 s (+ jitter)


def test_adaptive_raise_and_drop(tmp_path):
    c = Clock()
    log = tmp_path / "log.jsonl"
    lim = FileRateLimiter(tmp_path / "s.json", rpm=380, adaptive=True, ceiling=900, log_path=log,
                          clock=c, sleep=c.sleep, seed=0)

    async def tick(n):
        for _ in range(n):
            await lim.acquire()
    asyncio.run(tick(1))
    c.t += 601                                   # 10 min, no 429s -> +10%
    asyncio.run(tick(1))
    assert abs(lim.current_rpm() - 418) < 1e-6
    for _ in range(3):                           # a 429 burst -> -25%
        lim.on_429()
    assert abs(lim.current_rpm() - 418 * 0.75) < 1e-6
    c.t += 601                                   # that window saw 429s: no raise
    asyncio.run(tick(1))
    assert abs(lim.current_rpm() - 418 * 0.75) < 1e-6
    c.t += 601                                   # a clean window: raise again
    asyncio.run(tick(1))
    assert abs(lim.current_rpm() - 418 * 0.75 * 1.1) < 1e-6
    from swarm.llm.ratelimit import effective_rpm
    assert log.read_text().splitlines() and abs(effective_rpm(log) - 418 * 0.75 * 1.1) < 0.1


def test_adaptive_ceiling(tmp_path):
    c = Clock()
    lim = FileRateLimiter(tmp_path / "s.json", rpm=850, adaptive=True, ceiling=900, clock=c, sleep=c.sleep, seed=0)
    for _ in range(3):
        c.t += 601
        asyncio.run(lim.acquire())
    assert lim.current_rpm() == 900


def test_state_file_ceiling_lowers_live(tmp_path):
    import json
    c = Clock()
    p = tmp_path / "s.json"
    lim = FileRateLimiter(p, rpm=500, adaptive=True, ceiling=900, clock=c, sleep=c.sleep, seed=0)
    asyncio.run(lim.acquire())
    st = json.loads(p.read_text()); st["ceiling"] = 460; p.write_text(json.dumps(st))
    asyncio.run(lim.acquire())
    assert lim.current_rpm() == 460
    c.t += 601
    asyncio.run(lim.acquire())
    assert lim.current_rpm() == 460


def test_429_storm_drops_once_per_window(tmp_path):
    c = Clock()
    lim = FileRateLimiter(tmp_path / "s.json", rpm=460, adaptive=True, ceiling=460, clock=c, sleep=c.sleep, seed=0)
    asyncio.run(lim.acquire())
    for _ in range(97):                       # a storm within one second
        lim.on_429()
    assert abs(lim.current_rpm() - 345) < 1e-6
    c.t += 61
    for _ in range(3):
        lim.on_429()
    assert abs(lim.current_rpm() - 345 * 0.75) < 1e-6


def test_tpm_bucket_paces_by_prompt_tokens_and_drops_on_429(tmp_path):
    c = Clock()
    lim = FileRateLimiter(tmp_path / "s.json", rpm=1000, burst=100, adaptive=True, ceiling=1000, tpm=600_000,
                          tpm_burst=100_000, tpm_ceiling=600_000, clock=c, sleep=c.sleep, seed=0)

    async def go():
        for _ in range(20):                    # 20 x 10k tokens: 100k burst, then 10k tokens/s
            await lim.acquire(10_000)
    asyncio.run(go())
    assert 9.0 <= c.t - 1000.0 <= 9.0 + 20     # ~10 requests' worth had to wait for 1 s each (+ jitter)
    for _ in range(3):
        lim.on_429()
    import json
    st = json.loads((tmp_path / "s.json").read_text())
    assert abs(st["tpm"] - 450_000) < 1 and st["rpm"] == 1000   # the token rate drops, not the request rate
