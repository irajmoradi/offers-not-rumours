"""Figures are valid PNGs and the report is a single static HTML file with every section."""

from __future__ import annotations

from pathlib import Path

from swarm.analysis.ripple import ripple
from swarm.runlog.trace import load_run
from swarm.viz.figures import network_png, ripple_png, timeline_png
from swarm.viz.report import build_report, main

PNG = b"\x89PNG\r\n\x1a\n"


def test_figures_are_png(fake_run: Path) -> None:
    br = load_run(fake_run).pairs()[0]
    for png in (timeline_png(br["god"]), network_png(br["god"]), ripple_png(ripple(br["god"], br["nogod"]))):
        assert png.startswith(PNG) and len(png) > 5000


def test_report_rebuild_to_custom_path(fake_run: Path, tmp_path: Path) -> None:
    out = build_report(fake_run, tmp_path / "r.html")
    html = out.read_text()
    assert "message_fraud" in html and "cot_control" in html
    assert html.count("data:image/png;base64,") >= 5      # 2 timelines + 2 networks + ripple
    assert "plotly" not in html.lower() and "<script" not in html.lower()


def test_report_with_noise_replica(synth_run: Path) -> None:
    html = (synth_run / "report.html").read_text()
    assert 'id="ripple"' in html


def test_cli_usage() -> None:
    assert main([]) == 2
