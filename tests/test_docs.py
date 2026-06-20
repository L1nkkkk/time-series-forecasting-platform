from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase7_design_docs_exist() -> None:
    expected_paths = [
        "docs/durable_queue_design.md",
        "docs/deployment_design.md",
        "docs/security_model.md",
        "docs/roadmap.md",
        "docs/adr/0002-local-job-runner-and-production-roadmap.md",
    ]

    for relative_path in expected_paths:
        assert (ROOT / relative_path).is_file(), relative_path


def test_readme_contains_production_hardening_roadmap() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Production Hardening Roadmap" in readme
