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


def test_phase11_exogenous_feature_design_docs_exist() -> None:
    expected_paths = [
        "docs/exogenous_features_design.md",
        "docs/adr/0003-exogenous-feature-interface.md",
    ]

    for relative_path in expected_paths:
        assert (ROOT / relative_path).is_file(), relative_path


def test_phase11_exogenous_feature_links_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    data_format = (ROOT / "docs/data_format.md").read_text(encoding="utf-8")
    model_zoo = (ROOT / "docs/model_zoo.md").read_text(encoding="utf-8")

    assert "## Exogenous Features" in readme
    assert "## Exogenous Feature Columns" in data_format
    assert "## Exogenous Feature Support" in model_zoo


def test_phase13_release_docs_exist() -> None:
    expected_paths = [
        "docs/release_checklist.md",
        "docs/demo_guide.md",
        "docs/final_report_outline.md",
    ]

    for relative_path in expected_paths:
        assert (ROOT / relative_path).is_file(), relative_path


def test_phase13_readme_links_release_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/demo_guide.md" in readme
    assert "docs/release_checklist.md" in readme
    assert "docs/final_report_outline.md" in readme


def test_phase13_changelog_mentions_release_capabilities() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "Feature-aware training" in changelog
    assert "Feature-aware compare" in changelog
    assert "Checkpoint schema v2" in changelog
