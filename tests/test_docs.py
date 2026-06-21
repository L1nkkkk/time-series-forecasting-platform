from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL_QUALITY_GATE_COMMANDS = [
    "python -m pytest",
    "ruff check .",
    "ruff format --check .",
    "mypy src",
    "python -m ts_platform.cli.main train --config configs/examples/simple_forecast.yaml",
    "python -m ts_platform.cli.main train --config configs/examples/csv_forecast.yaml",
    "python -m ts_platform.cli.main train --config configs/examples/csv_feature_forecast.yaml",
    "python -m ts_platform.cli.main list-datasets",
    "python -m ts_platform.cli.main list-datasets --catalog configs/datasets/local_csv.yaml",
    (
        "python -m ts_platform.cli.main profile-dataset "
        "--path tests/fixtures/tiny_series.csv "
        "--target-cols value "
        "--timestamp-col timestamp "
        "--input-len 8 "
        "--output-len 2"
    ),
    (
        "python -m ts_platform.cli.main profile-catalog "
        "--catalog configs/datasets/local_csv.yaml "
        "--input-len 8 "
        "--output-len 2"
    ),
    (
        "python -m ts_platform.cli.main make-config-from-catalog "
        "--catalog configs/datasets/local_csv.yaml "
        "--dataset tiny_csv "
        "--output /tmp/tiny_csv_generated.yaml "
        "--input-len 8 "
        "--output-len 2 "
        "--model linear "
        "--epochs 1"
    ),
    "python -m ts_platform.cli.main list-models",
    "python -m ts_platform.cli.main compare --config configs/examples/compare_forecast.yaml",
    "python -m ts_platform.cli.main compare --config configs/examples/compare_model_zoo.yaml",
    (
        "python -m ts_platform.cli.main compare "
        "--config configs/examples/compare_feature_forecast.yaml"
    ),
    (
        "python -m ts_platform.cli.main show-results "
        "--experiment compare_feature_forecast "
        "--run latest"
    ),
    (
        "python -m ts_platform.cli.main show-leaderboard "
        "--experiment compare_feature_forecast "
        "--run latest"
    ),
    (
        "python -m ts_platform.cli.main show-artifacts "
        "--experiment compare_feature_forecast "
        "--run latest"
    ),
    (
        "python -m ts_platform.cli.main show-artifact "
        "--experiment compare_feature_forecast "
        "--run latest "
        "--artifact leaderboard_json"
    ),
    "python -m ts_platform.cli.main list-jobs",
]


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


def test_phase14_final_architecture_docs_are_current() -> None:
    architecture = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    assert "will need to pass both dimensions" not in architecture
    assert "Future checkpoints should record" not in architecture
    assert "until concrete models become feature-aware" not in architecture
    assert "Checkpoint schema v2" in architecture
    assert "feature-aware" in architecture
    assert "target-only metrics" in architecture


def test_phase14_final_roadmap_points_to_final_freeze() -> None:
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")

    assert "Phase 14: CLI Modularization" in roadmap
    assert "Final Freeze" in roadmap or "final freeze" in roadmap


def test_phase14_final_quality_gate_docs_are_aligned() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    release_checklist = (ROOT / "docs/release_checklist.md").read_text(encoding="utf-8")

    for command in FINAL_QUALITY_GATE_COMMANDS:
        assert command in contributing
        assert command in release_checklist


def test_phase15a_dashboard_docs_are_linked() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    demo_guide = (ROOT / "docs/demo_guide.md").read_text(encoding="utf-8")
    dashboard_demo = (ROOT / "docs/dashboard_demo.md").read_text(encoding="utf-8")

    assert (ROOT / "docs/dashboard_demo.md").is_file()
    assert "docs/dashboard_demo.md" in readme
    assert "Dashboard Demo" in readme
    assert "Dashboard Demo" in demo_guide
    assert "http://127.0.0.1:8000/ui" in demo_guide
    assert "## Recommended Demo Flow" in dashboard_demo
    assert "## Timing Notes" in dashboard_demo
    assert "compare_feature_forecast" in dashboard_demo


def test_phase15a_dashboard_static_assets_are_packaged() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in pyproject
    assert "ts_platform = [" in pyproject
    assert '"api/static/*.html"' in pyproject
    assert '"api/static/*.js"' in pyproject
    assert '"api/static/*.css"' in pyproject
