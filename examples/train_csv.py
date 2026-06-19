"""Run the CSV training example."""

from __future__ import annotations

from ts_platform.runner.trainer import Trainer


def main() -> None:
    """Run the CSV example config."""

    result = Trainer.from_config_path("configs/examples/csv_forecast.yaml").run()
    print(result.to_dict())


if __name__ == "__main__":
    main()
