"""Run the basic training example."""

from __future__ import annotations

from ts_platform.runner.trainer import Trainer


def main() -> None:
    """Run the example config."""

    result = Trainer.from_config_path("configs/examples/simple_forecast.yaml").run()
    print(result.to_dict())


if __name__ == "__main__":
    main()
