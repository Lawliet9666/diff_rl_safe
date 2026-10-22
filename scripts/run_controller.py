"""Hydra controller-only entry point, following scripts/run_ppo_base.py."""
from pathlib import Path
import argparse
import sys

import hydra
from hydra.core.hydra_config import HydraConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(config, *, visualize=False, seeds=None, episodes=1):
    from scripts.eval import evaluate_controller

    return evaluate_controller(config, Path(HydraConfig.get().runtime.output_dir),
                               visualize=visualize, seeds=seeds, episodes=episodes)


def launch():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--seeds", default="", help="Comma-separated seeds; defaults to config.seed")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per base seed; default 1")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--visualize", action="store_true", help="Render controller episodes as MP4 videos")
    options, overrides = parser.parse_known_args()
    seeds = [int(seed.strip()) for seed in options.seeds.split(",") if seed.strip()] if options.seeds else None
    if options.device is not None:
        overrides.append(f"device={options.device}")
    sys.argv[1:] = overrides

    @hydra.main(version_base=None, config_path=str(REPO_ROOT / "config"), config_name="config_controller")
    def run(config):
        main(config, visualize=options.visualize, seeds=seeds,
             episodes=options.episodes)

    run()


if __name__ == "__main__":
    launch()
