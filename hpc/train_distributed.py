"""Ray-powered distributed training entry point for the microgrid optimizer."""
import argparse
import json
from pathlib import Path
from typing import Optional

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from env.microgrid_env import MicrogridEnv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch distributed PPO training on a Ray (HPC) cluster."
    )
    parser.add_argument("--data-path", default="data/sample_data.csv",
                        help="CSV used by the microgrid environment.")
    parser.add_argument("--ray-address", default="",
                        help="Ray head address (use 'auto' inside Ray SLURM jobs). Leave blank to start a local Ray runtime.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of rollout workers to launch (maps to cluster CPUs).")
    parser.add_argument("--envs-per-worker", type=int, default=4,
                        help="Vector envs per worker to maximize throughput.")
    parser.add_argument("--num-gpus", type=float, default=0,
                        help="GPUs to dedicate to the trainer (fractional is ok).")
    parser.add_argument("--rollout-fragment-length", type=int, default=512,
                        help="Timesteps gathered per worker per iteration.")
    parser.add_argument("--train-batch-size", type=int, default=32768,
                        help="Total timesteps per SGD round (sum over workers).")
    parser.add_argument("--sgd-minibatch-size", type=int, default=4096,
                        help="Mini-batch size for PPO optimizer.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--num-iterations", type=int, default=50,
                        help="Upper bound on training iterations.")
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                        help="Save a checkpoint every N iterations.")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Directory for Ray checkpoints & metrics.")
    parser.add_argument("--target-reward", type=float, default=None,
                        help="Stop once the mean episode reward crosses this value.")
    parser.add_argument("--metrics-path", default="",
                        help="Optional JSONL file to append iteration metrics (for dashboards).")
    parser.add_argument("--local-cpus", type=int, default=None,
                        help="Override CPU count when starting an in-process Ray runtime.")
    parser.add_argument("--local-mode", action="store_true",
                        help="Enable Ray local_mode for debugging (disables parallelism).")
    parser.add_argument("--include-dashboard", action="store_true",
                        help="Expose the Ray dashboard when starting a local runtime.")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="Episode length in timesteps. Defaults to full dataset.")
    parser.add_argument("--grid-import-limit", type=float, default=None,
                        help="Optional cap on grid imports per step (kW).")
    parser.add_argument("--step-log-dir", default="",
                        help="Directory for per-step CSV logs (one file per worker). Disabled if empty.")
    parser.add_argument("--reward-weight-cost", type=float, default=1.0,
                        help="Weight applied to energy cost.")
    parser.add_argument("--reward-weight-carbon", type=float, default=0.1,
                        help="Weight applied to carbon intensity penalty.")
    parser.add_argument("--reward-weight-battery", type=float, default=1.0,
                        help="Weight applied to battery health penalty.")
    parser.add_argument("--reward-weight-unmet", type=float, default=100.0,
                        help="Weight applied to unmet demand penalty.")
    parser.add_argument("--reward-weight-export", type=float, default=0.2,
                        help="Credit multiplier for exporting energy back to grid.")
    return parser


def _register_env(args: argparse.Namespace) -> None:
    reward_weights = {
        "cost": args.reward_weight_cost,
        "carbon": args.reward_weight_carbon,
        "battery_penalty": args.reward_weight_battery,
        "unmet_demand": args.reward_weight_unmet,
        "export_credit": args.reward_weight_export,
    }

    def creator(env_config):
        data_path = env_config.get("data_path", args.data_path)
        log_path = None
        if args.step_log_dir:
            log_dir = Path(args.step_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            worker_index = getattr(env_config, "worker_index", env_config.get("worker_index", 0))
            vector_index = getattr(env_config, "vector_index", env_config.get("vector_index", 0))
            log_path = log_dir / f"steps_worker{worker_index}_env{vector_index}.csv"
        return MicrogridEnv(
            data_path=data_path,
            episode_length=args.episode_length,
            reward_weights=reward_weights,
            log_path=str(log_path) if log_path else None,
            grid_import_limit=args.grid_import_limit,
        )

    register_env("MicrogridEnv", creator)


def _maybe_write_metrics(metrics_path: str, payload: dict) -> None:
    if not metrics_path:
        return
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = _build_parser()
    args = args or parser.parse_args()

    _register_env(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(ignore_reinit_error=True, log_to_driver=True)
    if args.ray_address:
        init_kwargs["address"] = args.ray_address
    else:
        init_kwargs["include_dashboard"] = args.include_dashboard
        if args.local_cpus:
            init_kwargs["num_cpus"] = args.local_cpus
        if args.local_mode:
            init_kwargs["local_mode"] = True
    ray.init(**init_kwargs)

    config = (
        PPOConfig()
        .environment("MicrogridEnv", env_config={"data_path": args.data_path})
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
        )
    )

    algorithm = config.build()
    best_reward = float("-inf")

    try:
        for iteration in range(1, args.num_iterations + 1):
            result = algorithm.train()
            reward_mean = result.get("episode_reward_mean")
            reward_value = reward_mean if reward_mean is not None else 0.0
            timesteps_total = result.get("timesteps_total")
            print(
                f"[Iteration {iteration:03d}] reward_mean={reward_value:.2f} timesteps={timesteps_total}",
                flush=True,
            )

            payload = {
                "iteration": iteration,
                "reward_mean": reward_value,
                "timesteps_total": timesteps_total,
                "checkpoint_dir": None,
            }
            best_reward = max(best_reward, reward_value)

            if args.target_reward is not None and reward_value >= args.target_reward:
                print(
                    f"Target reward {args.target_reward} reached at iteration {iteration}. Stopping early.",
                    flush=True,
                )
                checkpoint_path = algorithm.save(args.output_dir)
                payload["checkpoint_dir"] = checkpoint_path
                _maybe_write_metrics(args.metrics_path, payload)
                break

            if iteration % args.checkpoint_freq == 0:
                checkpoint_path = algorithm.save(args.output_dir)
                print(f"Checkpoint saved to {checkpoint_path}", flush=True)
                payload["checkpoint_dir"] = checkpoint_path

            _maybe_write_metrics(args.metrics_path, payload)
    finally:
        algorithm.cleanup()
        ray.shutdown()
        print(f"Best mean reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
