"""Train the FNN value and move-ordering heads from alpha-beta records."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import torch

from hive_fnn.fnn_alphabeta_training import (
    AlphaBetaGenerationConfig,
    AlphaBetaLossConfig,
    AlphaBetaReplayBuffer,
    generate_alpha_beta_records,
    train_alpha_beta_batch,
)
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.fnn_alphabeta_tuning import load_search_config
from hive_fnn.fnn_native_alphabeta import AlphaBetaSearchConfig


def _load_network(path: str | None) -> HiveFNN:
    if path is None:
        return HiveFNN(FNNConfig.large()).cuda()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("net_config")
    if not isinstance(config, FNNConfig):
        config = FNNConfig.large()
    net = HiveFNN(config)
    state = (
        checkpoint.get("ema_state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("champion_state_dict")
    )
    if state is None:
        raise KeyError("checkpoint does not contain FNN weights")
    net.load_state_dict(state)
    return net.cuda()


def _model_state_cpu(net: HiveFNN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in net.state_dict().items()
    }


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_resume_state(
    path: Path,
    *,
    net: HiveFNN,
    ema: HiveFNN,
    optimizer: torch.optim.Optimizer,
    replay: AlphaBetaReplayBuffer,
    iteration: int,
    generation_config: AlphaBetaGenerationConfig,
    loss_config: AlphaBetaLossConfig,
) -> None:
    _atomic_torch_save(
        {
            "format": "fnn_alphabeta_training_state_v1",
            "model_state_dict": _model_state_cpu(net),
            "ema_state_dict": _model_state_cpu(ema),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_state_dict": replay.state_dict(),
            "net_config": net.config,
            "alpha_beta_generation_config": asdict(generation_config),
            "alpha_beta_loss_config": asdict(loss_config),
            "iteration": int(iteration),
            "rng_state": _capture_rng_state(),
        },
        path,
    )


def _load_resume_state(
    path: str | Path,
    *,
    net: HiveFNN,
    ema: HiveFNN,
    optimizer: torch.optim.Optimizer,
    replay: AlphaBetaReplayBuffer,
) -> tuple[int, dict | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "fnn_alphabeta_training_state_v1":
        raise ValueError(
            "resume file is not a full alpha-beta training state; "
            "use --checkpoint for a weights-only warm start"
        )
    net.load_state_dict(payload["model_state_dict"])
    ema.load_state_dict(payload["ema_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    replay.load_state_dict(payload["replay_state_dict"])
    _restore_rng_state(payload.get("rng_state"))
    return (
        int(payload["iteration"]) + 1,
        payload.get("alpha_beta_generation_config"),
    )


@torch.no_grad()
def _update_ema(ema: HiveFNN, net: HiveFNN, decay: float) -> None:
    source = net.state_dict()
    for name, target in ema.state_dict().items():
        if torch.is_floating_point(target):
            target.mul_(decay).add_(source[name], alpha=1.0 - decay)
        else:
            target.copy_(source[name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--resume",
        help="Full training state written as alpha_beta_training_state_latest.pt",
    )
    parser.add_argument("--iterations", type=int, default=1_500)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--game-nodes", type=int, default=1_000)
    parser.add_argument("--teacher-nodes", type=int, default=6_000)
    parser.add_argument("--relabel-fraction", type=float, default=0.25)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-game-length", type=int, default=300)
    parser.add_argument("--expansion-mask", type=int, default=0)
    parser.add_argument(
        "--search-config",
        help="Tuned SPSA JSON state or a direct AlphaBetaSearchConfig JSON",
    )
    parser.add_argument("--opening-diversity-plies", type=int, default=6)
    parser.add_argument("--opening-diversity-candidates", type=int, default=4)
    parser.add_argument("--opening-diversity-window", type=float, default=0.12)
    parser.add_argument(
        "--opening-diversity-temperature",
        type=float,
        default=0.0,
        help="0 samples uniformly; positive values use score softmax",
    )
    parser.add_argument("--buffer-size", type=int, default=75_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--search-value-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--exact-score-weight", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-dir", default="checkpoints_fnn_alphabeta")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint and args.resume:
        raise ValueError("--checkpoint and --resume are mutually exclusive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    net = _load_network(args.resume or args.checkpoint)
    ema = HiveFNN(net.config).cuda()
    ema.load_state_dict(net.state_dict())
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    replay = AlphaBetaReplayBuffer(args.buffer_size)
    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_config = AlphaBetaLossConfig(
        search_value_weight=args.search_value_weight,
        ranking_loss_weight=args.ranking_weight,
        exact_score_loss_weight=args.exact_score_weight,
    )
    start_iteration = 1
    resumed_generation_config = None
    if args.resume:
        start_iteration, resumed_generation_config = _load_resume_state(
            args.resume,
            net=net,
            ema=ema,
            optimizer=optimizer,
            replay=replay,
        )
        # Preserve moments while allowing intentional hyperparameter changes.
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
        print(
            f"resumed={args.resume} start_iteration={start_iteration} "
            f"replay={len(replay)}",
            flush=True,
        )
    if args.search_config:
        search_config = load_search_config(args.search_config)
    elif resumed_generation_config and resumed_generation_config.get("search_config"):
        search_config = AlphaBetaSearchConfig(
            **resumed_generation_config["search_config"],
        )
    else:
        search_config = AlphaBetaSearchConfig()

    for iteration in range(start_iteration, args.iterations + 1):
        generation_config = AlphaBetaGenerationConfig(
            games=args.games,
            game_node_budget=args.game_nodes,
            teacher_node_budget=args.teacher_nodes,
            teacher_relabel_fraction=args.relabel_fraction,
            max_depth=args.max_depth,
            max_game_length=args.max_game_length,
            expansion_mask=args.expansion_mask,
            opening_diversity_plies=args.opening_diversity_plies,
            opening_diversity_candidates=args.opening_diversity_candidates,
            opening_diversity_value_window=args.opening_diversity_window,
            opening_diversity_temperature=args.opening_diversity_temperature,
            search_config=search_config,
            seed=args.seed + iteration,
        )
        records, generation_stats = generate_alpha_beta_records(
            ema, generation_config,
        )
        replay.add(records)
        updates = (
            max(1, round(args.epochs * len(replay) / max(1, args.batch_size)))
            if len(replay)
            else 0
        )
        loss_sums: dict[str, float] = {}
        for update in range(updates):
            batch = replay.sample(
                args.batch_size,
                seed=args.seed + iteration * 1_000_000 + update,
            )
            metrics = train_alpha_beta_batch(
                net, optimizer, batch, loss_config,
            )
            _update_ema(ema, net, args.ema_decay)
            for name, value in metrics.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value
        averaged = {
            name: value / updates for name, value in loss_sums.items()
        }
        print(
            f"iteration={iteration} games={generation_stats['games']} "
            f"records={generation_stats['records']} replay={len(replay)} "
            f"truncated={generation_stats['truncated_games']} "
            f"depth_records={generation_stats['relabel_requested']} "
            f"diverse={generation_stats['opening_diverse_moves']} "
            f"loss={averaged.get('loss', 0.0):.5f} "
            f"value={averaged.get('value_loss', 0.0):.5f} "
            f"best={averaged.get('best_move_loss', 0.0):.5f} "
            f"ranking={averaged.get('ranking_loss', 0.0):.5f}",
            flush=True,
        )
        payload = {
            "model_state_dict": _model_state_cpu(net),
            "ema_state_dict": _model_state_cpu(ema),
            "net_config": net.config,
            "alpha_beta_generation_config": asdict(generation_config),
            "alpha_beta_loss_config": asdict(loss_config),
            "iteration": iteration,
        }
        torch.save(
            payload,
            output_dir / f"hive_fnn_alphabeta_{iteration:04d}.pt",
        )
        _save_resume_state(
            output_dir / "alpha_beta_training_state_latest.pt",
            net=net,
            ema=ema,
            optimizer=optimizer,
            replay=replay,
            iteration=iteration,
            generation_config=generation_config,
            loss_config=loss_config,
        )


if __name__ == "__main__":
    main()
