"""
Multi-phase curriculum training for Hive AlphaZero.

Automates a staged training strategy that gradually transitions from
endgame-only bootstrapping to full-game training. Each phase applies
config overrides (endgame_ratio, temperature, LR schedule, etc.) and
runs for a specified number of iterations before moving to the next phase.

Usage:
    from hive_engine.curriculum import CurriculumConfig, run_curriculum

    # Default 3-phase endgame curriculum
    curriculum = CurriculumConfig.default_endgame_curriculum()
    run_curriculum(curriculum, TrainConfig(), NetConfig.small())

    # Custom curriculum from JSON
    curriculum = CurriculumConfig.from_json("my_curriculum.json")
    run_curriculum(curriculum, TrainConfig(), NetConfig.small())
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

from hive_engine.neural_net import NetConfig
from hive_engine.trainer import Trainer, TrainConfig


@dataclass
class CurriculumPhase:
    """
    A single phase in curriculum training.

    Attributes:
        name: Human-readable phase name (printed during training).
        num_iterations: How many training iterations to run in this phase.
        config_overrides: Dictionary mapping TrainConfig field names to values.
            These override the base config for this phase only.
    """

    name: str
    num_iterations: int
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """
    Multi-phase curriculum training configuration.

    Defines a sequence of training phases, each with its own duration
    and config overrides. Phases run sequentially, with each phase
    resuming from the checkpoint of the previous phase.
    """

    phases: list[CurriculumPhase] = field(default_factory=list)

    @classmethod
    def default_endgame_curriculum(cls) -> CurriculumConfig:
        """
        Default 3-phase endgame bootstrap curriculum.

        Phase 1: Pure endgame positions with high exploration
        Phase 2: Mixed endgame/normal with standard exploration
        Phase 3: Full games with reduced exploration
        """
        return cls(
            phases=[
                CurriculumPhase(
                    name="Phase 1: Pure Endgame",
                    num_iterations=20,
                    config_overrides={
                        "endgame_ratio": 1.0,
                        "temperature": 1.25,
                        "lr_schedule": "cosine",
                        "lr_warmup_iterations": 3,
                    },
                ),
                CurriculumPhase(
                    name="Phase 2: Mixed",
                    num_iterations=20,
                    config_overrides={
                        "endgame_ratio": 0.5,
                        "temperature": 1.0,
                        "lr_schedule": "cosine",
                    },
                ),
                CurriculumPhase(
                    name="Phase 3: Full Games",
                    num_iterations=20,
                    config_overrides={
                        "endgame_ratio": 0.0,
                        "temperature": 0.8,
                        "lr_schedule": "cosine",
                    },
                ),
            ]
        )

    def to_json(self) -> str:
        """Serialize curriculum to JSON string."""
        data = {
            "phases": [
                {
                    "name": p.name,
                    "num_iterations": p.num_iterations,
                    "config_overrides": p.config_overrides,
                }
                for p in self.phases
            ]
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> CurriculumConfig:
        """Deserialize curriculum from JSON string."""
        data = json.loads(json_str)
        phases = [
            CurriculumPhase(
                name=p["name"],
                num_iterations=p["num_iterations"],
                config_overrides=p.get("config_overrides", {}),
            )
            for p in data["phases"]
        ]
        return cls(phases=phases)

    @classmethod
    def from_json_file(cls, path: str) -> CurriculumConfig:
        """Load curriculum from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


def _latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the most recent checkpoint file in a directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    files = sorted(
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("hive_checkpoint_") and f.endswith(".pt")
    )
    if not files:
        return None
    return os.path.join(checkpoint_dir, files[-1])


def run_curriculum(
    curriculum: CurriculumConfig,
    base_config: TrainConfig,
    net_config: NetConfig,
    resume_path: str | None = None,
) -> None:
    """
    Run multi-phase curriculum training.

    Each phase applies its config_overrides to a copy of base_config,
    creates or resumes a Trainer, runs for the phase's num_iterations,
    and saves a checkpoint. The next phase resumes from that checkpoint.

    Args:
        curriculum: The curriculum configuration with phases.
        base_config: Base training config (phases override specific fields).
        net_config: Network architecture config.
        resume_path: Optional checkpoint path to resume mid-curriculum.
    """
    checkpoint_path = resume_path

    for i, phase in enumerate(curriculum.phases):
        print(f"\n{'='*60}")
        print(f"CURRICULUM {phase.name} ({phase.num_iterations} iterations)")
        print(f"  Config overrides: {phase.config_overrides}")
        print(f"{'='*60}")

        # Build config for this phase
        phase_config = copy.copy(base_config)
        phase_config.num_iterations = phase.num_iterations
        for key, val in phase.config_overrides.items():
            if hasattr(phase_config, key):
                setattr(phase_config, key, val)
            else:
                print(f"  Warning: unknown config key '{key}', skipping")

        if checkpoint_path:
            print(f"  Resuming from: {checkpoint_path}")
            trainer = Trainer.from_checkpoint(
                checkpoint_path, config_overrides=phase_config
            )
        else:
            trainer = Trainer(config=phase_config, net_config=net_config)

        trainer.run()

        # Find latest checkpoint for next phase
        checkpoint_path = _latest_checkpoint(phase_config.checkpoint_dir)
        if checkpoint_path:
            print(f"  Phase complete. Checkpoint: {checkpoint_path}")
        else:
            print(f"  Phase complete. No checkpoint found.")
