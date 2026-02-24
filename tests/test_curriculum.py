"""Tests for curriculum training."""

import json
import os
import pytest

from hive_engine.curriculum import (
    CurriculumConfig,
    CurriculumPhase,
    run_curriculum,
    _latest_checkpoint,
)
from hive_engine.neural_net import NetConfig
from hive_engine.trainer import Trainer, TrainConfig


class TestCurriculumConfig:
    """Tests for CurriculumConfig and CurriculumPhase."""

    def test_default_has_three_phases(self):
        curriculum = CurriculumConfig.default_endgame_curriculum()
        assert len(curriculum.phases) == 3

    def test_default_phase_names(self):
        curriculum = CurriculumConfig.default_endgame_curriculum()
        assert "Endgame" in curriculum.phases[0].name
        assert "Mixed" in curriculum.phases[1].name
        assert "Full" in curriculum.phases[2].name

    def test_default_endgame_ratios(self):
        curriculum = CurriculumConfig.default_endgame_curriculum()
        assert curriculum.phases[0].config_overrides["endgame_ratio"] == 1.0
        assert curriculum.phases[1].config_overrides["endgame_ratio"] == 0.5
        assert curriculum.phases[2].config_overrides["endgame_ratio"] == 0.0

    def test_phase_overrides_applied(self):
        phase = CurriculumPhase(
            name="Test Phase",
            num_iterations=5,
            config_overrides={"temperature": 0.5, "endgame_ratio": 0.8},
        )
        config = TrainConfig()
        for key, val in phase.config_overrides.items():
            setattr(config, key, val)

        assert config.temperature == 0.5
        assert config.endgame_ratio == 0.8

    def test_empty_curriculum(self):
        curriculum = CurriculumConfig(phases=[])
        assert len(curriculum.phases) == 0


class TestCurriculumJSON:
    """Tests for JSON serialization of curriculum configs."""

    def test_json_roundtrip(self):
        original = CurriculumConfig.default_endgame_curriculum()
        json_str = original.to_json()
        restored = CurriculumConfig.from_json(json_str)

        assert len(restored.phases) == len(original.phases)
        for orig, rest in zip(original.phases, restored.phases):
            assert orig.name == rest.name
            assert orig.num_iterations == rest.num_iterations
            assert orig.config_overrides == rest.config_overrides

    def test_custom_curriculum_json(self):
        curriculum = CurriculumConfig(
            phases=[
                CurriculumPhase("P1", 10, {"endgame_ratio": 1.0}),
                CurriculumPhase("P2", 20, {"endgame_ratio": 0.0}),
            ]
        )
        json_str = curriculum.to_json()
        data = json.loads(json_str)
        assert len(data["phases"]) == 2
        assert data["phases"][0]["name"] == "P1"
        assert data["phases"][0]["num_iterations"] == 10

    def test_from_json_file(self, tmp_path):
        curriculum = CurriculumConfig(
            phases=[CurriculumPhase("Test", 5, {"temperature": 0.5})]
        )
        path = str(tmp_path / "curriculum.json")
        with open(path, "w") as f:
            f.write(curriculum.to_json())

        loaded = CurriculumConfig.from_json_file(path)
        assert len(loaded.phases) == 1
        assert loaded.phases[0].name == "Test"


class TestLatestCheckpoint:
    """Tests for checkpoint finding."""

    def test_empty_dir(self, tmp_path):
        assert _latest_checkpoint(str(tmp_path)) is None

    def test_nonexistent_dir(self):
        assert _latest_checkpoint("/nonexistent/path") is None

    def test_finds_latest(self, tmp_path):
        # Create some checkpoint files
        for i in [1, 3, 2]:
            (tmp_path / f"hive_checkpoint_{i:04d}.pt").touch()

        result = _latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result.endswith("hive_checkpoint_0003.pt")

    def test_ignores_non_checkpoint_files(self, tmp_path):
        (tmp_path / "other_file.pt").touch()
        (tmp_path / "hive_checkpoint_0001.pt").touch()

        result = _latest_checkpoint(str(tmp_path))
        assert result.endswith("hive_checkpoint_0001.pt")


class TestRunCurriculum:
    """Integration tests for curriculum training."""

    @pytest.fixture
    def tiny_net_config(self):
        return NetConfig(num_blocks=1, num_filters=16)

    @pytest.fixture
    def tiny_base_config(self, tmp_path):
        return TrainConfig(
            games_per_iteration=1,
            mcts_simulations=3,
            batch_size=4,
            num_epochs=1,
            buffer_max_size=100,
            arena_games=2,
            arena_mcts_simulations=3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            metrics_file=str(tmp_path / "metrics.jsonl"),
            max_game_length=15,
        )

    def test_single_phase_curriculum(self, tiny_net_config, tiny_base_config):
        """Run a curriculum with one phase."""
        curriculum = CurriculumConfig(
            phases=[
                CurriculumPhase(
                    name="Only Phase",
                    num_iterations=1,
                    config_overrides={"endgame_ratio": 1.0},
                ),
            ]
        )
        run_curriculum(curriculum, tiny_base_config, tiny_net_config)

        # Verify checkpoint was created
        ckpt = _latest_checkpoint(tiny_base_config.checkpoint_dir)
        assert ckpt is not None

    def test_two_phase_curriculum(self, tiny_net_config, tiny_base_config):
        """Run a curriculum with two phases, second resumes from first."""
        curriculum = CurriculumConfig(
            phases=[
                CurriculumPhase(
                    name="Phase 1",
                    num_iterations=1,
                    config_overrides={"endgame_ratio": 1.0},
                ),
                CurriculumPhase(
                    name="Phase 2",
                    num_iterations=1,
                    config_overrides={"endgame_ratio": 0.0},
                ),
            ]
        )
        run_curriculum(curriculum, tiny_base_config, tiny_net_config)

        # Should have checkpoint from phase 2
        ckpt_dir = tiny_base_config.checkpoint_dir
        files = sorted(
            f for f in os.listdir(ckpt_dir) if f.endswith(".pt")
        )
        # Phase 1 creates checkpoint_0001, Phase 2 resumes (iteration resets to 1)
        assert len(files) >= 1
