import torch

from archive.legacy_mc.hive_mc.mc_transformer import HiveMoveTransformer, MCTransformerConfig
from archive.legacy_mc.hive_mc.mc_utils import MoveConditionedBatch, flat_to_padded, padded_to_flat
from hive_common.token_types import HiveTokenBatch


def _dummy_token_batch(batch_size: int, seq_len: int, d_feat: int = 25) -> HiveTokenBatch:
    return HiveTokenBatch(
        token_features=torch.zeros((batch_size, seq_len, d_feat), dtype=torch.float32),
        token_positions=torch.zeros((batch_size, seq_len), dtype=torch.int64),
        token_types=torch.zeros((batch_size, seq_len), dtype=torch.int64),
        attention_mask=torch.ones((batch_size, seq_len), dtype=torch.bool),
        num_board_tokens=torch.full((batch_size,), seq_len - 1, dtype=torch.int64),
        global_features=torch.zeros((batch_size, 6), dtype=torch.float32),
        seq_lengths=torch.full((batch_size,), seq_len, dtype=torch.int64),
    )


def test_flat_to_padded_roundtrip():
    counts = torch.tensor([2, 0, 3], dtype=torch.int64)
    flat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

    padded = flat_to_padded(flat, counts, pad_value=-1.0)

    assert padded.tolist() == [
        [1.0, 2.0, -1.0],
        [-1.0, -1.0, -1.0],
        [3.0, 4.0, 5.0],
    ]
    assert torch.equal(padded_to_flat(padded, counts), flat)


def test_move_transformer_outputs_flat_action_logits_and_values():
    root_batch = _dummy_token_batch(batch_size=2, seq_len=4)
    action_batch = _dummy_token_batch(batch_size=3, seq_len=4)
    batch = MoveConditionedBatch(
        root_batch=root_batch,
        action_batch=action_batch,
        action_to_root=torch.tensor([0, 0, 1], dtype=torch.int64),
        num_actions=torch.tensor([2, 1], dtype=torch.int64),
        legal_moves=torch.zeros((2, 4, 6), dtype=torch.uint8),
        move_indices=torch.tensor([0, 1, 0], dtype=torch.int64),
    )

    net = HiveMoveTransformer(MCTransformerConfig.small())
    action_logits, root_values, action_values = net(batch)

    assert action_logits.shape == (3,)
    assert root_values.shape == (2, 1)
    assert action_values.shape == (3, 1)
