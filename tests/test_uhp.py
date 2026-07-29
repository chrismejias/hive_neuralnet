from hive_engine.game_state import GameState
from hive_engine.uhp import move_from_uhp, move_to_uhp


def test_uhp_round_trip_across_legal_moves() -> None:
    state = GameState()
    for chosen_index in (0, 2, 0, 3, 1, 0):
        legal = state.legal_moves()
        chosen = legal[chosen_index % len(legal)]
        encoded = move_to_uhp(state, chosen)
        assert move_from_uhp(state, encoded) == chosen
        state.apply_move(chosen)
        for move in state.legal_moves():
            assert move_from_uhp(state, move_to_uhp(state, move)) == move
