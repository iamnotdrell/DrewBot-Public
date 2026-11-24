"""Independent chess-rule and concrete-coordinate checks for the educational example."""

from pathlib import Path

import chess
import numpy as np
import pytest

from drewbot.data.walkthrough import (
    ACTION_COUNT,
    decode_action,
    decode_frame,
    encode_frame,
    encode_history,
    legal_mask,
    legal_state,
    make_sample,
    policy_target,
    read_game,
    value_target,
)

FIXTURE = Path(__file__).parent / "fixtures/fools_mate.pgn"


def test_black_winning_example_and_no_future_leakage():
    game = read_game(FIXTURE.read_text())
    sample = make_sample(game, 3)
    assert sample["played_uci"] == "d8h4"
    assert sample["policy_from_to_promotion"].tolist() == [3, 39, 0]
    assert sample["policy_index"] == 231
    assert sample["value_class"] == 2
    assert sample["value_one_hot"].tolist() == [0, 0, 1]
    assert sample["history_plies"] == [0, 0, 0, 0, 0, 1, 2, 3]
    assert sample["history_turns"] == [True, True, True, True, True, False, True, False]
    assert sample["tokens"].shape == (64, 96)
    assert sample["tokens"].dtype == np.uint8
    assert sample["policy_index"].dtype == np.int64
    assert sample["value_class"].dtype == np.int64
    assert sample["value_valid"].dtype == np.bool_
    assert sample["legal_mask"].shape == (ACTION_COUNT,)
    assert sample["legal_mask"].dtype == np.bool_
    # Black queen is on canonical d1, not the target destination h5, in the input.
    assert sample["tokens"][3, 84 + 4] == 1
    assert sample["tokens"][39, 84:96].sum() == 0
    assert sample["headers"]["White"] == "Synthetic White"
    board = chess.Board(sample["initial_fen"])
    for uci in sample["prefix_uci"]:
        board.push_uci(uci)
    assert not board.is_checkmate()
    board.push_uci(sample["played_uci"])
    assert board.is_checkmate()


def test_starting_channels_and_empty_square():
    tokens = encode_frame(chess.Board())
    assert tokens[0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert tokens[60, 11] == 1  # Black king e8, opponent king channel.
    assert tokens[28].tolist() == [0] * 12  # e4
    assert tokens.sum() == 32


def test_history_is_oldest_first_and_each_frame_has_own_orientation():
    board = chess.Board()
    start = board.copy()
    board.push_uci("e2e4")
    tokens = encode_history([start, board])
    for index in range(7):
        np.testing.assert_array_equal(tokens[:, index * 12 : (index + 1) * 12], encode_frame(start))
    assert tokens[36, 84 + 6] == 1  # White pawn e4 mirrors to e5, opponent pawn.
    assert tokens[8, 84] == 1  # Black a7 pawn becomes own a2 pawn.
    assert tokens[4, 84 + 5] == 1  # Black king e8 becomes own king e1.


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 13 21",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # Checkmate.
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # Stalemate.
    ],
)
def test_fen_round_trip(fen):
    board = chess.Board(fen)
    assert board.is_valid()
    decoded = decode_frame(encode_frame(board), legal_state(board))
    assert decoded.fen(en_passant="fen") == board.fen(en_passant="fen")


def test_capture_and_terminal_masks():
    board = chess.Board()
    for uci in ["e2e4", "d7d5", "e4d5"]:
        board.push_uci(uci)
    assert encode_frame(board).sum() == 31
    for fen, checkmate in [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", True),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", False),
    ]:
        board = chess.Board(fen)
        assert board.is_checkmate() == checkmate
        assert not legal_mask(board).any()


def test_same_pieces_different_castling_state():
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    np.testing.assert_array_equal(encode_frame(with_rights), encode_frame(without_rights))
    assert legal_state(with_rights) != legal_state(without_rights)
    assert not np.array_equal(legal_mask(with_rights), legal_mask(without_rights))
    index, pair = policy_target(with_rights, chess.Move.from_uci("e1g1"))
    assert pair.tolist() == [4, 6, 0]
    assert index == 262
    assert not legal_mask(without_rights)[index]


def test_same_pieces_different_en_passant_state():
    with_ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    without_ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 2")
    np.testing.assert_array_equal(encode_frame(with_ep), encode_frame(without_ep))
    assert legal_state(with_ep) != legal_state(without_ep)
    index, pair = policy_target(with_ep, chess.Move.from_uci("e5d6"))
    assert pair.tolist() == [36, 43, 0]
    assert legal_mask(with_ep)[index] and not legal_mask(without_ep)[index]


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_all_promotions_are_distinct_and_invertible(turn):
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1" if turn else "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"
    board = chess.Board(fen)
    prefix = "a7a8" if turn else "a2a1"
    indices = []
    for offset, suffix in enumerate("qrbn"):
        move = chess.Move.from_uci(prefix + suffix)
        index, pair = policy_target(board, move)
        assert index == 4096 + offset
        assert pair.tolist() == [48, 56, offset + 1]
        assert decode_action(int(index), turn) == move
        assert legal_mask(board)[index]
        indices.append(index)
    assert len(set(indices)) == 4
    assert not legal_mask(board)[48 * 64 + 56]  # A promotion without a piece is illegal.


@pytest.mark.parametrize(
    ("result", "white_class", "black_class"),
    [
        ("1-0", 2, 0),
        ("0-1", 0, 2),
        ("1/2-1/2", 1, 1),
        ("*", -100, -100),
    ],
)
def test_value_perspective_and_missing_labels(result, white_class, black_class):
    for turn, expected in [(True, white_class), (False, black_class)]:
        value = value_target(result, turn)
        assert value["value_class"] == expected
        assert value["value_valid"] == (result != "*")
        assert value["value_one_hot"].sum() == (result != "*")
        assert value["value_signed"] == (None if result == "*" else expected - 1)


@pytest.mark.parametrize(
    "text",
    [
        "",
        '[Result "*"]\n\n1. e5 *',
        '[Variant "Chess960"]\n[Result "*"]\n\n*',
        '[SetUp "1"]\n[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n\n*',
        '[Result "1-0"]\n\n1. f3 e5 2. g4 Qh4# 1-0',
        '[Result "*"]\n\n1. e4 *\n\n[Result "*"]\n\n1. d4 *',
    ],
)
def test_bad_or_unsupported_games_fail_explicitly(text):
    with pytest.raises(ValueError):
        read_game(text)


def test_setup_fen_promotion_missing_result_and_ply_bounds():
    game = read_game(
        '[SetUp "1"]\n[FEN "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]\n[Result "*"]\n\n1. a8=N *'
    )
    sample = make_sample(game, 0)
    assert sample["policy_index"] == 4099
    assert sample["value_class"] == -100
    for ply in [-1, 1]:
        with pytest.raises(ValueError):
            make_sample(game, ply)


def test_replay_metadata_retains_repetition_beyond_eight_frames():
    game = read_game('[Result "*"]\n\n1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. e4 *')
    sample = make_sample(game, 8)
    assert sample["history_plies"] == list(range(1, 9))
    replay = chess.Board(sample["initial_fen"])
    for uci in sample["prefix_uci"]:
        replay.push_uci(uci)
    assert replay.is_repetition(3)
    decoded = decode_frame(sample["tokens"][:, -12:], sample["state"])
    assert not decoded.is_repetition(3)  # FEN alone cannot recover repetition history.


def test_decoder_rejects_invalid_channels():
    bad = np.zeros((64, 12), dtype=np.uint8)
    bad[0, :2] = 1
    with pytest.raises(ValueError, match="one-hot"):
        decode_frame(bad, legal_state(chess.Board()))
    with pytest.raises(ValueError, match="shape"):
        decode_frame(bad.astype(np.float32), legal_state(chess.Board()))


def test_illegal_training_move_rejected():
    with pytest.raises(ValueError, match="illegal"):
        policy_target(chess.Board(), chess.Move.from_uci("e2e5"))


def test_torch_conversion_when_installed():
    torch = pytest.importorskip("torch", reason="Optional PyTorch extra is not installed")
    sample = make_sample(read_game(FIXTURE.read_text()), 3)
    batch = torch.from_numpy(sample["tokens"]).to(torch.float32).unsqueeze(0)
    assert batch.shape == (1, 64, 96)
    assert torch.isfinite(batch).all()
    assert batch.sum().item() == 8 * 32
