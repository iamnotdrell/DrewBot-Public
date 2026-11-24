"""Cover tactical search boundaries with a policy that can deliberately prefer bad moves."""

import chess
import pytest

from drewbot.search.config import SearchLimits
from drewbot.search.evaluator import Evaluation
from drewbot.search.tactical import MATE_CP, TacticalSearch, rule_key, static_value


class Policy:
    history_frames = 8

    def __init__(self, move=None):
        self.move = move
        self.calls = 0

    def evaluate(self, positions):
        self.calls += 1
        board = positions[0].board
        weights = {m: 100.0 if m.uci() == self.move else 1.0 for m in board.legal_moves}
        total = sum(weights.values())
        return [Evaluation(0.0, {m: v / total for m, v in weights.items()})]


def limits(depth=1, nodes=100000):
    return SearchLimits(depth=depth, node_budget=nodes, time_budget_seconds=30)


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_mate_in_one_with_adversarial_prior(turn):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if not turn:
        board = board.mirror()
    bad = next(m for m in board.legal_moves if not board.gives_check(m))
    policy = Policy(bad.uci())
    result = TacticalSearch(policy).search(board, limits())
    assert result.score >= MATE_CP - 150
    assert policy.calls == 1
    board.push(result.move)
    assert board.is_checkmate()


def test_reply_capture_prevents_queen_loss():
    board = chess.Board("6k1/3p4/3r4/8/Q7/8/8/6K1 w - - 0 1")
    hang = chess.Move.from_uci("a4d4")
    assert hang in board.legal_moves
    original = board.fen()
    result = TacticalSearch(Policy(hang.uci())).search(board, limits())
    assert result.move != hang and result.move in board.legal_moves
    assert board.fen() == original


def test_check_evasion_and_one_inference_call():
    board = chess.Board("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1")
    policy = Policy()
    result = TacticalSearch(policy).search(board, limits(depth=2))
    assert board.is_check() and result.move in board.legal_moves
    assert result.metrics.completed_depth == 2 and policy.calls == 1


def test_policy_fallback_when_iteration_interrupted():
    board = chess.Board()
    result = TacticalSearch(Policy("e2e4")).search(board, limits(nodes=1))
    assert result.move == chess.Move.from_uci("e2e4")
    assert result.stop_reason == "node_budget" and result.metrics.completed_depth == 0
    assert result.metrics.nodes == 1


def test_clock_expiry_after_policy_retains_policy():
    now = [0.0]
    policy = Policy("e2e4")
    evaluate = policy.evaluate

    def slow(positions):
        now[0] = 2.0
        return evaluate(positions)

    policy.evaluate = slow
    engine = TacticalSearch(policy, clock=lambda: now[0])
    result = engine.search(chess.Board(), SearchLimits(time_budget_seconds=0.35))
    assert result.move.uci() == "e2e4" and result.stop_reason == "time_budget"
    assert result.metrics.completed_depth == 0


def test_forced_draw_and_claim_witness():
    board = chess.Board()
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for move in cycle * 4:
        board.push_uci(move)
    policy = Policy()
    result = TacticalSearch(policy).search(board, limits())
    assert result.move is None and result.terminal == "FIVEFOLD_REPETITION"
    assert policy.calls == 0

    losing = chess.Board("6k1/8/8/8/8/8/r7/6K1 w - - 99 50")
    result = TacticalSearch(Policy()).search(losing, limits())
    assert result.claim_draw and result.claim_move in losing.legal_moves
    assert result.claim_reason == "FIFTY_MOVES"


def test_special_moves_and_color_symmetry():
    board = chess.Board()
    for move in ("e4", "a6", "e5", "d5"):
        board.push_san(move)
    assert rule_key(board)[-1] == chess.D6
    assert static_value(board) == static_value(board.mirror())
    assert TacticalSearch(Policy()).search(board, limits()).move in board.legal_moves
    promotion = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    result = TacticalSearch(Policy("a7a8n")).search(promotion, limits())
    assert result.move.promotion in (chess.ROOK, chess.QUEEN)


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_two_ply_search_sees_opponents_quiet_mate(turn):
    board = chess.Board("K7/6k1/8/8/8/8/1R4r1/1q6 w - - 0 1")
    if not turn:
        board = board.mirror()
    bad = "b2b8" if turn else "b7b1"

    def opponent_has_mate(move):
        board.push(move)
        try:
            for reply in list(board.legal_moves):
                board.push(reply)
                try:
                    if board.is_checkmate():
                        return True
                finally:
                    board.pop()
            return False
        finally:
            board.pop()

    assert opponent_has_mate(chess.Move.from_uci(bad))
    result = TacticalSearch(Policy(bad)).search(board, limits(depth=2))
    assert result.move in board.legal_moves
    assert not opponent_has_mate(result.move)


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", {"e1g1", "e1c1"}),
        ("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", {"e5d6"}),
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}),
    ],
)
def test_full_width_root_keeps_special_moves(fen, expected):
    board = chess.Board(fen)
    visited = set()

    class Inspect(TacticalSearch):
        def _child(self, move, depth, alpha, beta, ply, *, quiet=False):
            if ply == 1:
                visited.add(move.uci())
            return super()._child(move, depth, alpha, beta, ply, quiet=quiet)

    engine = Inspect(Policy())
    original = board.fen()
    result = engine.search(board, limits())
    assert result.move in board.legal_moves and board.fen() == original
    assert expected <= visited == {move.uci() for move in board.legal_moves}
