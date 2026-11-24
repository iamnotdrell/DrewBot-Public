"""Independent Phase 5 rule, minimax, cache, interruption and neural integration checks."""

import hashlib
import math
import random
from dataclasses import replace

import chess
import numpy as np
import pytest
import torch

from drewbot.data.samples import auxiliary
from drewbot.data.walkthrough import encode_history, legal_mask, policy_target
from drewbot.models import DrewBot, ModelSettings, legal_policy
from drewbot.search.cache import LRU, MATE, Bound, TTEntry, load_score, store_score
from drewbot.search.config import SearchConfig, SearchLimits
from drewbot.search.engine import SearchEngine
from drewbot.search.evaluator import Evaluation, NeuralEvaluator, outcome_scalar
from drewbot.search.state import SearchState, zobrist
from drewbot.settings import SearchPlan, Settings


class SyntheticEvaluator:
    """Explicit synthetic deterministic fixture, not a chess model or strength baseline."""

    history_frames = 8

    def __init__(self, function=None, preferred=None):
        self.function = function or self.value
        self.preferred = preferred
        self.batches = []

    @staticmethod
    def value(board):
        number = int.from_bytes(hashlib.sha256(board.fen().encode()).digest()[:4], "big")
        return (number % 1801 - 900) / 1000

    def evaluate(self, positions):
        self.batches.append([p.board.fen() for p in positions])
        values = []
        for p in positions:
            moves = list(p.board.legal_moves)
            weights = {m: 20.0 if m.uci() == self.preferred else 1.0 for m in moves}
            total = sum(weights.values())
            values.append(
                Evaluation(self.function(p.board), {m: w / total for m, w in weights.items()})
            )
        return values


def limits(**changes):
    return replace(SearchLimits(depth=2, time_budget_seconds=30, node_budget=100000), **changes)


def minimax(board, depth, function, ply=0):
    """Independent exhaustive reference, without engine state/hash/TT/batching helpers."""
    if board.is_checkmate():
        return -MATE + ply
    if board.outcome() is not None:
        return 0.0
    claim = board.can_claim_draw()
    if depth == 0:
        value = function(board)
        return max(value, 0.0) if claim else value
    best = 0.0 if claim else -math.inf
    for move in board.legal_moves:
        board.push(move)
        try:
            best = max(best, -minimax(board, depth - 1, function, ply + 1))
        finally:
            board.pop()
    return best


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("batch,cache", [(1, 0), (4, 10000)])
def test_against_exhaustive_minimax(depth, batch, cache):
    board = chess.Board("8/8/7k/8/8/8/P6P/4K3 w - - 0 1")
    evaluator = SyntheticEvaluator()
    result = SearchEngine(evaluator).search(
        board,
        limits(
            depth=depth,
            inference_batch_size=batch,
            cache_entries=cache,
            transposition_entries=cache,
        ),
    )
    assert result.metrics.completed_depth == depth
    assert result.score == pytest.approx(minimax(board, depth, evaluator.function))
    assert result.move in board.legal_moves
    board.push(result.move)
    assert result.score == pytest.approx(-minimax(board, depth - 1, evaluator.function, 1))
    board.pop()
    for move in result.principal_variation:
        assert move in board.legal_moves
        board.push(move)


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_mover_perspective_not_absolute_color(turn):
    board = chess.Board()
    board.turn = turn
    result = SearchEngine(SyntheticEvaluator(lambda b: 0.73)).search(board, limits(depth=1))
    assert result.score == -0.73


@pytest.mark.parametrize(
    "fen,reason,score",
    [
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "CHECKMATE", -MATE),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "STALEMATE", 0),
        ("7k/8/6K1/8/8/8/8/8 b - - 0 1", "INSUFFICIENT_MATERIAL", 0),
        ("7k/8/6K1/8/8/8/R7/8 b - - 150 80", "SEVENTYFIVE_MOVES", 0),
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 150 80", "CHECKMATE", -MATE),
    ],
)
def test_automatic_terminals_without_inference(fen, reason, score):
    evaluator = SyntheticEvaluator()
    result = SearchEngine(evaluator).search(chess.Board(fen))
    assert result.terminal == reason and result.score == score
    assert result.move is None and not result.claim_draw
    assert result.metrics.nodes == result.metrics.inference_calls == 0
    assert not evaluator.batches


def repeated(cycles):
    board = chess.Board()
    for _ in range(cycles):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
    return board


def test_fivefold_is_automatic_but_threefold_is_optional():
    automatic = SearchEngine(SyntheticEvaluator()).search(repeated(4))
    assert automatic.terminal == "FIVEFOLD_REPETITION" and automatic.move is None
    board = repeated(2)
    claim = SearchEngine(SyntheticEvaluator(lambda b: 0.8)).search(board, limits(depth=1))
    assert claim.claim_draw and claim.score == 0 and claim.move is None
    winning = SearchEngine(SyntheticEvaluator(lambda b: -0.8)).search(board, limits(depth=1))
    assert not winning.claim_draw and winning.move in board.legal_moves and winning.score == 0.8


def test_intended_move_claim_and_fifty_move_option():
    board = repeated(1)
    for uci in ("g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)
    assert not board.is_repetition(3) and board.can_claim_threefold_repetition()
    for position in (board, chess.Board("7k/8/8/8/8/8/R7/4K3 w - - 99 80")):
        result = SearchEngine(SyntheticEvaluator(lambda b: 0.8)).search(position, limits(depth=1))
        assert result.claim_draw and result.score == 0
        assert result.claim_move in position.legal_moves
        witness = position.copy()
        witness.push(result.claim_move)
        assert witness.is_fifty_moves() or witness.is_repetition(3)


def test_mate_distance_prefers_shorter_win_and_longer_loss():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    result = SearchEngine(SyntheticEvaluator()).search(board, limits(depth=3))
    board.push(result.move)
    assert board.is_checkmate() and result.score == MATE - 1
    # Independent exhaustive reference verifies the negative forced-mate distance.
    losing = chess.Board("8/8/8/8/8/5k2/8/4q2K w - - 0 1")
    expected = minimax(losing, 4, lambda b: 0)
    result = SearchEngine(SyntheticEvaluator(lambda b: 0)).search(losing, limits(depth=4))
    assert result.score == expected
    for score in (MATE - 7, -MATE + 9, 0.25):
        assert load_score(store_score(score, 5), 5) == score
    assert load_score(store_score(MATE - 7, 5), 2) == MATE - 4
    assert load_score(store_score(-MATE + 7, 5), 2) == -MATE + 4


@pytest.mark.parametrize("kind", [Bound.EXACT, Bound.LOWER, Bound.UPPER])
def test_transposition_bounds_cutoff_only_when_valid(kind):
    board = chess.Board("8/8/7k/8/8/8/P6P/4K3 w - - 0 1")
    evaluator = SyntheticEvaluator()
    engine = SearchEngine(evaluator)
    engine.search(board, limits(depth=1))
    state = SearchState(board)
    exact = minimax(board, 2, evaluator.function)
    move = next(iter(board.legal_moves))
    engine.transpositions.put(state.key(), TTEntry(2, exact, kind, move))
    alpha, beta = (
        (-2, 2)
        if kind == Bound.EXACT
        else ((exact - 0.2, exact) if kind == Bound.LOWER else (exact, exact + 0.2))
    )
    before = engine.metrics.tt_cutoffs
    result = engine._node(state, 2, alpha, beta, 0)
    assert result.score == exact and engine.metrics.tt_cutoffs == before + 1
    # A bound that does not cross the requested window must not become an exact hit.
    if kind != Bound.EXACT:
        before = engine.metrics.tt_cutoffs
        result = engine._node(state, 2, -2, 2, 0)
        assert result.score == pytest.approx(exact)
        assert engine.metrics.tt_cutoffs == before
        assert engine.transpositions.get(state.key()).bound == Bound.EXACT
    # Even an exact shallower fact cannot answer a deeper query.
    engine.transpositions.put(state.key(), TTEntry(1, 0.999, Bound.EXACT, move))
    assert engine._node(state, 2, -2, 2, 0).score == pytest.approx(exact)


@pytest.mark.parametrize(
    "alpha,beta,bound", [(-2, 2, Bound.EXACT), (-2, -1, Bound.LOWER), (1, 2, Bound.UPPER)]
)
def test_stored_fail_soft_bounds(alpha, beta, bound):
    board = chess.Board("8/8/7k/8/8/8/P6P/4K3 w - - 0 1")
    engine = SearchEngine(SyntheticEvaluator())
    engine.search(board, limits(depth=1))
    state = SearchState(board)
    engine.transpositions = LRU(1000)
    line = engine._node(state, 2, alpha, beta, 0)
    entry = engine.transpositions.get(state.key())
    assert entry.bound == bound
    exact = minimax(board, 2, engine.evaluator.function)
    if bound == Bound.EXACT:
        assert line.score == exact
    elif bound == Bound.LOWER:
        assert beta <= line.score <= exact
    else:
        assert exact <= line.score <= alpha


@pytest.mark.parametrize(
    "fen,uci",
    [
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"),
        *[("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8" + p) for p in "qrbn"],
        *[("4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", "a2b1" + p) for p in "qrbn"],
    ],
)
def test_hash_make_unmake_special_moves(fen, uci):
    state = SearchState(chess.Board(fen))
    key = state.key()
    move = chess.Move.from_uci(uci)
    assert move in state.board.legal_moves
    state.push(move)
    assert state.hash == zobrist(state.board)
    state.pop()
    assert state.key() == key and state.hash == zobrist(state.board)


def test_random_hash_replay_and_full_unwind():
    state = SearchState(chess.Board())
    keys = []
    generator = random.Random(25)
    for _ in range(100):
        if state.board.is_game_over():
            break
        keys.append(state.key())
        state.push(generator.choice(list(state.board.legal_moves)))
        assert state.hash == zobrist(state.board)
        replayed = SearchState(state.board)
        assert replayed.key() == state.key()
    for key in reversed(keys):
        state.pop()
        assert state.key() == key and state.hash == zobrist(state.board)


def test_draw_and_model_history_are_part_of_cache_identity():
    known = repeated(2)
    unknown = chess.Board(known.fen())
    left, right = SearchState(known), SearchState(unknown)
    assert left.hash == right.hash and left.key() != right.key()
    # Equal current repetition count and pieces can still have different ancestor claims.
    a, b = chess.Board(), chess.Board()
    for uci in ("g1f3", "g8f6", "b1c3", "b8c6"):
        a.push_uci(uci)
    for uci in ("b1c3", "b8c6", "g1f3", "g8f6"):
        b.push_uci(uci)
    x, y = SearchState(a, 1), SearchState(b, 1)
    assert x.board.fen() == y.board.fen() and x.frames[-1] == y.frames[-1]
    assert x.key() != y.key()  # The reversible path can affect future draw claims.
    changed = known.copy()
    changed.halfmove_clock += 1
    with pytest.raises(ValueError, match="replay"):
        SearchState(changed)
    fen_changed = chess.Board(unknown.fen())
    fen_changed.halfmove_clock += 1
    assert SearchState(fen_changed).hash == right.hash
    assert SearchState(fen_changed).key() != right.key()
    fen_changed = chess.Board(unknown.fen())
    fen_changed.fullmove_number += 1
    assert SearchState(fen_changed).key() != right.key()
    # A forced Zobrist collision cannot alias different exact witnesses.
    right.hash = left.hash
    cache = LRU(2)
    cache.put(left.key(), 123)
    assert cache.get(right.key()) is None


def test_true_transposition_after_history_and_reversible_context_converge():
    a, b = chess.Board(), chess.Board()
    for uci in ("g1f3", "g8f6", "b1c3", "b8c6", "e2e4"):
        a.push_uci(uci)
    for uci in ("b1c3", "b8c6", "g1f3", "g8f6", "e2e4"):
        b.push_uci(uci)
    assert SearchState(a, 1).key() == SearchState(b, 1).key()
    assert SearchState(a, 8).key() != SearchState(b, 8).key()


def test_cache_limits_hits_and_disabled_equivalence():
    cache = LRU(2)
    for i in range(10):
        cache.put((i,), i)
    assert len(cache.entries) == 2 and cache.get((7,)) is None
    cache.get((8,))
    cache.put((10,), 10)
    assert cache.get((9,)) is None
    disabled = LRU(0)
    disabled.put((1,), 1)
    assert not disabled.entries
    board = chess.Board()
    enabled = SearchEngine(SyntheticEvaluator())
    result = enabled.search(board, limits(depth=2, cache_entries=25, transposition_entries=25))
    reference = SearchEngine(SyntheticEvaluator()).search(
        board, limits(depth=2, cache_entries=0, transposition_entries=0)
    )
    assert result.score == reference.score
    assert len(enabled.evaluations.entries) <= 25 and len(enabled.transpositions.entries) <= 25
    large = SearchEngine(SyntheticEvaluator()).search(board, limits(depth=2))
    assert large.metrics.evaluation_cache_hits > 0 and large.metrics.tt_hits > 0


def test_policy_ordering_and_legal_tie_break():
    evaluator = SyntheticEvaluator(lambda b: 0, preferred="e2e4")
    result = SearchEngine(evaluator).search(chess.Board(), limits(depth=1))
    assert result.move.uci() == "e2e4"
    first_child = chess.Board(evaluator.batches[1][0])
    expected = chess.Board()
    expected.push_uci("e2e4")
    assert first_child == expected
    tie = SearchEngine(SyntheticEvaluator(lambda b: 0)).search(chess.Board(), limits(depth=1))
    assert tie.move.uci() == "a2a3"


def test_interruption_preserves_last_completed_iteration_and_input():
    board = repeated(1)
    original_fen, original_moves = board.fen(), list(board.move_stack)
    one = SearchEngine(SyntheticEvaluator()).search(board, limits(depth=1))
    engine = SearchEngine(SyntheticEvaluator())
    interrupted = engine.search(board, limits(depth=4, node_budget=one.metrics.nodes + 3))
    assert interrupted.stop_reason == "node_budget"
    assert interrupted.metrics.nodes == one.metrics.nodes + 3
    assert interrupted.metrics.completed_depth == 1
    assert (interrupted.move, interrupted.score, interrupted.principal_variation) == (
        one.move,
        one.score,
        one.principal_variation,
    )
    assert board.fen() == original_fen and board.move_stack == original_moves
    for budget in (1, 2, 10):
        result = engine.search(board, limits(depth=3, node_budget=budget))
        assert result.metrics.nodes <= budget
        assert result.move in board.legal_moves and result.score is None
        assert result.metrics.completed_depth == 0


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_wall_budget_and_cancellation_are_cooperative():
    clock = FakeClock()
    evaluator = SyntheticEvaluator()
    real = evaluator.evaluate

    def slow(positions):
        result = real(positions)
        clock.now += 0.2
        return result

    evaluator.evaluate = slow
    engine = SearchEngine(evaluator, clock=clock)
    result = engine.search(chess.Board(), limits(time_budget_seconds=0.1))
    assert result.stop_reason == "time_budget" and result.score is None
    assert result.metrics.total_seconds == result.metrics.inference_seconds == 0.2
    assert result.metrics.inference_calls == 1
    cancelled = SearchEngine(SyntheticEvaluator()).search(chess.Board(), cancel=lambda: True)
    assert cancelled.stop_reason == "cancelled" and cancelled.metrics.nodes == 0
    assert cancelled.move in chess.Board().legal_moves


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_batched_leaves_preserve_results_and_bound_dispatch(batch):
    evaluator = SyntheticEvaluator()
    result = SearchEngine(evaluator).search(chess.Board(), limits(inference_batch_size=batch))
    baseline = SearchEngine(SyntheticEvaluator()).search(chess.Board(), limits())
    assert (result.score, result.move) == (baseline.score, baseline.move)
    assert max(map(len, evaluator.batches)) <= batch
    if batch > 1:
        assert result.metrics.inference_calls < baseline.metrics.inference_calls


@pytest.mark.parametrize(
    "changes",
    [
        {"depth": 0},
        {"depth": 65},
        {"depth": True},
        {"node_budget": None},
        {"node_budget": -1},
        {"inference_batch_size": 257},
        {"inference_batch_size": False},
        {"time_budget_seconds": float("nan")},
        {"time_budget_seconds": float("inf")},
        {"time_budget_seconds": 0},
        {"cache_entries": -1},
        {"cache_entries": True},
        {"transposition_entries": 1000001},
    ],
)
def test_limits_reject_invalid_capacities(changes):
    with pytest.raises(ValueError):
        SearchLimits(**changes)


def test_central_profiles_and_legacy_plan_are_active():
    config = Settings.load().search
    assert isinstance(config, SearchConfig) and SearchPlan is SearchConfig and config.implemented
    for mode in ("fast", "balanced", "deep"):
        assert isinstance(config.limits(mode), SearchLimits)
    with pytest.raises(ValueError):
        SearchPlan(implemented=False)
    with pytest.raises(ValueError):
        SearchConfig(fast={"typo": 5})
    with pytest.raises(ValueError):
        config.limits("unknown")


@pytest.mark.parametrize("bad", [chess.Board.empty(), chess.Board(chess960=True)])
def test_invalid_board_rejected(bad):
    with pytest.raises(ValueError):
        SearchEngine(SyntheticEvaluator()).search(bad)


def test_scalar_and_bad_evaluations_fail_explicitly():
    torch.testing.assert_close(
        outcome_scalar(torch.tensor([[0.1, 0.2, 0.7], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        torch.tensor([0.6, -1.0, 0.0]),
    )
    for bad in ([1, 1, 1], [-0.1, 0.1, 1.0], [float("nan"), 0, 1]):
        with pytest.raises(ValueError):
            outcome_scalar(torch.tensor(bad))
    with pytest.raises(ValueError, match="utility"):
        SearchEngine(SyntheticEvaluator(lambda b: float("nan"))).search(chess.Board())


@pytest.mark.parametrize("gab", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_actual_neural_adapter_search_and_unchanged_contract(gab, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Actual MPS unavailable")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        model = DrewBot(replace(ModelSettings(), gab_enabled=gab)).to(device)
        model.train()
        adapter = NeuralEvaluator(model)
        board = repeated(1)
        state = SearchState(board)
        position = state.snapshot()
        result = adapter.evaluate([position])[0]
        with torch.no_grad():
            output = model.forward_batch(
                {
                    "tokens": torch.tensor(
                        encode_history(list(position.frames))[None],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "auxiliary": torch.tensor(
                        auxiliary(board, position.repetitions)[None], device=device
                    ),
                }
            )
            mask = torch.tensor(legal_mask(board)[None], device=device)
            probabilities, _ = legal_policy(output.policy_logits, mask)
        assert model.training
        assert result.value == pytest.approx(
            float(outcome_scalar(output.value_logits.softmax(-1))[0]), abs=1e-6
        )
        for move, probability in result.policy.items():
            assert probability == pytest.approx(
                float(probabilities[0, policy_target(board, move)[0]]), abs=1e-6
            )
        batch = adapter.evaluate([position, position])
        assert batch[0].value == pytest.approx(result.value, abs=1e-5)
        for index in range(2):
            assert batch[index].policy == pytest.approx(result.policy, abs=1e-5)
        decision = SearchEngine(adapter).search(board, limits(depth=1, inference_batch_size=4))
        assert decision.metrics.completed_depth == 1 and decision.move in board.legal_moves
        assert model.training and decision.metrics.evaluated_positions > 1
        assert np.isfinite(decision.score)
    finally:
        torch.set_num_threads(previous)


def test_calibration_known_forecasts_and_confidence_bins():
    from drewbot.search.calibration import calibration_metrics

    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    perfect = calibration_metrics(torch.eye(3), labels)
    assert perfect["nll"] == perfect["multiclass_brier"] == perfect["confidence_ece"] == 0
    assert perfect["bins"][-1]["count"] == 3
    uniform = calibration_metrics(torch.full((3, 3), 1 / 3), labels)
    assert uniform["nll"] == pytest.approx(math.log(3))
    assert uniform["multiclass_brier"] == pytest.approx(2 / 3)
    assert uniform["confidence_ece"] == pytest.approx(0, abs=1e-7)
    wrong = calibration_metrics(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([0]))
    assert wrong["nll"] is None and wrong["zero_target_probability_count"] == 1
    assert wrong["multiclass_brier"] == 2 and wrong["confidence_ece"] == 1
    with pytest.raises(ValueError):
        calibration_metrics(torch.empty(0, 3), torch.empty(0, dtype=torch.long))


def test_rule_context_prevents_transposition_and_evaluation_cache_poisoning():
    known = repeated(2)
    fresh = chess.Board(known.fen())
    engine = SearchEngine(SyntheticEvaluator(lambda b: 0.8))
    engine.search(fresh, limits(depth=1))
    state = SearchState(known)
    # Same FEN/hash, but no claim rights in the cached board's unknown prefix.
    assert engine.transpositions.get(state.key()) is None
    assert engine.evaluations.get(state.key()) is None
    line = engine._node(state, 1, -2, 2, 0)
    assert line.claim and line.score == 0 and line.move is None
    assert engine.transpositions.get(state.key()).claim_draw


def test_claim_does_not_hide_a_forced_win():
    board = chess.Board("7k/8/6K1/8/8/8/R7/8 w - - 100 80")
    result = SearchEngine(SyntheticEvaluator(lambda b: 0.8)).search(board, limits(depth=1))
    assert result.score == MATE - 1 and not result.claim_draw
    board.push(result.move)
    assert board.is_checkmate()


def test_inference_error_restores_caller_board_and_model_mode():
    board = repeated(1)
    before = board.fen(), list(board.move_stack)
    evaluator = SyntheticEvaluator()
    original = evaluator.evaluate

    def broken(positions):
        if evaluator.batches:
            raise RuntimeError("synthetic inference failure")
        return original(positions)

    evaluator.evaluate = broken
    with pytest.raises(RuntimeError, match="synthetic inference"):
        SearchEngine(evaluator).search(board, limits(depth=2))
    assert (board.fen(), board.move_stack) == before


def test_invalid_replay_and_future_input_are_rejected():
    board = chess.Board()
    board.push(chess.Move.null())
    with pytest.raises(ValueError, match="replay"):
        SearchState(board)
    with pytest.raises(ValueError):
        SearchState(chess.Board(), 9)


def test_search_cli_terminal_and_configuration_round_trip(tmp_path, monkeypatch, capsys):
    from drewbot.search import cli

    monkeypatch.setattr(cli, "project_path", lambda name: tmp_path / name)
    result = cli.main(
        [
            "search",
            "--random-model",
            "--device",
            "cpu",
            "--output",
            "terminal",
            "--fen",
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            "--mode",
            "balanced",
        ]
    )
    import json

    saved = json.loads((tmp_path / "terminal/search.json").read_text())
    assert saved == json.loads(json.dumps(result)) and saved["terminal"] == "CHECKMATE"
    assert saved["limits"] == Settings.load().search.limits("balanced").__dict__
    with pytest.raises(SystemExit):
        cli.main(["search", "--random-model", "--output", "terminal"])
    capsys.readouterr()


def test_synthetic_terminal_oracle_prefers_delayed_forced_loss():
    class SyntheticMateEngine(SearchEngine):
        """Synthetic mate oracle on legal opening moves; not real chess mate claims."""

        @staticmethod
        def _terminal(board, ply):
            moves = board.move_stack
            target = 2 if moves and moves[0].uci() == "a2a3" else 4
            return (-MATE + ply, "SYNTHETIC_MATE") if len(moves) >= target else None

    engine = SyntheticMateEngine(SyntheticEvaluator(lambda b: 0))
    result = engine.search(chess.Board(), limits(depth=4))
    assert result.metrics.completed_depth == 4
    assert result.score == -MATE + 4
    assert result.move.uci() != "a2a3"  # That branch loses two plies earlier.


def test_keyboard_interruption_returns_completed_iteration():
    evaluator = SyntheticEvaluator()
    one = SearchEngine(evaluator).search(chess.Board(), limits(depth=1))
    calls_at_one = len(evaluator.batches)
    evaluator.batches.clear()
    original = evaluator.evaluate

    def interrupted(positions):
        if len(evaluator.batches) > calls_at_one:
            raise KeyboardInterrupt
        return original(positions)

    evaluator.evaluate = interrupted
    result = SearchEngine(evaluator).search(chess.Board(), limits(depth=4))
    assert result.stop_reason == "cancelled" and result.metrics.completed_depth == 1
    assert (result.move, result.score) == (one.move, one.score)
