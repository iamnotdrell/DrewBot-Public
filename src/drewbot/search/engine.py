"""Iterative-deepening fail-soft negamax with neural ordering and strict context reuse."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter

import chess

from drewbot.search.cache import LRU, MATE, Bound, TTEntry, load_score, store_score
from drewbot.search.config import SearchLimits
from drewbot.search.evaluator import Evaluation, Evaluator
from drewbot.search.state import PositionInput, SearchState


@dataclass
class SearchMetrics:
    """WHAT: Actual work counters. HOW: Count nodes/expansions/inference separately. WHY: Audit
    cost.
    """

    nodes: int = 0
    expanded_nodes: int = 0
    cutoffs: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    evaluation_cache_hits: int = 0
    inference_calls: int = 0
    evaluated_positions: int = 0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0
    completed_depth: int = 0

    def to_dict(self) -> dict:
        """WHAT: Export derived rates. HOW: Guard zero denominators. WHY: Comparable timing
        scope.
        """
        return {
            **asdict(self),
            "nodes_per_second": self.nodes / self.total_seconds if self.total_seconds else 0.0,
            "cutoff_rate": self.cutoffs / self.expanded_nodes if self.expanded_nodes else 0.0,
            "evaluation_latency_ms": (
                1000 * self.inference_seconds / self.evaluated_positions
                if self.evaluated_positions
                else 0.0
            ),
        }


@dataclass(frozen=True)
class SearchResult:
    """WHAT: Committed root decision. HOW: Last full iteration plus metrics. WHY: Honest
    interruption.
    """

    move: chess.Move | None
    score: float | None
    claim_draw: bool
    terminal: str | None
    principal_variation: tuple[chess.Move, ...]
    stop_reason: str
    metrics: SearchMetrics
    claim_move: chess.Move | None = None
    claim_reason: str | None = None

    def to_dict(self) -> dict:
        """WHAT: JSON-ready decision. HOW: UCI moves and explicit claim/terminal. WHY: No
        ambiguity.
        """
        return {
            "move": self.move.uci() if self.move else None,
            "score": self.score,
            "score_units": "mover_expected_outcome; mate band +/-10000",
            "claim_draw": self.claim_draw,
            "claim_move": self.claim_move.uci() if self.claim_move else None,
            "claim_reason": self.claim_reason,
            "terminal": self.terminal,
            "principal_variation": [move.uci() for move in self.principal_variation],
            "stop_reason": self.stop_reason,
            "metrics": self.metrics.to_dict(),
        }


class _Interrupted(Exception):
    pass


@dataclass(frozen=True)
class _Line:
    score: float
    move: chess.Move | None = None
    claim: bool = False
    pv: tuple[chess.Move, ...] = ()


class SearchEngine:
    """WHAT: Choose a move with iterative-deepening alpha-beta and existing neural heads.

    HOW: Fail-soft negamax, policy-sorted legal children, optional zero draw action,
    mate-distance scores, context-safe TT bounds and sibling leaf batching. Commit
    only full root iterations. WHY: Simple full-width search makes correctness
    independently testable before any selective pruning or chess-strength claims.
    Worst-case O(b^d) nodes; good ordering can approach O(b^(d/2)). Retained caches
    are bounded, recursion <=64. See SEARCH_SPEC.md for cooperative wall-time limits.
    """

    def __init__(self, evaluator: Evaluator, *, clock: Callable[[], float] = perf_counter):
        self.evaluator = evaluator
        self.clock = clock

    def _check(self, *, entering: bool = False) -> None:
        """WHAT: Enforce cooperative budgets. HOW: Check at safe boundaries. WHY: Retain results."""
        if self.cancel is not None and self.cancel():
            self.stop_reason = "cancelled"
            raise _Interrupted
        if self.clock() - self.started >= self.limits.time_budget_seconds:
            self.stop_reason = "time_budget"
            raise _Interrupted
        if entering and self.metrics.nodes >= self.limits.node_budget:
            self.stop_reason = "node_budget"
            raise _Interrupted

    def _evaluate(self, positions: list[PositionInput]) -> dict[tuple, Evaluation]:
        """WHAT: Reuse or infer snapshots. HOW: LRU, deduplicate, bounded chunks. WHY: Limit
        dispatch.
        """
        results = {}
        missing = {}
        for position in positions:
            cached = self.evaluations.get(position.key)
            if cached is not None:
                self.metrics.evaluation_cache_hits += 1
                results[position.key] = cached
            else:
                missing[position.key] = position
        pending = list(missing.values())
        size = self.limits.inference_batch_size
        for offset in range(0, len(pending), size):
            self._check()
            chunk = pending[offset : offset + size]
            synchronize = getattr(self.evaluator, "synchronize", lambda: None)
            synchronize()
            started = self.clock()
            values = self.evaluator.evaluate(chunk)
            synchronize()
            self.metrics.inference_seconds += self.clock() - started
            self.metrics.inference_calls += 1
            self.metrics.evaluated_positions += len(chunk)
            if len(values) != len(chunk):
                raise ValueError("Evaluator returned the wrong batch length")
            for position, evaluation in zip(chunk, values, strict=True):
                self._validate_evaluation(position, evaluation)
                results[position.key] = evaluation
                self.evaluations.put(position.key, evaluation)
            self._check()
        return results

    @staticmethod
    def _validate_evaluation(position: PositionInput, value: Evaluation) -> None:
        if not math.isfinite(value.value) or not -1 <= value.value <= 1:
            raise ValueError("Evaluator utility must be finite in [-1,1]")
        if set(value.policy) != set(position.board.legal_moves):
            raise ValueError("Evaluator policy must cover exactly the legal moves")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in value.policy.values()):
            raise ValueError("Policy probabilities must be finite in [0,1]")
        if value.policy and not math.isclose(sum(value.policy.values()), 1, abs_tol=1e-5):
            raise ValueError("Legal policy probabilities must sum to one")

    @staticmethod
    def _terminal(board: chess.Board, ply: int) -> tuple[float, str] | None:
        """WHAT: Score endings. HOW: Mate first, then draws. WHY: Rules override networks."""
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            return None
        return (-MATE + ply if outcome.winner is not None else 0.0, outcome.termination.name)

    @staticmethod
    def _claim_details(board: chess.Board) -> tuple[str, chess.Move | None]:
        """WHAT: Explain a chosen claim. HOW: Check now, then legal intended moves.

        WHY: A claim requiring a declared move must expose that move to its caller;
        it is not a played move or an automatic terminal. Always restore the board.
        """
        if board.is_fifty_moves():
            return "FIFTY_MOVES", None
        if board.is_repetition(3):
            return "THREEFOLD_REPETITION", None
        for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
            board.push(move)
            try:
                if board.is_fifty_moves():
                    return "FIFTY_MOVES", move
                if board.is_repetition(3):
                    return "THREEFOLD_REPETITION", move
            finally:
                board.pop()
        raise RuntimeError("Chosen claim has no legal witness")

    def _children(self, state: SearchState, moves: list[chess.Move], depth: int):
        """WHAT: Yield ordered children. HOW: Batch only sibling horizon leaves. WHY: Keep DFS
        pruning.
        """
        if depth != 1:
            for move in moves:
                yield move, None
            return
        size = self.limits.inference_batch_size
        for offset in range(0, len(moves), size):
            self._check(entering=True)
            chunk = moves[offset : offset + size]
            snapshots = {}
            for move in chunk:
                state.push(move)
                try:
                    if self._terminal(state.board, 0) is None:
                        snapshots[move] = state.snapshot()
                finally:
                    state.pop()
            evaluations = self._evaluate(list(snapshots.values()))
            for move in chunk:
                position = snapshots.get(move)
                yield move, evaluations[position.key] if position is not None else None

    def _node(
        self,
        state: SearchState,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        prefetched: Evaluation | None = None,
    ) -> _Line:
        """WHAT: Search one fail-soft negamax node.

        HOW: V=max(-V_child), reversing the child window to [-beta,-alpha]. A
        lower bound >=beta or upper bound <=alpha is sufficient for a TT cutoff;
        other bounds only supply ordering. Optional claims initialize V=0. Store
        UPPER on fail-low, LOWER on fail-high and EXACT inside the original window.
        WHY: Bound labels prevent pruning evidence from becoming a false exact score.
        """
        self._check(entering=True)
        self.metrics.nodes += 1
        terminal = self._terminal(state.board, ply)
        if terminal is not None:
            return _Line(terminal[0])
        claim = state.board.can_claim_draw()
        if depth == 0:
            position = state.snapshot()
            value = prefetched or self._evaluate([position])[position.key]
            return _Line(max(0.0, value.value) if claim else value.value)
        key = state.key()
        entry = self.transpositions.get(key)
        if entry is not None:
            self.metrics.tt_hits += 1
            score = load_score(entry.score, ply)
            if entry.depth >= depth and (
                entry.bound == Bound.EXACT
                or entry.bound == Bound.LOWER
                and score >= beta
                or entry.bound == Bound.UPPER
                and score <= alpha
            ):
                self.metrics.tt_cutoffs += 1
                return _Line(
                    score, entry.move, entry.claim_draw, (entry.move,) if entry.move else ()
                )
        original_alpha = alpha
        best = _Line(0.0, claim=True) if claim else _Line(-math.inf)
        alpha = max(alpha, best.score)
        self.metrics.expanded_nodes += 1
        if alpha < beta:
            position = state.snapshot()
            evaluation = self._evaluate([position])[position.key]
            moves = sorted(
                state.board.legal_moves,
                key=lambda move: (
                    0 if entry is not None and entry.move == move else 1,
                    -evaluation.policy[move],
                    move.uci(),
                ),
            )
            for move, child_evaluation in self._children(state, moves, depth):
                state.push(move)
                try:
                    child = self._node(state, depth - 1, -beta, -alpha, ply + 1, child_evaluation)
                finally:
                    state.pop()
                score = -child.score
                if score > best.score:
                    best = _Line(score, move, False, (move, *child.pv))
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        if alpha >= beta:
            self.metrics.cutoffs += 1
        bound = (
            Bound.UPPER
            if best.score <= original_alpha
            else (Bound.LOWER if best.score >= beta else Bound.EXACT)
        )
        self.transpositions.put(
            key, TTEntry(depth, store_score(best.score, ply), bound, best.move, best.claim)
        )
        return best

    def search(
        self,
        board: chess.Board,
        limits: SearchLimits | None = None,
        *,
        cancel: Callable[[], bool] | None = None,
    ) -> SearchResult:
        """WHAT: Return the last completed decision. HOW: Deepen and catch budget stops.

        WHY: Partial iterations are biased by ordering. A zero-depth fallback score
        remains None; no unfinished evaluation is advertised as a search result.
        """
        self.started = self.clock()
        self.limits = limits if limits is not None else SearchLimits()
        if not isinstance(self.limits, SearchLimits):
            raise ValueError("limits must be SearchLimits")
        self.cancel = cancel
        self.metrics = SearchMetrics()
        self.stop_reason = "depth_limit"
        self.evaluations: LRU[Evaluation] = LRU(self.limits.cache_entries)
        self.transpositions: LRU[TTEntry] = LRU(self.limits.transposition_entries)
        state = SearchState(board, self.evaluator.history_frames)
        terminal = self._terminal(state.board, 0)
        if terminal is not None:
            self.metrics.total_seconds = self.clock() - self.started
            return SearchResult(None, terminal[0], False, terminal[1], (), "terminal", self.metrics)
        can_claim = state.board.can_claim_draw()
        fallback = min(state.board.legal_moves, key=lambda move: move.uci())
        best = _Line(0.0, claim=True) if can_claim else _Line(0.0, fallback)
        completed = False
        try:
            for depth in range(1, self.limits.depth + 1):
                iteration = self._node(state, depth, -math.inf, math.inf, 0)
                self._check()
                best = iteration
                completed = True
                self.metrics.completed_depth = depth
        except _Interrupted:
            pass
        except KeyboardInterrupt:
            self.stop_reason = "cancelled"
        reason, intended = self._claim_details(state.board) if best.claim else (None, None)
        self.metrics.total_seconds = self.clock() - self.started
        return SearchResult(
            best.move,
            best.score if completed or best.claim else None,
            best.claim,
            None,
            best.pv,
            self.stop_reason,
            self.metrics,
            intended,
            reason,
        )
