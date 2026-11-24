"""Opt-in CPU negamax with one root policy call and bounded tactical leaves."""

from __future__ import annotations

import math
from collections import Counter

import chess

from drewbot.search.config import SearchLimits
from drewbot.search.engine import SearchEngine, SearchMetrics, SearchResult
from drewbot.search.state import SearchState

PIECE_CP = (0, 100, 320, 335, 510, 950, 0)
MATE_CP = 100000
INFINITY = MATE_CP * 2


def _square_tables():
    """Create simple positional terms from advancement and centralization."""
    early, late = [], []
    for square in chess.SQUARES:
        file, rank = chess.square_file(square), chess.square_rank(square)
        center = 7 - abs(2 * file - 7) - abs(2 * rank - 7)
        edge = min(file, 7 - file)
        early.append(
            (
                0,
                6 * rank + 5 * edge,
                9 * center,
                4 * center + 5 * min(rank, 2),
                2 * center + 14 * (rank == 6),
                2 * center - 5 * max(0, rank - 1),
                -9 * center - 12 * rank + 18 * (rank == 0 and file in (1, 2, 6)),
            )
        )
        late.append(
            (
                0,
                9 * rank + 2 * rank * rank + 2 * edge,
                6 * center,
                3 * center,
                2 * center + 14 * (rank == 6),
                2 * center,
                8 * center,
            )
        )
    return tuple(early), tuple(late)


EARLY, LATE = _square_tables()
FILES = tuple(chess.BB_FILES)
PASSED = tuple(
    sum(
        chess.BB_SQUARES[other]
        for other in chess.SQUARES
        if chess.square_rank(other) > chess.square_rank(square)
        and abs(chess.square_file(other) - chess.square_file(square)) <= 1
    )
    for square in chess.SQUARES
)


def static_value(board: chess.Board) -> int:
    """Evaluate material, development, king position and pawn structure in mover cp."""
    phase = min(
        24,
        board.knights.bit_count()
        + board.bishops.bit_count()
        + 2 * board.rooks.bit_count()
        + 4 * board.queens.bit_count(),
    )
    piece_boards = (
        0,
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    )
    values = []
    for color in (chess.WHITE, chess.BLACK):
        occupancy = board.occupied_co[color]
        own_pawns = board.pawns & occupancy
        enemy_pawns = board.pawns & board.occupied_co[not color]
        value = 25 if (board.bishops & occupancy).bit_count() >= 2 else 0
        for piece in chess.PIECE_TYPES:
            positions = piece_boards[piece] & occupancy
            value += PIECE_CP[piece] * positions.bit_count()
            for square in chess.scan_forward(positions):
                relative = square if color else square ^ 56
                value += (
                    EARLY[relative][piece] * phase + LATE[relative][piece] * (24 - phase)
                ) // 24
                if piece == chess.PAWN:
                    passed_mask = PASSED[relative]
                    if not color:
                        passed_mask = chess.flip_vertical(passed_mask)
                    if not passed_mask & enemy_pawns:
                        rank = chess.square_rank(relative)
                        value += rank * rank * (50 - phase) // 24
                    if (own_pawns & FILES[chess.square_file(square)]).bit_count() > 1:
                        value -= 10
                elif piece == chess.ROOK:
                    file = FILES[chess.square_file(square)]
                    if not file & own_pawns:
                        value += 12 if file & enemy_pawns else 20
        values.append(value)
    white = values[0] - values[1]
    return white if board.turn else -white


def rule_key(board: chess.Board) -> tuple:
    """Use exact board identity with legal en passant for repetition bookkeeping."""
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.turn,
        board.castling_rights,
        board.ep_square if board.has_legal_en_passant() else None,
    )


class _Stop(Exception):
    """Abort a partial iteration so only a complete root decision is committed."""


class TacticalResult(SearchResult):
    """Use the common decision interface and identify the classical score scale."""

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "score_units": "heuristic mover centipawns; mate band +/-100000"
        }


class TacticalSearch(SearchEngine):
    """Rank all root moves once, then run classical alpha-beta and tactical continuation.

    Full-width main iterations include every legal move. Tactical leaves visit
    captures and promotions or all evasions in check. The deadline, node cap
    and last completed root iteration take precedence over additional depth.
    """

    def __init__(self, evaluator, *, prior_cp=35, qdepth=5, clock=None):
        super().__init__(evaluator, **({"clock": clock} if clock is not None else {}))
        if not 0 <= prior_cp <= 90 or not 1 <= qdepth <= 10:
            raise ValueError("Invalid tactical search profile")
        self.prior_cp = prior_cp
        self.qdepth = qdepth

    def _visit(self):
        self._stop()
        if self.metrics.nodes >= self.limits.node_budget:
            self.stop_reason = "node_budget"
            raise _Stop
        self.metrics.nodes += 1

    def _stop(self):
        if self.cancel is not None and self.cancel():
            self.stop_reason = "cancelled"
            raise _Stop
        if self.clock() >= self.deadline:
            self.stop_reason = "time_budget"
            raise _Stop

    def _push(self, move):
        self.board.push(move)
        key = rule_key(self.board)
        self.repetitions[key] += 1
        if self.repetitions[key] == 2:
            self.repeated += 1
        return key

    def _pop(self, key):
        if self.repetitions[key] == 2:
            self.repeated -= 1
        self.repetitions[key] -= 1
        if not self.repetitions[key]:
            del self.repetitions[key]
        self.board.pop()

    def _terminal(self, key, moves, check, ply):
        if not moves:
            return -MATE_CP + 100 * ply if check else 0
        if (
            self.board.halfmove_clock >= 150
            or self.repetitions[key] >= 5
            or self.board.is_insufficient_material()
        ):
            return 0
        return None

    def _claim(self, key, moves):
        board = self.board
        if self.repetitions[key] >= 3 or board.halfmove_clock >= 100:
            return True
        if self.repeated or board.halfmove_clock == 99:
            for move in moves:
                board.push(move)
                try:
                    if (self.repeated and self.repetitions[rule_key(board)] >= 2) or (
                        board.halfmove_clock >= 100 and any(board.generate_legal_moves())
                    ):
                        return True
                finally:
                    board.pop()
        return False

    def _order(self, moves, preferred=None, killer=None):
        board = self.board

        def priority(move):
            if move == preferred:
                return 20000
            victim = board.piece_type_at(move.to_square)
            if victim or move.promotion or board.is_en_passant(move):
                worth = PIECE_CP[victim or chess.PAWN] if board.is_capture(move) else 0
                return (
                    10000
                    + 8 * worth
                    + PIECE_CP[move.promotion or 0]
                    - PIECE_CP[board.piece_type_at(move.from_square)]
                )
            return 1 if move == killer else 0

        return sorted(moves, key=priority, reverse=True)

    def _child(self, move, depth, alpha, beta, ply, *, quiet=False):
        key = self._push(move)
        try:
            return -(
                self._quiet(depth, alpha, beta, ply, key)
                if quiet
                else self._node(depth, alpha, beta, ply, key)
            )
        finally:
            self._pop(key)

    def _quiet(self, depth, alpha, beta, ply, key):
        self._visit()
        board = self.board
        check = board.is_check()
        if check or self.repeated or board.halfmove_clock >= 99:
            legal = list(board.generate_legal_moves())
            moves = legal if check else [m for m in legal if m.promotion or board.is_capture(m)]
        else:
            moves = list(board.generate_legal_captures())
            promotion_pawns = board.pawns & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2)
            if promotion_pawns:
                moves.extend(
                    m
                    for m in board.generate_legal_moves(from_mask=promotion_pawns)
                    if m.promotion and not board.is_capture(m)
                )
            first = None if moves else next(board.generate_legal_moves(), None)
            legal = moves or ([first] if first is not None else [])
        terminal = self._terminal(key, legal, check, ply)
        if terminal is not None:
            return terminal
        claim = self._claim(key, legal)
        if check:
            best = 0 if claim else -INFINITY
        else:
            stand_pat = static_value(board)
            best = max(0, stand_pat) if claim else stand_pat
            if depth <= 0 or best >= beta:
                return best
        alpha = max(alpha, best)
        if alpha >= beta:
            return best
        for move in self._order(moves):
            if depth <= -2:
                child = self._push(move)
                try:
                    self._visit()
                    replies = list(board.generate_legal_moves())
                    value = self._terminal(child, replies, board.is_check(), ply + 1)
                    if value is None:
                        value = static_value(board)
                        if self._claim(child, replies):
                            value = max(0, value)
                    score = -value
                finally:
                    self._pop(child)
            else:
                score = self._child(move, depth - 1, -beta, -alpha, ply + 1, quiet=True)
            best = max(best, score)
            if best >= beta:
                break
            alpha = max(alpha, best)
        return best

    def _node(self, depth, alpha, beta, ply, key):
        if depth == 0:
            return self._quiet(self.qdepth, alpha, beta, ply, key)
        self._visit()
        board = self.board
        moves = list(board.generate_legal_moves())
        terminal = self._terminal(key, moves, board.is_check(), ply)
        if terminal is not None:
            return terminal
        best = 0 if self._claim(key, moves) else -INFINITY
        alpha = max(alpha, best)
        if alpha >= beta:
            return best
        self.metrics.expanded_nodes += 1
        cache_limit = self.limits.transposition_entries
        for move in self._order(moves, self.orders.get(key), self.killers.get(ply)):
            quiet = not board.is_capture(move) and not move.promotion
            score = self._child(move, depth - 1, -beta, -alpha, ply + 1)
            if score > best:
                best = score
                if len(self.orders) < cache_limit or key in self.orders:
                    self.orders[key] = move
            if score >= beta:
                self.metrics.cutoffs += 1
                if quiet:
                    self.killers[ply] = move
                break
            alpha = max(alpha, score)
        return best

    def search(self, board, limits=None, *, cancel=None):
        """Choose a legal root action, retaining the policy fallback on interruption."""
        started = self.clock()
        self.limits = (
            limits
            if limits is not None
            else SearchLimits(depth=8, time_budget_seconds=0.35, node_budget=100000)
        )
        if not isinstance(self.limits, SearchLimits):
            raise ValueError("Expected SearchLimits")
        self.deadline = started + 0.98 * self.limits.time_budget_seconds
        self.cancel = cancel
        self.stop_reason = "depth_limit"
        self.metrics = SearchMetrics()
        state = SearchState(board, self.evaluator.history_frames)
        self.board = board.root()
        self.repetitions = Counter({rule_key(self.board): 1})
        self.repeated = 0
        for move in board.move_stack:
            self._push(move)
        self.orders, self.killers = {}, {}
        key = rule_key(self.board)
        legal = list(self.board.generate_legal_moves())
        terminal = self._terminal(key, legal, self.board.is_check(), 0)
        if terminal is not None:
            outcome = self.board.outcome(claim_draw=False)
            self.metrics.total_seconds = self.clock() - started
            return TacticalResult(
                None, terminal, False, outcome.termination.name, (), "terminal", self.metrics
            )
        claim = self._claim(key, legal)
        best_move = min(legal, key=chess.Move.uci)
        best_score, use_claim = (0, True) if claim else (None, False)
        try:
            self._stop()
            inference = self.clock()
            probabilities = self.evaluator.evaluate([state.snapshot()])[0].policy
            self.metrics.inference_calls = self.metrics.evaluated_positions = 1
            self.metrics.inference_seconds = self.clock() - inference
            ranked = sorted(legal, key=lambda m: (-probabilities[m], m.uci()))
            best_move = ranked[0]
            top = max(1e-9, probabilities[best_move])
            bonuses = {
                move: -min(
                    self.prior_cp, round(10 * math.log(top / max(1e-9, probabilities[move])))
                )
                if self.prior_cp
                else 0
                for move in ranked
            }
            for depth in range(1, self.limits.depth + 1):
                self._visit()
                chosen, value = None, 0 if claim else -INFINITY
                order = [best_move, *(move for move in ranked if move != best_move)]
                for move in order:
                    bonus = bonuses[move]
                    score = self._child(move, depth - 1, -INFINITY, bonus - value, 1) + bonus
                    if score > value:
                        value, chosen = score, move
                best_score, use_claim = value, chosen is None
                if chosen is not None:
                    best_move = chosen
                self.metrics.completed_depth = depth
                if abs(value) >= MATE_CP - 100 * depth - self.prior_cp:
                    break
        except _Stop:
            pass
        self.metrics.total_seconds = self.clock() - started
        claim_reason, witness = self._claim_details(board) if use_claim else (None, None)
        move = None if use_claim else best_move
        return TacticalResult(
            move,
            best_score,
            use_claim,
            None,
            (move,) if move else (),
            self.stop_reason,
            self.metrics,
            witness,
            claim_reason,
        )
