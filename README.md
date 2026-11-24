# DrewBot

DrewBot is a chess engine research project for experimenting with geometric attention,
move ordering, and position evaluation. This release runs on CPU, includes a command-line
demo, and supports training and evaluation on synthetic or user-supplied PGNs.

## Quick start

Run from the repository root on macOS or Linux with Python 3.11 or later.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Try two hand-authored positions:

```sh
drewbot-search search --random-model --device cpu --depth 1 --time-budget-seconds 10 \
  --fen 'k7/8/1QK5/8/8/8/8/8 w - - 0 1'

drewbot-search search --baseline --device cpu --depth 2 \
  --fen '4k3/8/8/8/8/8/P5P1/4K3 w - - 0 1'
```

The first position has mate in one. The neural demo initializes the six-layer model
with seed 42, runs inference to order legal moves and score leaves, and detects mate
with the chess rules in search. The random outputs illustrate the mechanics and cannot
establish playing strength. The second command uses the existing tactical search with
handcrafted evaluation and equal root priors. Both commands print a JSON result with the
chosen move, principal variation, and search metrics.

## How the model and search fit together

- The encoder maps up to eight boards to 64 square tokens with mover-relative piece
  channels and auxiliary features for game state.
- The transformer adds learned, position-dependent geometric attention bias to attention
  logits. Set `gab_enabled` to `false` in `ModelSettings` to run a standard-attention
  comparison. The architecture is inspired by [Chessformer](https://arxiv.org/abs/2605.19091).
- The policy head scores 4,352 move slots, and the legal mask removes unavailable moves.
  The value head predicts loss, draw, and win from the mover's perspective.
- Iterative-deepening alpha-beta search uses legal policy probabilities for ordering and
  `P(win) − P(loss)` for leaf scores. It handles terminal positions directly. A FEN
  encodes a single board state. Supply subsequent UCI moves with `--moves` to build
  replay history.

## Train and evaluate locally

Generate a small synthetic dataset and run the default eight-update training pilot:

```sh
mkdir -p data
python -c "from pathlib import Path; from drewbot.data.benchmark import synthetic_pgn; synthetic_pgn(Path('data/synthetic.pgn'), 32, plies=8)"
drewbot-data process data/synthetic.pgn
drewbot-train train --device cpu
```

The synthetic generator uses legal trajectories with artificial outcome labels for
nonterminal games. It exercises the training loop. To train with your own games, pass
their PGN paths explicitly to `drewbot-data process`. Choose a fresh project-relative
`--output` for a separate dataset and pass `drewbot-train --config <training.json> train`
with a matching `dataset` field.

Use the run ID printed by training to evaluate the checkpoint on the held-out test
split, or to search with the resulting weights:

```sh
run_id=YOUR_RUN_ID
drewbot-train evaluate "checkpoints/$run_id/final.pt" --device cpu --output experiments/eval
drewbot-search search --checkpoint "checkpoints/$run_id/final.pt" --device cpu
```

Pretrained weights and strength results are unavailable in this release. The test
suite exercises encoding, model math, search, data ingestion, training, and evaluation
with composed positions and synthetic games. Run it with `python -m pytest`.
