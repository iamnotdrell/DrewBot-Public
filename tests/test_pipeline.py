"""Local integration checks for streaming, split isolation, transactions and tensor loading."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import sqlite3
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import zstandard
from torch.utils.data import DataLoader

from drewbot.config import DataConfig, project_path
from drewbot.data import pipeline
from drewbot.data.benchmark import synthetic_pgn
from drewbot.data.cli import main
from drewbot.data.dataset import ChessDataset
from drewbot.data.ingest import safe_games, stream_games
from drewbot.data.samples import auxiliary
from drewbot.data.storage import connect, writer_lock
from drewbot.data.validate import validate_dataset
from drewbot.hardware import inspect_hardware
from drewbot.observability import digest_file, write_json

FIXTURE = Path(__file__).parent / "fixtures/fools_mate.pgn"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "project_path", lambda name: tmp_path / name)
    return tmp_path


def test_end_to_end_dedup_split_replay_and_loader(workspace):
    source = workspace / "games.pgn"
    synthetic_pgn(source, 50, plies=8)
    config = DataConfig(
        output="dataset", shard_positions=7, max_positions_per_game=3, player="Synthetic W 0"
    )
    result = pipeline.process([source], config)
    assert result["games"] == 50 and result["positions"] == 150
    directory = workspace / "dataset"
    validation = validate_dataset(directory)
    assert validation["positions"] == 150
    assert all(validation["counts"][split]["games"] for split in ("train", "validation", "test"))
    dataset = ChessDataset(directory, cache_shards=1)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        assert batch["tokens"].shape == (4, 64, 96)
        assert batch["auxiliary"].shape == (4, 72)
        assert batch["tokens"].dtype == torch.float32
        assert batch["auxiliary"].dtype == torch.float32
        assert batch["policy"].dtype == torch.int64
        assert batch["legal_mask"].shape == (4, 4352)
        assert batch["legal_mask"].dtype == torch.bool
        assert batch["value_valid"].dtype == torch.bool
        assert not dataset[-1]["tokens"].isnan().any()
        assert len(dataset._readers) == 1
        with pytest.raises(IndexError):
            dataset[len(dataset)]
    finally:
        dataset.close()
    connection = connect(directory)
    try:
        rows = connection.execute("SELECT record FROM games").fetchall()
        assert "white" in {json.loads(row[0])["player_color"] for row in rows}
    finally:
        connection.close()
    # Change source headers but preserve actual trajectories: all are duplicates.
    duplicate = workspace / "duplicate.pgn"
    duplicate.write_text(source.read_text().replace("Synthetic W", "Renamed W"))
    result = pipeline.process([duplicate], config)
    assert result["duplicates"] == 50 and result["games"] == result["positions"] == 0
    assert validate_dataset(directory)["positions"] == 150
    connection = connect(directory)
    try:
        assert connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] == 100
        assert (
            "Renamed W"
            in connection.execute(
                "SELECT detail FROM occurrences WHERE status='duplicate' LIMIT 1"
            ).fetchone()[0]
        )
    finally:
        connection.close()


def test_incremental_limit_resume_and_config_guard(workspace):
    source = workspace / "games.pgn"
    synthetic_pgn(source, 6)
    config = DataConfig(output="dataset", max_games=2, shard_positions=3)
    assert pipeline.process([source], config)["games"] == 2
    result = pipeline.process([source], replace(config, max_games=None))
    assert result["games"] == 4 and result["already_processed"] == 2
    assert pipeline.process([source], config)["already_processed"] == 2
    assert validate_dataset(workspace / "dataset")["games"] == 6
    with pytest.raises(ValueError, match="settings differ"):
        pipeline.process([source], replace(config, seed=43))


@pytest.mark.parametrize(
    "suffix,compress",
    [
        (".pgn", lambda raw: raw),
        (".pgn.gz", gzip.compress),
        (".pgn.bz2", bz2.compress),
        (".pgn.xz", lzma.compress),
        (".pgn.zst", zstandard.ZstdCompressor().compress),
    ],
)
def test_streaming_compression(tmp_path, suffix, compress):
    source = tmp_path / ("games" + suffix)
    source.write_bytes(compress(FIXTURE.read_bytes() + b"\n" + FIXTURE.read_bytes()))
    records = list(stream_games(source, DataConfig()))
    assert len(records) == 2 and all(record and not error for _, record, error in records)
    assert records[0][1].game_id == records[1][1].game_id


def test_concatenated_zstd_and_truncation(tmp_path):
    compressor = zstandard.ZstdCompressor(write_checksum=True)
    first = compressor.compress(FIXTURE.read_bytes() + b"\n")
    second = compressor.compress(FIXTURE.read_bytes())
    source = tmp_path / "games.zst"
    source.write_bytes(first + second)
    assert len(list(stream_games(source, DataConfig()))) == 2
    source.write_bytes(first + second[:-2])
    failures = list(safe_games(source, DataConfig()))
    assert len(failures) == 1 and failures[0][0] == -1
    assert "Truncated" in failures[0][2]


@pytest.mark.parametrize(
    "body",
    [
        '[Result "*"]\n\n1. e5 *',
        '[Result "*"]\n\n1. e4 rubbish *',
        '[Result "*"]\n\n1. e4',
        '[Result "1-0"]\n\n1. e4 0-1',
        '[Result "*"]\n\n1. e4 * e5 *',
        '[Result "*"]\n\n1. e4 ) e5 *',
        '[Variant "Chess960"]\n[Result "*"]\n\n*',
        '[SetUp "1"]\n[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]\n\n*',
    ],
)
def test_bad_games_quarantined_before_positions(workspace, body):
    source = workspace / "bad.pgn"
    source.write_text(body + "\n\n" + FIXTURE.read_text())
    result = pipeline.process([source], DataConfig(output="dataset"))
    assert result["invalid"] == 1
    assert result["games"] == 1 and result["positions"] == 4
    assert validate_dataset(workspace / "dataset")["games"] == 1


def test_limits_and_source_failures_are_explicit(workspace):
    source = workspace / "long.pgn"
    source.write_text(FIXTURE.read_text())
    result = pipeline.process([source], DataConfig(output="limited", max_line_chars=10))
    assert result["source_errors"] == 1 and result["games"] == 0
    result = pipeline.process([source], DataConfig(output="plies", max_game_plies=2))
    assert result["invalid"] == 1 and result["positions"] == 0
    bad_utf8 = workspace / "bad-encoding.pgn"
    bad_utf8.write_bytes(b"\xff\xff")
    assert pipeline.process([bad_utf8], DataConfig(output="encoding"))["source_errors"] == 1


def test_conflicting_duplicate_result_does_not_relabel(workspace):
    first = workspace / "first.pgn"
    second = workspace / "second.pgn"
    first.write_text('[White "Drew"]\n[Result "1-0"]\n\n1. e4 1-0\n')
    second.write_text('[White "Other"]\n[Result "0-1"]\n\n1. e4 0-1\n')
    config = DataConfig(output="dataset", player="Drew")
    pipeline.process([first], config)
    result = pipeline.process([second], config)
    assert result["invalid"] == 1 and result["games"] == 0
    dataset = ChessDataset(workspace / "dataset")
    try:
        assert dataset[0]["value"].item() == 2
    finally:
        dataset.close()


def test_split_identity_stable_across_input_order_and_sampling(workspace):
    source = workspace / "games.pgn"
    synthetic_pgn(source, 15)
    config = DataConfig(output="first", max_positions_per_game=1)
    pipeline.process([source], config)
    texts = source.read_text().strip().split("\n\n")
    # PGN export has header/movetext pairs; reverse complete games, not positions.
    games = ["\n\n".join(texts[index : index + 2]) for index in range(0, len(texts), 2)]
    reversed_source = workspace / "reversed.pgn"
    reversed_source.write_text("\n\n".join(reversed(games)) + "\n")
    pipeline.process([reversed_source], replace(config, output="second", max_positions_per_game=2))
    assignments = []
    for name in ("first", "second"):
        connection = connect(workspace / name)
        try:
            assignments.append(dict(connection.execute("SELECT id,split FROM games")))
        finally:
            connection.close()
    assert assignments[0] == assignments[1] and len(assignments[0]) == 15


def test_writer_failure_rolls_back_even_after_files_move(workspace, monkeypatch):
    config = DataConfig(output="dataset", shard_positions=1)
    pipeline.process([FIXTURE], config)
    source = workspace / "new.pgn"
    source.write_text('[Result "*"]\n\n1. e4 e5 *\n')
    finish = pipeline.ShardWriter.finish

    def fail(writer):
        finish(writer)
        raise RuntimeError("Simulated crash after file publication")

    monkeypatch.setattr(pipeline.ShardWriter, "finish", fail)
    with pytest.raises(RuntimeError, match="Simulated crash"):
        pipeline.process([source], config)
    assert validate_dataset(workspace / "dataset")["positions"] == 4
    monkeypatch.setattr(pipeline.ShardWriter, "finish", finish)
    assert pipeline.process([source], config)["positions"] == 2
    assert validate_dataset(workspace / "dataset")["positions"] == 6


def test_manifest_recovered_after_commit(workspace, monkeypatch):
    config = DataConfig(output="dataset")
    publish = pipeline.publish_manifest
    calls = 0

    def fail_after_commit(directory, connection):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Simulated manifest replacement failure")
        return publish(directory, connection)

    monkeypatch.setattr(pipeline, "publish_manifest", fail_after_commit)
    with pytest.raises(OSError, match="manifest"):
        pipeline.process([FIXTURE], config)
    monkeypatch.setattr(pipeline, "publish_manifest", publish)
    result = pipeline.process([FIXTURE], config)
    assert result["already_processed"] == 1 and result["games"] == 0
    assert validate_dataset(workspace / "dataset")["positions"] == 4


@pytest.mark.parametrize("semantic", [False, True])
def test_integrity_and_semantic_corruption_detected(workspace, semantic):
    pipeline.process([FIXTURE], DataConfig(output="dataset"))
    directory = workspace / "dataset"
    manifest = json.loads((directory / "manifest.json").read_text())
    shard = manifest["shards"][0]
    path = directory / shard["path"]
    values = np.load(path / "value.npy")
    values[0] = 2  # First position is White losing, so true class is 0.
    np.save(path / "value.npy", values)
    if semantic:
        shard["files"]["value.npy"]["sha256"] = digest_file(path / "value.npy")
        sidecar = {key: value for key, value in shard.items() if key != "path"}
        write_json(path / "metadata.json", sidecar)
        write_json(directory / "manifest.json", manifest)
        connection = sqlite3.connect(directory / "registry.sqlite3")
        connection.execute(
            "UPDATE shards SET metadata=? WHERE path=?", (json.dumps(sidecar), shard["path"])
        )
        connection.commit()
        connection.close()
    with pytest.raises(ValueError, match="value mismatch" if semantic else "Integrity failure"):
        validate_dataset(directory)


def test_short_zero_position_game_and_long_history(workspace):
    short = workspace / "short.pgn"
    short.write_text('[Result "*"]\n\n*\n')
    long = workspace / "long.pgn"
    synthetic_pgn(long, 1, plies=120)
    result = pipeline.process(
        [short, long], DataConfig(output="dataset", max_positions_per_game=None)
    )
    assert result["games"] == 2 and result["positions"] >= 60
    assert validate_dataset(workspace / "dataset")["positions"] == result["positions"]


def test_auxiliary_distinguishes_rights_ep_and_repetition():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 10")
    vector = auxiliary(board, 3)
    np.testing.assert_array_equal(vector[:5], [1, 1, 1, 1, 1])
    np.testing.assert_allclose(vector[69:], [0.99, 0.1, 0.6])
    board.castling_rights = 0
    assert not np.array_equal(vector, auxiliary(board, 3))
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert auxiliary(board, 1)[5 + chess.D6] == 1
    board.ep_square = None
    assert auxiliary(board, 1)[5:69].sum() == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"split": (0.8, 0.1, 0.2)},
        {"split": (float("nan"), 0, 1)},
        {"seed": -1},
        {"max_games": 0},
        {"sample_stride": 0},
        {"shard_positions": True},
        {"output": "../escape"},
    ],
)
def test_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        DataConfig(**changes)


def test_paths_locks_cli_and_json_logging(workspace, capsys):
    with pytest.raises(ValueError):
        project_path("/tmp/external")
    with (
        writer_lock(workspace / "locked"),
        pytest.raises(ValueError, match="already being written"),
    ):
        with writer_lock(workspace / "locked"):
            pass
    assert (
        main(["process", str(FIXTURE), "--output", "dataset", "--player", "Synthetic Black"]) == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["positions"] == 4
    assert json.loads(output.err.splitlines()[-1])["event"] == "processing_finished"
    assert main(["validate", str(workspace / "dataset")]) == 0
    capsys.readouterr()
    assert main(["inspect", str(workspace / "dataset"), "--index", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["game"]["player_color"] == "black"


def test_hardware_cpu_operational():
    hardware = inspect_hardware()
    assert hardware["cpu"]["operational"] and hardware["memory_bytes"] > 0
    assert hardware["selected_device"] in {"cpu", "mps"}


def test_mps_tensor_operation_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this environment")
    assert inspect_hardware()["mps"]["operational"]


def test_spawned_dataloader_worker(workspace):
    pipeline.process([FIXTURE], DataConfig(output="dataset", shard_positions=2))
    dataset = ChessDataset(workspace / "dataset")
    try:
        dataset[0]  # Populate parent's cache before pickling for spawn.
        batches = list(
            DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
        )
        assert sum(len(batch["tokens"]) for batch in batches) == 4
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "fen,move,result",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "*"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "*"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", "*"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", "*"),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1b", "*"),
        ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", None, "1-0"),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", None, "1/2-1/2"),
    ],
)
def test_rule_edge_cases_survive_sharding(workspace, fen, move, result):
    import chess.pgn

    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    if move:
        game.add_variation(chess.Move.from_uci(move))
    game.headers["Result"] = result
    source = workspace / "edge.pgn"
    source.write_text(str(game) + "\n")
    stats = pipeline.process([source], DataConfig(output="dataset"))
    assert stats["games"] == 1 and stats["positions"] == int(move is not None)
    assert validate_dataset(workspace / "dataset")["games"] == 1


def test_unknown_outcomes_and_variations_do_not_enter_mainline(workspace):
    source = workspace / "annotated.pgn"
    source.write_text(
        '[White "Drew"]\n[Black "Other"]\n[Result "*"]\n\n'
        "1. e4 {clock annotation\n[%clk 0:03:00]} (1. d4 d5) e5 *\n"
    )
    stats = pipeline.process([source], DataConfig(output="dataset", player="Drew"))
    assert stats["positions"] == 2 and stats["invalid"] == 0
    dataset = ChessDataset(workspace / "dataset")
    try:
        assert not dataset[0]["value_valid"].item()
        assert dataset[0]["value"].item() == -100
    finally:
        dataset.close()


def test_benchmark_metrics_are_finite_and_count_real_work(workspace, monkeypatch):
    from drewbot.data import benchmark as module

    source = workspace / "bench.pgn"
    synthetic_pgn(source, 2, plies=4)
    monkeypatch.setattr(module, "project_path", lambda output: workspace / output)
    result = module.worker(source, "dataset", 2)
    assert result["games"] == 2 and result["positions"] == 8 and result["validated"]
    for key in (
        "games_per_second",
        "positions_per_second",
        "encoding_latency_us",
        "peak_process_rss_bytes",
        "processed_bytes",
        "bytes_per_position",
        "shard_writing_positions_per_second",
        "loading_positions_per_second",
    ):
        assert np.isfinite(result[key]) and result[key] > 0
    assert result["hardware"]["cpu"]["operational"]


def test_source_package_is_not_ignored_with_raw_data():
    import subprocess

    from drewbot.config import ROOT

    checked = subprocess.run(
        ["git", "check-ignore", "src/drewbot/data/ingest.py"], cwd=ROOT, capture_output=True
    )
    assert checked.returncode == 1
