"""Full structural, integrity and semantic validation with bounded game-level replay."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import chess
import numpy as np

from drewbot.config import SPLITS, DataConfig, split_for
from drewbot.data.dataset import ShardReader
from drewbot.data.ingest import GameRecord, filter_reason, identity
from drewbot.data.samples import iter_samples, selected_plies
from drewbot.data.storage import (
    ARRAY_SCHEMA,
    DATASET_VERSION,
    ENCODING_VERSION,
    connect,
    contained,
    read_manifest,
)
from drewbot.observability import digest_file


def require(condition: bool, message: str) -> None:
    """WHAT: Enforce an invariant under -O. HOW: Raise ValueError. WHY: Reliable validation."""
    if not condition:
        raise ValueError(message)


def validate_dataset(directory: Path) -> dict:
    """WHAT: Verify the entire published corpus, including every sample's meaning.

    HOW: Check hashes/arrays, then replay registry records and compare expected selected
    positions in each split's write order. A temporary disk-backed seen-game table detects
    duplicate blocks. WHY: Checksums alone cannot detect consistently mislabeled samples.
    Time includes full encoding/rule checks; RAM is O(one game + shard mappings + inventory).
    """
    directory = directory.resolve()
    manifest = read_manifest(directory)
    require(manifest["dataset_version"] == DATASET_VERSION, "Dataset version mismatch")
    require(manifest["encoding_version"] == ENCODING_VERSION, "Encoding version mismatch")
    config = DataConfig(**manifest["configuration"])
    connection = connect(directory)
    try:
        require(
            DataConfig(
                **json.loads(
                    connection.execute("SELECT value FROM settings WHERE key='config'").fetchone()[
                        0
                    ]
                )
            ).semantic()
            == config.semantic(),
            "Registry configuration differs",
        )
        registered = {
            row[0]: json.loads(row[1])
            for row in connection.execute("SELECT path,metadata FROM shards")
        }
        require(len(registered) == len(manifest["shards"]), "Shard inventory count mismatch")
        for shard in manifest["shards"]:
            require(shard["path"] in registered, "Unregistered shard")
            require(
                DataConfig(**shard["configuration"]).semantic() == config.semantic(),
                "Shard processing configuration differs from dataset",
            )
            require(
                {key: value for key, value in shard.items() if key != "path"}
                == registered.pop(shard["path"]),
                "Manifest shard metadata mismatch",
            )
        require(not registered, "Missing shard in manifest")
        positions = 0
        with tempfile.TemporaryDirectory(prefix="drewbot-validation-") as temporary:
            seen = sqlite3.connect(Path(temporary) / "seen.sqlite3")
            seen.execute("PRAGMA cache_size=-2048")
            seen.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
            try:
                for split in SPLITS:
                    expected = iter(())
                    current_id = None
                    split_positions = 0
                    for shard in (item for item in manifest["shards"] if item["split"] == split):
                        path = contained(directory, shard["path"])
                        require(
                            json.loads((path / "metadata.json").read_text())
                            == {key: value for key, value in shard.items() if key != "path"},
                            "Shard sidecar mismatch",
                        )
                        files = shard["files"]
                        require(
                            set(files)
                            == {f"{key}.npy" for key in ARRAY_SCHEMA}
                            | {"offsets.npy", "samples.jsonl"},
                            "Unexpected shard file schema",
                        )
                        for filename, integrity in files.items():
                            file = contained(path, filename)
                            require(
                                file.stat().st_size == integrity["bytes"]
                                and digest_file(file) == integrity["sha256"],
                                f"Integrity failure: {file}",
                            )
                        reader = ShardReader(path)
                        try:
                            count = shard["positions"]
                            require(count > 0, "Empty shard")
                            for name, (dtype, tail) in ARRAY_SCHEMA.items():
                                array = reader.arrays[name]
                                require(
                                    array.shape == (count, *tail) and array.dtype == dtype,
                                    f"Wrong shape/dtype: {name}",
                                )
                                if np.issubdtype(dtype, np.floating):
                                    require(
                                        bool(np.isfinite(array).all()), f"Nonfinite {name} features"
                                    )
                            offsets = reader.arrays["offsets"]
                            require(
                                offsets.dtype == np.uint64 and offsets.shape == (count + 1,),
                                "Wrong metadata offset schema",
                            )
                            require(
                                int(offsets[0]) == 0
                                and int(offsets[-1]) == (path / "samples.jsonl").stat().st_size
                                and bool(np.all(offsets[1:] > offsets[:-1])),
                                "Invalid metadata offsets",
                            )
                            shard_games = set()
                            for index in range(count):
                                row = reader.row(index)
                                game_id = row["metadata"]["game_id"]
                                shard_games.add(game_id)
                                require(row["metadata"]["split"] == split, "Sample split mismatch")
                                if game_id != current_id:
                                    require(
                                        next(expected, None) is None, "Missing selected positions"
                                    )
                                    try:
                                        seen.execute("INSERT INTO seen VALUES (?)", (game_id,))
                                    except sqlite3.IntegrityError as error:
                                        raise ValueError(
                                            "Duplicate game block or cross-split leakage"
                                        ) from error
                                    stored = connection.execute(
                                        "SELECT * FROM games WHERE id=?", (game_id,)
                                    ).fetchone()
                                    require(
                                        stored is not None and stored["split"] == split,
                                        "Unregistered game or split leakage",
                                    )
                                    record = GameRecord(**json.loads(stored["record"]))
                                    plies = json.loads(stored["plies"])
                                    require(
                                        record.game_id
                                        == identity(record.initial_fen, record.moves)
                                        == game_id,
                                        "Game identity mismatch",
                                    )
                                    require(
                                        split_for(game_id, config) == split,
                                        "Wrong deterministic split",
                                    )
                                    require(
                                        plies == selected_plies(record, config)
                                        and len(plies) == stored["positions"],
                                        "Sampling completeness mismatch",
                                    )
                                    expected = iter_samples(record, plies, split)
                                    current_id = game_id
                                target = next(expected, None)
                                require(target is not None, "Extra or duplicate sample")
                                require(
                                    row["metadata"] == target["metadata"],
                                    "Position metadata mismatch",
                                )
                                for name in ARRAY_SCHEMA:
                                    require(
                                        np.array_equal(row[name], target[name]),
                                        f"Sample {name} mismatch at "
                                        f"{game_id}:{row['metadata']['ply']}",
                                    )
                                split_positions += 1
                            require(len(shard_games) == shard["games"], "Shard game count mismatch")
                        finally:
                            reader.close()
                    require(next(expected, None) is None, "Missing final selected positions")
                    stored_count = connection.execute(
                        "SELECT COUNT(*),COALESCE(SUM(positions),0) FROM games WHERE split=?",
                        (split,),
                    ).fetchone()
                    require(
                        manifest["counts"][split]
                        == {"games": stored_count[0], "positions": stored_count[1]},
                        "Manifest totals mismatch",
                    )
                    require(split_positions == stored_count[1], "Position completeness mismatch")
                    positions += split_positions
                # Zero-position/short games still need valid identities, splits and sample counts.
                for stored in connection.execute("SELECT * FROM games"):
                    record = GameRecord(**json.loads(stored["record"]))
                    require(
                        filter_reason(record, config) is None,
                        "Stored game violates dataset filters",
                    )
                    require(
                        record.result == stored["result"] == record.headers.get("Result", "*"),
                        "Registry outcome mismatch",
                    )
                    board = chess.Board(record.initial_fen)
                    require(
                        board.is_valid() and not board.chess960, "Invalid registry initial board"
                    )
                    require(
                        len(record.moves) <= config.max_game_plies, "Registry game exceeds limit"
                    )
                    for uci in record.moves:
                        move = chess.Move.from_uci(uci)
                        require(
                            not board.is_game_over() and move in board.legal_moves,
                            "Illegal registry game",
                        )
                        board.push(move)
                    require(
                        board.outcome() is None or record.result in {"*", board.result()},
                        "Registry outcome contradicts terminal board",
                    )
                    plies = selected_plies(record, config)
                    require(
                        stored["id"] == identity(record.initial_fen, record.moves),
                        "Bad registry identity",
                    )
                    require(
                        stored["split"] == split_for(stored["id"], config),
                        "Registry split mismatch",
                    )
                    require(
                        json.loads(stored["plies"]) == plies and stored["positions"] == len(plies),
                        "Registry completeness mismatch",
                    )
                    if plies:
                        require(
                            seen.execute(
                                "SELECT 1 FROM seen WHERE id=?", (stored["id"],)
                            ).fetchone()
                            is not None,
                            "Missing entire game",
                        )
            finally:
                seen.close()
        return {
            "valid": True,
            "games": sum(item["games"] for item in manifest["counts"].values()),
            "positions": positions,
            "shards": len(manifest["shards"]),
            "counts": manifest["counts"],
        }
    finally:
        connection.close()
