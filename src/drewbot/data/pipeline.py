"""Incremental, transactional PGN-to-shard orchestration; no model or training loop."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from time import perf_counter

from drewbot.config import DataConfig, project_path, split_for
from drewbot.data.ingest import filter_reason, safe_games
from drewbot.data.samples import iter_samples, selected_plies
from drewbot.data.storage import ShardWriter, connect, publish_manifest, writer_lock
from drewbot.observability import digest_file, event


def process(paths: list[Path], config: DataConfig) -> dict:
    """WHAT: Stream input files into an appendable deduplicated dataset.

    HOW: Fingerprint files, validate games, hash split IDs, sample and stage shards inside
    one SQLite transaction. Commit registry, then atomically publish its manifest.
    WHY: Interrupted runs cannot publish partial games; resuming preserves prior samples.
    Memory is independent of total games apart from a small O(shards) inventory; identity
    and occurrence tracking live in SQLite. Input files are read twice (hash + parse).
    """
    paths = sorted({path.resolve() for path in paths})
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Supply at least one existing PGN/archive file")
    directory = project_path(config.output)
    stats = {
        key: 0
        for key in (
            "encountered",
            "games",
            "positions",
            "duplicates",
            "invalid",
            "already_processed",
            "source_errors",
            "filtered",
        )
    }
    stats["encoding_seconds"] = 0.0
    started = perf_counter()
    with writer_lock(directory):
        connection = connect(directory, config)
        try:
            # Repair a prior committed run whose manifest publication was interrupted.
            publish_manifest(directory, connection)
            connection.execute("BEGIN")
            writer = ShardWriter(directory, config, connection)
            for path in paths:
                remaining = (
                    None if config.max_games is None else config.max_games - stats["encountered"]
                )
                if remaining == 0:
                    break
                signature = (path.stat().st_size, path.stat().st_mtime_ns)
                event("source_started", source=str(path), compressed_bytes=signature[0])
                source_id = digest_file(path)
                connection.execute(
                    "INSERT OR IGNORE INTO sources VALUES (?,?,?)",
                    (source_id, str(path), signature[0]),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO source_aliases VALUES (?,?,?)",
                    (source_id, str(path), config.source),
                )
                games = safe_games(path, config)
                try:
                    for ordinal, record, error in islice(games, remaining):
                        if ordinal == -1:
                            stats["source_errors"] += 1
                            detail = {"reason": error, "remainder_processed": False}
                            connection.execute(
                                "INSERT OR REPLACE INTO occurrences "
                                "VALUES (?, -1, NULL, 'source_error', ?)",
                                (source_id, json.dumps(detail)),
                            )
                            event("source_aborted", source=str(path), **detail)
                            continue
                        stats["encountered"] += 1
                        if connection.execute(
                            "SELECT 1 FROM occurrences WHERE source_id=? AND ordinal=?",
                            (source_id, ordinal),
                        ).fetchone():
                            stats["already_processed"] += 1
                            if stats["encountered"] % config.progress_every == 0:
                                event("progress", **stats)
                            continue
                        status = "accepted"
                        detail = {"source": config.source}
                        if record is None:
                            status = "invalid"
                            detail["reason"] = error
                            stats["invalid"] += 1
                            event("invalid_game", source=str(path), ordinal=ordinal, reason=error)
                        else:
                            detail.update(headers=record.headers, player_color=record.player_color)
                            reason = filter_reason(record, config)
                            if reason is not None:
                                stats["filtered"] += 1
                                detail["reason"] = reason
                                connection.execute(
                                    "INSERT INTO occurrences VALUES (?,?,?,?,?)",
                                    (
                                        source_id,
                                        ordinal,
                                        record.game_id,
                                        "filtered",
                                        json.dumps(detail),
                                    ),
                                )
                                event(
                                    "filtered_game",
                                    source=str(path),
                                    ordinal=ordinal,
                                    reason=reason,
                                )
                                if stats["encountered"] % config.progress_every == 0:
                                    event("progress", **stats)
                                continue
                            existing = connection.execute(
                                "SELECT result FROM games WHERE id=?", (record.game_id,)
                            ).fetchone()
                            if existing:
                                status = "duplicate" if existing[0] == record.result else "conflict"
                                stats["duplicates" if status == "duplicate" else "invalid"] += 1
                                if status == "conflict":
                                    detail["reason"] = (
                                        "Duplicate trajectory has conflicting outcome"
                                    )
                                    event(
                                        "conflicting_outcome",
                                        source=str(path),
                                        ordinal=ordinal,
                                        game_id=record.game_id,
                                    )
                            else:
                                split = split_for(record.game_id, config)
                                plies = selected_plies(record, config)
                                connection.execute(
                                    "INSERT INTO games VALUES (?,?,?,?,?,?)",
                                    (
                                        record.game_id,
                                        split,
                                        record.result,
                                        json.dumps(asdict(record)),
                                        json.dumps(plies),
                                        len(plies),
                                    ),
                                )
                                for sample in iter_samples(record, plies, split):
                                    stats["encoding_seconds"] += sample["encoding_seconds"]
                                    writer.add(sample)
                                    stats["positions"] += 1
                                stats["games"] += 1
                        connection.execute(
                            "INSERT INTO occurrences VALUES (?,?,?,?,?)",
                            (
                                source_id,
                                ordinal,
                                record.game_id if record else None,
                                status,
                                json.dumps(detail),
                            ),
                        )
                        if stats["encountered"] % config.progress_every == 0:
                            event("progress", **stats)
                finally:
                    games.close()
                if signature != (path.stat().st_size, path.stat().st_mtime_ns):
                    raise ValueError("Source changed during processing; invocation rolled back")
            writer.finish()
            connection.commit()
            manifest = publish_manifest(directory, connection)
            stats["shard_write_seconds"] = writer.write_seconds
            stats["total_games"] = sum(count["games"] for count in manifest["counts"].values())
            stats["total_positions"] = sum(
                count["positions"] for count in manifest["counts"].values()
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    stats["seconds"] = perf_counter() - started
    event("processing_finished", **stats)
    return stats
