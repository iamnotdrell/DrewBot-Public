"""Phase 4 independent numerical references, recovery and real-shard integration."""

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from drewbot.config import DataConfig
from drewbot.data import pipeline
from drewbot.data.walkthrough import policy_target
from drewbot.evaluation.evaluator import evaluate
from drewbot.evaluation.reports import compare_runs
from drewbot.models import DrewBot, ModelSettings
from drewbot.models.model import ModelOutput
from drewbot.training.checkpoints import atomic_save, load_model, read_checkpoint
from drewbot.training.config import TrainingConfig
from drewbot.training.data import EpochSampler, audit_dataset, make_loader
from drewbot.training.losses import compute_losses
from drewbot.training.metrics import Metrics
from drewbot.training.optim import WarmupCosine, checked_gradient_norm, make_optimizer
from drewbot.training.trainer import Trainer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(gab=True, dropout=0):
    return ModelSettings(
        embedding_dim=8,
        attention_heads=2,
        feedforward_dim=12,
        gab_d1=2,
        gab_d2=3,
        gab_templates=2,
        value_hidden_dim=8,
        gab_enabled=gab,
        dropout=dropout,
    )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Explicit synthetic legal trajectories; no external data or fabricated engine targets."""
    generator = random.Random(19)
    pgn = tmp_path / "synthetic.pgn"
    texts = []
    for index in range(30):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        for _ in range(6):
            move = generator.choice(list(board.legal_moves))
            node = node.add_variation(move)
            board.push(move)
        game.headers["Result"] = ["1-0", "0-1", "1/2-1/2", "*"][index % 4]
        texts.append(str(game))
    pgn.write_text("\n\n".join(texts))
    monkeypatch.setattr(pipeline, "project_path", lambda value: tmp_path / value)
    pipeline.process(
        [pgn],
        DataConfig(
            output="dataset",
            split=(0.6, 0.2, 0.2),
            shard_positions=16,
            max_positions_per_game=6,
            source="synthetic-unit-fixture",
        ),
    )
    return tmp_path


def config(**kwargs):
    return replace(
        TrainingConfig(
            dataset="dataset", batch_size=3, epochs=2, max_steps=4, tensorboard=True, device="cpu"
        ),
        **kwargs,
    )


def labels():
    mask = torch.zeros(3, 4352, dtype=torch.bool)
    mask[0, [2, 5, 10]] = True
    mask[1, [4, 8]] = True
    return {
        "policy": torch.tensor([5, 8, -100]),
        "value": torch.tensor([2, -100, 0]),
        "value_valid": torch.tensor([True, False, True]),
        "legal_mask": mask,
    }


@pytest.mark.parametrize(
    "changes",
    [
        dict(batch_size=0),
        dict(workers=-1),
        dict(seed=True),
        dict(learning_rate=float("nan")),
        dict(objective="oops"),
        dict(min_lr_ratio=2),
        dict(policy_weight=0),
        dict(device="cuda"),
        dict(tensorboard=1),
        dict(epochs=0),
        dict(gradient_clip=0),
        dict(dataset="../escape"),
        dict(schedule="magic"),
    ],
)
def test_configuration_validation(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_losses_against_independent_numpy_and_gradients():
    batch = labels()
    p = torch.zeros(3, 4352, requires_grad=True)
    v = torch.tensor([[0.2, 0.1, 1.0], [1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], requires_grad=True)
    losses = compute_losses(ModelOutput(p, v), batch, config())
    expected_p = math.log(3) + math.log(2)
    expected_v = sum(
        np.log(np.exp(row).sum()) - row[target]
        for row, target in [(np.array([0.2, 0.1, 1.0]), 2), (np.array([0.0, 1.0, 2.0]), 0)]
    )
    assert float(losses.policy_sum.detach()) == pytest.approx(expected_p)
    assert float(losses.value_sum.detach()) == pytest.approx(expected_v)
    losses.weighted(config(policy_weight=2, value_weight=0.5)).backward()
    assert p.grad[~batch["legal_mask"]].count_nonzero() == 0
    assert v.grad[1].count_nonzero() == 0
    assert p.grad[0, 5] < 0 and v.grad[0, 2] < 0
    assert torch.isfinite(p.grad).all() and torch.isfinite(v.grad).all()


@pytest.mark.parametrize("objective", ["policy", "value", "joint"])
def test_head_and_missing_gradients(objective):
    p, v = torch.zeros(3, 4352, requires_grad=True), torch.zeros(3, 3, requires_grad=True)
    batch = labels()
    c = config(objective=objective)
    losses = compute_losses(ModelOutput(p, v), batch, c)
    losses.weighted(c).backward()
    assert (p.grad is not None) == (objective != "value")
    assert (v.grad is not None) == (objective != "policy")
    batch["policy"].fill_(-100)
    batch["value"].fill_(-100)
    batch["value_valid"].fill_(False)
    missing = compute_losses(ModelOutput(p, v), batch, c)
    assert missing.weighted(c) is None


@pytest.mark.parametrize("fault", ["illegal", "out_of_range", "wrong_valid", "nan", "dtype"])
def test_bad_labels_and_numerical_failures(fault):
    batch = labels()
    p, v = torch.zeros(3, 4352), torch.zeros(3, 3)
    if fault == "illegal":
        batch["policy"][0] = 3
    if fault == "out_of_range":
        batch["value"][0] = 3
    if fault == "wrong_valid":
        batch["value_valid"][1] = True
    if fault == "nan":
        p[0, 0] = float("nan")
    if fault == "dtype":
        batch["legal_mask"] = batch["legal_mask"].float()
    with pytest.raises((ValueError, FloatingPointError)):
        compute_losses(ModelOutput(p, v), batch, config())


def test_extreme_logits_and_promotions():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    indices = [policy_target(board, chess.Move.from_uci("a7a8" + piece))[0] for piece in "qrbn"]
    assert len(set(indices)) == 4
    batch = labels()
    batch["legal_mask"][0].zero_()
    batch["legal_mask"][0, indices] = True
    batch["policy"][0] = indices[-1]
    p = torch.full((3, 4352), 1e30, requires_grad=True)
    with torch.no_grad():
        p[batch["legal_mask"]] = -1e30
    losses = compute_losses(ModelOutput(p, torch.zeros(3, 3)), batch, config())
    assert float(losses.policy_sum.detach()) == pytest.approx(math.log(4) + math.log(2))
    losses.weighted(config()).backward()
    assert torch.isfinite(p.grad).all()


def test_metrics_known_confusion_and_legal_topk():
    batch = labels()
    p = torch.zeros(3, 4352)
    p[0, 5] = p[1, 4] = 4
    v = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    output = ModelOutput(p, v)
    metrics = Metrics()
    metrics.update(output, batch, compute_losses(output, batch, config()))
    result = metrics.result(config())
    assert result["policy_top1"] == 0.5 and result["policy_top5"] == 1
    assert result["legal_move_accuracy"] == 1
    assert result["value_accuracy"] == 0.5
    assert result["value_confusion_true_rows"] == [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
    assert result["value_recall"] == [0.0, 0.0, 1.0]


def test_optimizer_scheduler_and_nonfinite_gradients():
    model = DrewBot(tiny())
    c = config(warmup_steps=2, min_lr_ratio=0.1)
    optimizer = make_optimizer(model, c)
    assert isinstance(optimizer, torch.optim.AdamW)
    scheduler = WarmupCosine(optimizer, c, 6)
    ratios = []
    for _ in range(6):
        ratios.append(optimizer.param_groups[0]["lr"] / c.learning_rate)
        scheduler.step()
    assert ratios[:3] == [0.5, 1.0, 1.0]
    assert ratios[-1] == pytest.approx(0.1)
    copy = WarmupCosine(optimizer, c, 6)
    copy.load_state_dict(scheduler.state_dict())
    assert copy.completed == 6
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, float("inf"))
    before = parameter.detach().clone()
    with pytest.raises(FloatingPointError):
        checked_gradient_norm(model, 1)
    assert torch.equal(before, parameter)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 15, 16, 17, 127, 1000])
def test_lazy_sampler_is_permutation_and_restartable(size):
    a = list(EpochSampler(size, 42, 2))
    assert sorted(a) == list(range(size))
    assert a[3:] == list(EpochSampler(size, 42, 2, 3))
    assert a == list(EpochSampler(size, 42, 2))
    if size > 3:
        assert a != list(EpochSampler(size, 43, 2))


@pytest.mark.parametrize("gab", [False, True])
@pytest.mark.parametrize("objective", ["policy", "value", "joint"])
def test_end_to_end_modes_logging_and_inference(corpus, gab, objective):
    c = config(objective=objective, max_steps=2)
    trainer = Trainer(c, tiny(gab), root=corpus)
    result = trainer.fit()
    assert result["global_step"] == 2 and result["validation"]["loss"] > 0
    state = read_checkpoint(Path(result["checkpoint"]))
    assert state["dataset_version"] == "drewbot-shards-v1"
    assert state["architecture"]["config"]["layers"] == 6
    model, _ = load_model(Path(result["checkpoint"]))
    loader = make_loader(c, "validation", root=corpus)
    batch = next(iter(loader))
    trainer.model.eval()
    with torch.no_grad():
        a, b = trainer.model.forward_batch(batch), model.forward_batch(batch)
    torch.testing.assert_close(a.policy_logits, b.policy_logits, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    directory = corpus / "experiments" / result["run_id"]
    assert list((directory / "tensorboard").glob("events.out.tfevents.*"))
    records = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert any(r["event"] == "progress" and r["phase"] == "train_step" for r in records)
    assert (directory / "metrics.csv").stat().st_size > 100
    loader.dataset.close()


def test_exact_resume_with_dropout_and_accumulation(corpus):
    c = config(accumulation_steps=2)
    mc = tiny(dropout=0.15)
    whole = Trainer(c, mc, root=corpus)
    expected = whole.fit()
    interrupted = Trainer(c, mc, root=corpus).fit(stop_after_updates=2)
    resumed = Trainer(c, mc, resume=Path(interrupted["checkpoint"]), root=corpus)
    actual = resumed.fit()
    for key, tensor in whole.model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[key], rtol=0, atol=0)
    assert expected["validation"] == {
        **actual["validation"],
        "duration_seconds": expected["validation"]["duration_seconds"],
        "positions_per_second": expected["validation"]["positions_per_second"],
    }
    assert resumed.scheduler.state_dict() == whole.scheduler.state_dict()
    assert actual["global_step"] == 4
    state = read_checkpoint(Path(actual["checkpoint"]))
    assert state["progress"]["offset"] == 24


def test_resume_rejects_changed_budget_and_dataset(corpus):
    c = config()
    result = Trainer(c, tiny(), root=corpus).fit(stop_after_updates=1)
    with pytest.raises(ValueError, match="differs"):
        Trainer(replace(c, max_steps=8), tiny(), resume=Path(result["checkpoint"]), root=corpus)
    path = corpus / "dataset" / "manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        Trainer(c, tiny(), resume=Path(result["checkpoint"]), root=corpus)


def test_accumulation_matches_large_batch_with_missing_targets(corpus):
    c = config(accumulation_steps=2, tensorboard=False)
    a = Trainer(c, tiny(), root=corpus)
    b = Trainer(replace(c, batch_size=6, accumulation_steps=1), tiny(), root=corpus)
    loader = make_loader(c, "train", root=corpus)
    iterator = iter(loader)
    batches = [next(iterator), next(iterator)]
    batches[0]["value"][0] = -100
    batches[0]["value_valid"][0] = False
    combined = {
        k: torch.cat([x[k] for x in batches])
        if isinstance(batches[0][k], torch.Tensor)
        else sum([list(x[k]) for x in batches], [])
        for k in batches[0]
    }
    try:
        # Compare derivatives directly: Adam's division by tiny second moments can
        # amplify harmless near-zero batched-kernel rounding after the first update.
        with patch.object(a.optimizer, "step"), patch.object(b.optimizer, "step"):
            a.train_window(batches)
            b.train_window([combined])
        for x, y in zip(a.model.parameters(), b.model.parameters(), strict=True):
            if x.grad is None:
                assert y.grad is None
            else:
                torch.testing.assert_close(x.grad, y.grad, rtol=2e-4, atol=2e-6)
        assert a.model.policy_head.source.weight.grad.abs().sum() > 0
        assert a.model.value_head.network[-1].weight.grad.abs().sum() > 0
        assert a.model.templates.grad.abs().sum() > 0
    finally:
        a.log.close()
        b.log.close()
        a.restore_runtime()
        b.restore_runtime()
        loader.dataset.close()


def test_atomic_failure_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    atomic_save(path, {"number": 1})
    with patch("drewbot.training.checkpoints.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            atomic_save(path, {"number": 2})
    assert torch.load(path, weights_only=True)["number"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_audit_and_no_test_access_during_fit(corpus):
    info = audit_dataset(corpus / "dataset")
    assert info["validation"]["valid"]
    assert len(set(info["split_identifiers"].values())) == 3
    real_loader = make_loader
    seen = []

    def loader(config, split, **kwargs):
        seen.append(split)
        assert split != "test"
        return real_loader(config, split, **kwargs)

    with patch("drewbot.training.trainer.make_loader", loader):
        Trainer(config(max_steps=1), tiny(), root=corpus).fit()
    assert "validation" in seen


def test_validation_preserves_parameters_and_mode(corpus):
    trainer = Trainer(config(), tiny(), root=corpus)
    loader = make_loader(config(), "validation", root=corpus)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    try:
        trainer.model.train()
        result = evaluate(trainer.model, loader, config())
        assert trainer.model.training and result["positions"] == len(loader.dataset)
        for key, value in before.items():
            assert torch.equal(value, trainer.model.state_dict()[key])
    finally:
        trainer.log.close()
        trainer.restore_runtime()
        loader.dataset.close()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Operational MPS unavailable")
@pytest.mark.parametrize("gab", [False, True])
def test_mps_actual_training_and_placement(corpus, gab):
    c = config(device="mps", allow_cpu_fallback=False, max_steps=1)
    trainer = Trainer(c, tiny(gab), root=corpus)
    result = trainer.fit()
    assert result["device"] == "mps"
    assert all(p.device.type == "mps" for p in trainer.model.parameters())
    assert math.isfinite(result["validation"]["loss"])


def test_comparison_matches_and_rejects_unmatched(corpus):
    runs = []
    for gab in (False, True):
        result = Trainer(config(max_steps=1), tiny(gab), root=corpus).fit()
        runs.append(corpus / "experiments" / result["run_id"])
    report = compare_runs(runs, corpus / "comparison")
    assert report["runs"][0]["parameters"] != report["runs"][1]["parameters"]
    summary = runs[1] / "summary.json"
    raw = json.loads(summary.read_text())
    raw["training"]["learning_rate"] *= 2
    summary.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unmatched"):
        compare_runs(runs, corpus / "bad-compare")


def test_epoch_boundary_recovery_and_early_stopping(corpus):
    # Cross an epoch boundary, validating every epoch; interrupted state must not
    # skip the pending validation or repeat the already consumed positions.
    c = config(batch_size=128, max_steps=None, epochs=3, patience=1, warmup_steps=0)
    original = evaluate

    def fixed_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        result["loss"] = 1.0
        return result

    with patch("drewbot.training.trainer.evaluate", fixed_validation):
        whole = Trainer(c, tiny(), root=corpus)
        whole.fit()
        partial = Trainer(c, tiny(), root=corpus).fit(stop_after_updates=1)
        resumed = Trainer(c, tiny(), resume=Path(partial["checkpoint"]), root=corpus)
        result = resumed.fit()
    assert result["early_stopped"] and result["epoch"] == 2
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)


def test_annotation_example_executes(corpus):
    from drewbot.training.example import demonstrate

    result = demonstrate(config(), tiny(), root=corpus)
    assert len(result["trace"]) == 12
    assert result["trace"][-1]["exact_output_match"]
    assert all(all(key in row for key in ("WHAT", "HOW", "WHY")) for row in result["trace"])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_entropy_with_zero_illegal_and_underflow_probabilities(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    batch = {k: v.to(device) for k, v in labels().items()}
    p = torch.full((3, 4352), -1e20, device=device)
    p[0, 5] = p[1, 8] = 1e20
    output = ModelOutput(p, torch.zeros(3, 3, device=device))
    metrics = Metrics()
    metrics.update(output, batch, compute_losses(output, batch, config()))
    result = metrics.result(config())
    assert result["policy_entropy_nats"] == 0.0
    assert result["policy_confidence"] == 1.0


def test_value_only_skips_unlabelled_window_without_update(corpus):
    trainer = Trainer(config(objective="value"), tiny(), root=corpus)
    loader = make_loader(config(), "train", root=corpus)
    batch = next(iter(loader))
    batch["value"].fill_(-100)
    batch["value_valid"].fill_(False)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    try:
        result = trainer.train_window([batch])
        assert result["skipped"] and trainer.global_step == 0
        assert trainer.scheduler.completed == 0 and trainer.offset == len(batch["value"])
        assert not trainer.optimizer.state
        for k, v in before.items():
            assert torch.equal(v, trainer.model.state_dict()[k])
    finally:
        trainer.log.close()
        trainer.restore_runtime()
        loader.dataset.close()
