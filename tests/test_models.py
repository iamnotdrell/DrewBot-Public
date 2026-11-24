"""Phase 3 numerical, contract, legal-state and real Dataset/device acceptance tests."""

import io
import json
import math
from dataclasses import asdict, replace

import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from drewbot.config import DataConfig
from drewbot.data import pipeline
from drewbot.data.dataset import ChessDataset
from drewbot.data.samples import auxiliary
from drewbot.data.validate import validate_dataset
from drewbot.data.walkthrough import encode_history, legal_mask, policy_target
from drewbot.models import DrewBot, ModelSettings, choose_move, latest_v1_history, legal_policy
from drewbot.models.attention import EncoderBlock, SquareAttention
from drewbot.models.gab import GeometricAttentionBias
from drewbot.models.heads import PolicyHead


def tiny(**changes):
    return replace(
        ModelSettings(
            embedding_dim=16,
            attention_heads=2,
            feedforward_dim=24,
            gab_d1=2,
            gab_d2=4,
            gab_templates=3,
            value_hidden_dim=12,
        ),
        **changes,
    )


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(29)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "changes",
    [
        {"layers": 0},
        {"history_frames": True},
        {"history_frames": 0},
        {"embedding_dim": None},
        {"attention_heads": 3},
        {"feedforward_dim": -1},
        {"gab_d1": 0},
        {"gab_d2": False},
        {"gab_templates": 0},
        {"dropout": 1},
        {"dropout": float("inf")},
        {"dropout": True},
        {"gab_enabled": 1},
        {"board_tokens": 63},
        {"auxiliary_features": 71},
        {"auxiliary_mode": "typo"},
        {"benchmark_batch_sizes": []},
        {"benchmark_batch_sizes": [True]},
        {"value_hidden_dim": 0},
        {"implemented": False},
    ],
)
def test_config_errors(changes):
    with pytest.raises(ValueError):
        tiny(**changes)


def test_config_copies_batch_sizes():
    sizes = [1, 3]
    config = tiny(benchmark_batch_sizes=sizes)
    sizes.append(4)
    assert config.benchmark_batch_sizes == (1, 3)
    assert json.loads(json.dumps(asdict(config)))["history_frames"] == 8


@pytest.mark.parametrize("gab", [False, True])
@pytest.mark.parametrize("batch,history", [(1, 1), (3, 4), (2, 8), (1, 10)])
def test_shapes_history_and_inspection(gab, batch, history):
    model = DrewBot(tiny(gab_enabled=gab, history_frames=history)).eval()
    x, a = torch.randn(batch, 64, 12 * history), torch.randn(batch, 72)
    result = model(x, a, inspect=True)
    assert result.policy_logits.shape == (batch, 4352)
    assert result.value_logits.shape == (batch, 3)
    assert len(result.inspections) == 6
    for layer in result.inspections:
        assert layer.probabilities.shape == (batch, 2, 64, 64)
        torch.testing.assert_close(layer.probabilities.sum(-1), torch.ones(batch, 2, 64))
        assert (layer.bias is not None) == gab
    assert model(x, a).inspections is None
    assert torch.equal(result.policy_logits, model(x, a).policy_logits)


@pytest.mark.parametrize(
    "which",
    [
        "tokens_shape",
        "aux_shape",
        "tokens_dtype",
        "aux_dtype",
        "nan",
        "inf",
        "empty",
        "device",
        "inspect",
        "model_dtype",
    ],
)
def test_input_errors(which):
    model = DrewBot(tiny())
    x, a = torch.zeros(1, 64, 96), torch.zeros(1, 72)
    inspect = False
    if which == "tokens_shape":
        x = x[:, :, :84]
    elif which == "aux_shape":
        a = a[:, :71]
    elif which == "tokens_dtype":
        x = x.to(torch.uint8)
    elif which == "aux_dtype":
        a = a.double()
    elif which == "nan":
        x[0, 0, 0] = torch.nan
    elif which == "inf":
        a[0, 0] = torch.inf
    elif which == "empty":
        x, a = x[:0], a[:0]
    elif which == "device":
        a = a.to("meta")
    elif which == "inspect":
        inspect = 1
    elif which == "model_dtype":
        model.double()
    with pytest.raises(ValueError):
        model(x, a, inspect=inspect)


def test_v1_adapter_selects_latest_frames_and_refuses_fabricated_history():
    x = torch.arange(96).float().repeat(2, 64, 1)
    assert torch.equal(latest_v1_history(x, 2), x[:, :, 72:])
    for frames in (0, 9, True):
        with pytest.raises(ValueError, match="1..8"):
            latest_v1_history(x, frames)
    with pytest.raises(ValueError, match="96"):
        latest_v1_history(x[:, :, :84], 1)
    model = DrewBot(tiny(history_frames=2))
    batch = {"tokens": x, "auxiliary": torch.zeros(2, 72)}
    torch.testing.assert_close(
        model.forward_batch(batch).policy_logits,
        model(x[:, :, -24:], batch["auxiliary"]).policy_logits,
    )
    with pytest.raises(ValueError, match="1..8"):
        DrewBot(tiny(history_frames=9)).forward_batch(batch)


def np_linear(x, layer):
    out = x @ layer.weight.detach().numpy().T
    return out if layer.bias is None else out + layer.bias.detach().numpy()


def np_norm(x, layer):
    # Independent population variance and affine normalization, not torch.layer_norm.
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(
        variance + layer.eps
    ) * layer.weight.detach().numpy() + layer.bias.detach().numpy()


def np_gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def test_gab_independent_numpy_reference_and_dynamic_bias():
    config = tiny(layers=1)
    gab = GeometricAttentionBias(config)
    x = torch.randn(2, 64, 16)
    templates = torch.randn(4096, 3)
    projected = np_linear(x.numpy(), gab.square_projection).reshape(2, -1)
    compressed = np_norm(np_gelu(np_linear(projected, gab.compression)), gab.compression_norm)
    coefficients = np_norm(np_gelu(np_linear(compressed, gab.generation)), gab.generation_norm)
    coefficients = coefficients.reshape(2, 2, 3)
    # Explicit per-head template mixture independently verifies head/flatten ordering.
    expected = np.empty((2, 2, 64, 64))
    for b in range(2):
        for h in range(2):
            for i in range(64):
                expected[b, h, i] = [
                    sum(coefficients[b, h, t] * templates[64 * i + j, t].item() for t in range(3))
                    for j in range(64)
                ]
    torch.testing.assert_close(
        gab(x, templates), torch.tensor(expected).float(), atol=3e-6, rtol=2e-5
    )
    changed = x.clone()
    changed[:, 5, :] += 2
    assert not torch.allclose(gab(x, templates), gab(changed, templates))


@pytest.mark.parametrize("gab", [False, True])
def test_attention_independent_numpy_before_softmax_reference(gab):
    config = tiny(layers=1, gab_enabled=gab)
    attention = SquareAttention(config).eval()
    x = torch.randn(2, 64, 16)
    templates = torch.randn(4096, 3) * 0.1 if gab else None
    actual, details = attention(x, templates, inspect=True)
    qkv = np_linear(x.numpy(), attention.qkv).reshape(2, 64, 3, 2, 8)
    expected = np.zeros((2, 64, 16), dtype=np.float32)
    for b in range(2):
        for h in range(2):
            q, k, v = (qkv[b, :, component, h] for component in range(3))
            logits = q @ k.T / math.sqrt(8)
            if gab:
                logits += details.bias[b, h].detach().numpy()
            weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(
                details.probabilities[b, h].detach().numpy(), weights, rtol=2e-5, atol=1e-6
            )
            expected[b, :, h * 8 : (h + 1) * 8] = weights @ v
    expected = np_linear(expected, attention.output)
    torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-6, rtol=2e-5)


def test_zero_template_equivalence_and_torch_attention_reference():
    biased = SquareAttention(tiny()).eval()
    standard = SquareAttention(tiny(gab_enabled=False)).eval()
    standard.qkv.load_state_dict(biased.qkv.state_dict())
    standard.output.load_state_dict(biased.output.state_dict())
    x = torch.randn(2, 64, 16)
    zeros = torch.zeros(4096, 3)
    gab_result, gab_details = biased(x, zeros, inspect=True)
    standard_result, _ = standard(x)
    torch.testing.assert_close(gab_result, standard_result, atol=0, rtol=0)
    assert torch.count_nonzero(gab_details.bias) == 0
    reference = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(standard.qkv.weight)
        reference.in_proj_bias.copy_(standard.qkv.bias)
        reference.out_proj.load_state_dict(standard.output.state_dict())
    expected, _ = reference(x, x, x, need_weights=False)
    torch.testing.assert_close(standard_result, expected, atol=1e-6, rtol=1e-5)


def test_block_post_norm_order():
    block = EncoderBlock(tiny(gab_enabled=False)).eval()
    x = torch.randn(1, 64, 16)
    attended, _ = block.attention(x)
    first = block.attention_norm(x + attended)
    expected = block.feedforward_norm(first + block.feedforward(first))
    torch.testing.assert_close(block(x, None)[0], expected, rtol=0, atol=0)


def test_templates_registered_once_and_shared_gradient_accumulates():
    model = DrewBot(tiny(layers=2))
    assert [name for name in model.state_dict() if name.endswith("templates")] == ["templates"]
    assert sum(p is model.templates for p in model.parameters()) == 1
    assert model.layers[0].attention.gab is not model.layers[1].attention.gab
    seen = []
    handles = [
        layer.attention.gab.register_forward_pre_hook(lambda _, args: seen.append(args[1]))
        for layer in model.layers
    ]
    output = model(torch.randn(1, 64, 96), torch.randn(1, 72))
    output.policy_logits.square().mean().backward()
    for handle in handles:
        handle.remove()
    assert len(seen) == 2 and all(t is model.templates for t in seen)
    assert model.templates.grad.abs().sum() > 0
    # Independent per-generator gradient sums verify true shared accumulation.
    x = torch.randn(1, 64, 16)
    terms = [layer.attention.gab(x, model.templates).square().mean() for layer in model.layers]
    separate = [torch.autograd.grad(term, model.templates, retain_graph=True)[0] for term in terms]
    joint = torch.autograd.grad(sum(terms), model.templates)[0]
    torch.testing.assert_close(joint, sum(separate))


@pytest.mark.parametrize("gab", [False, True])
@pytest.mark.parametrize("objective", ["policy", "value", "joint"])
def test_finite_gradients_independent_heads(gab, objective):
    model = DrewBot(tiny(gab_enabled=gab))
    x = torch.randn(2, 64, 96, requires_grad=True)
    a = torch.randn(2, 72, requires_grad=True)
    output = model(x, a)
    assert torch.isfinite(output.policy_logits).all() and torch.isfinite(output.value_logits).all()
    losses = []
    if objective in {"policy", "joint"}:
        losses.append(
            torch.nn.functional.cross_entropy(output.policy_logits, torch.tensor([1, 4351]))
        )
    if objective in {"value", "joint"}:
        losses.append(torch.nn.functional.cross_entropy(output.value_logits, torch.tensor([0, 2])))
    sum(losses).backward()
    for name, parameter in model.named_parameters():
        unused = (name.startswith("policy_head") and objective == "value") or (
            name.startswith("value_head") and objective == "policy"
        )
        if unused:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for gradient in (
        x.grad,
        a.grad,
        model.input_projection.projection.weight.grad,
        model.layers[0].attention.qkv.weight.grad,
    ):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if gab:
        for layer in model.layers:
            for module in (
                layer.attention.gab.square_projection,
                layer.attention.gab.compression,
                layer.attention.gab.generation,
            ):
                assert module.weight.grad.abs().sum() > 0
        assert model.templates.grad.abs().sum() > 0


@pytest.mark.parametrize("gab", [False, True])
def test_serialization_deterministic_eval_and_large_finite_inputs(gab):
    config = tiny(gab_enabled=gab, dropout=0.2)
    torch.manual_seed(111)
    first = DrewBot(config).eval()
    torch.manual_seed(111)
    second = DrewBot(config).eval()
    x, a = torch.randn(2, 64, 96) * 1e4, torch.randn(2, 72) * 1e4
    expected = first(x, a)
    torch.testing.assert_close(expected.policy_logits, second(x, a).policy_logits, atol=0, rtol=0)
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    loaded = DrewBot(config).eval()
    loaded.load_state_dict(torch.load(buffer, weights_only=True))
    restored = loaded(x, a)
    assert torch.equal(expected.policy_logits, restored.policy_logits)
    assert torch.equal(expected.value_logits, restored.value_logits)
    assert (
        torch.isfinite(restored.policy_logits).all() and torch.isfinite(restored.value_logits).all()
    )


def test_policy_head_promotion_reference_all_entries():
    head = PolicyHead(8)
    x = torch.randn(2, 64, 8)
    logits = head(x)
    source, destination = head.source(x), head.destination(x)
    for f in range(8):
        for t in range(8):
            base = sum(
                source[:, 48 + f, d] * destination[:, 56 + t, d] for d in range(8)
            ) / math.sqrt(8)
            for p in range(4):
                bias = sum(
                    destination[:, 56 + t, d] * head.promotion.weight[p, d] for d in range(8)
                ) * math.sqrt(8)
                index = 4096 + 4 * (8 * f + t) + p
                torch.testing.assert_close(logits[:, index], base + bias)
    assert not torch.equal(logits[:, 4096], logits[:, 4097])


@pytest.mark.parametrize(
    "fen,moves",
    [
        (chess.STARTING_FEN, ["e2e4"]),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", ["e7e5"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e1c1"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ["e8g8", "e8c8"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", ["e4d3"]),
        (
            "k6r/6P1/8/8/8/8/8/K7 w - - 0 1",
            ["g7g8" + p for p in "qrbn"] + ["g7h8" + p for p in "qrbn"],
        ),
        (
            "k7/8/8/8/8/8/6p1/K6R b - - 0 1",
            ["g2g1" + p for p in "qrbn"] + ["g2h1" + p for p in "qrbn"],
        ),
    ],
)
def test_legal_inference_coordinates_special_moves(fen, moves):
    board = chess.Board(fen)
    mask = torch.from_numpy(legal_mask(board)).unsqueeze(0)
    for uci in moves:
        move = chess.Move.from_uci(uci)
        index, _ = policy_target(board, move)
        logits = torch.full((1, 4352), 1e6)  # Give every illegal move an overwhelming raw score.
        logits[mask] = -1e6
        logits[0, index] = -1e6 + 10
        probabilities, indices = legal_policy(logits, mask)
        assert indices.item() == index and probabilities[~mask].sum() == 0
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1))
        assert choose_move(logits[0], board) == move


@pytest.mark.parametrize(
    "fen",
    [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # mate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
        "8/8/8/8/8/8/4k3/6K1 w - - 0 1",  # automatic insufficient material
    ],
)
def test_terminal_no_action(fen):
    board = chess.Board(fen)
    assert board.is_game_over()
    assert choose_move(torch.randn(4352), board) is None


def test_mixed_terminal_rows_and_mask_errors():
    logits = torch.randn(2, 4352)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 12] = True
    probabilities, indices = legal_policy(logits, mask)
    assert indices.tolist() == [12, -1]
    assert probabilities[0, 12] == 1 and probabilities[1].sum() == 0
    assert torch.isfinite(probabilities).all()
    for bad_logits, bad_mask in (
        (logits, mask.float()),
        (logits[:, :64], mask),
        (logits.double(), mask),
        (logits, mask[:1]),
    ):
        with pytest.raises(ValueError):
            legal_policy(bad_logits, bad_mask)
    logits[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        legal_policy(logits, mask)


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ],
)
def test_same_pieces_different_rights_condition_representations(fen):
    first = chess.Board(fen)
    second = first.copy()
    second.castling_rights = 0
    second.ep_square = None
    x = torch.from_numpy(encode_history([first])).float().unsqueeze(0)
    assert np.array_equal(encode_history([first]), encode_history([second]))
    a, b = (torch.from_numpy(auxiliary(board, 1)).unsqueeze(0) for board in (first, second))
    assert not torch.equal(a, b)
    for gab in (False, True):
        model = DrewBot(tiny(gab_enabled=gab)).eval()
        assert not torch.allclose(model.input_projection(x, a), model.input_projection(x, b))
        assert not torch.allclose(model(x, a).policy_logits, model(x, b).policy_logits)
        ablated = DrewBot(tiny(gab_enabled=gab, auxiliary_mode="none")).eval()
        assert torch.equal(ablated(x, a).policy_logits, ablated(x, b).policy_logits)


@pytest.mark.parametrize("gab", [False, True])
def test_real_dataset_batch_preserves_missing_labels(tmp_path, monkeypatch, gab):
    monkeypatch.setattr(pipeline, "project_path", lambda name: tmp_path / name)
    pgn = tmp_path / "synthetic.pgn"
    pgn.write_text('[Event "Synthetic mechanics"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    pipeline.process([pgn], DataConfig(output="dataset"))
    assert validate_dataset(tmp_path / "dataset")["positions"] == 4
    dataset = ChessDataset(tmp_path / "dataset")
    try:
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        assert (batch["value"] == -100).all() and not batch["value_valid"].any()
        model = DrewBot(tiny(gab_enabled=gab)).eval()
        result = model.forward_batch(batch)
        probabilities, _ = legal_policy(result.policy_logits, batch["legal_mask"])
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(4))
        assert probabilities[~batch["legal_mask"]].sum() == 0
        changed = {
            **batch,
            "value": torch.ones(4),
            "value_valid": torch.ones(4),
            "policy": torch.zeros(4),
            "metadata": "do not consume",
            "legal_mask": None,
        }
        assert torch.equal(result.value_logits, model.forward_batch(changed).value_logits)
        assert torch.equal(result.policy_logits, model.forward_batch(changed).policy_logits)
    finally:
        dataset.close()


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("gab", [False, True])
def test_actual_device_forward_backward_and_cpu_agreement(device, gab):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this execution environment; CPU test runs independently")
    config = tiny(gab_enabled=gab)
    cpu_model = DrewBot(config).eval()
    model = DrewBot(config).to(device).eval()
    model.load_state_dict(cpu_model.state_dict())
    x, a = torch.randn(2, 64, 96), torch.randn(2, 72)
    expected = cpu_model(x, a)
    result = model(x.to(device), a.to(device), inspect=True)
    (result.policy_logits.square().mean() + result.value_logits.square().mean()).backward()
    if device == "mps":
        torch.mps.synchronize()
    torch.testing.assert_close(
        result.policy_logits.cpu(), expected.policy_logits, atol=2e-4, rtol=2e-4
    )
    torch.testing.assert_close(
        result.value_logits.cpu(), expected.value_logits, atol=2e-4, rtol=2e-4
    )
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(detail.probabilities.device.type == device for detail in result.inspections)


def test_default_six_layer_model_runs():
    model = DrewBot().eval()
    assert len(model.layers) == 6 and model.config.embedding_dim == 128
    result = model(torch.zeros(1, 64, 96), torch.zeros(1, 72))
    assert result.policy_logits.shape == (1, 4352)
    assert torch.isfinite(result.value_logits).all()


def test_benchmark_exports_exact_config_and_measured_rows(tmp_path):
    import csv

    from drewbot.models.benchmark import run_benchmarks

    config = tiny(layers=1, benchmark_batch_sizes=(1, 2))
    result = run_benchmarks(
        "bench", config, root=tmp_path, devices=("cpu",), warmup=1, iterations=1
    )
    saved = json.loads((tmp_path / "bench/metrics.json").read_text())
    with (tmp_path / "bench/metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result["records"]) == 16
    assert saved["device_status"] == [{"device": "cpu", "status": "executed", "reason": None}]
    for row, record in zip(rows, saved["records"], strict=True):
        assert json.loads(row["configuration"]) == record["configuration"]
        assert json.loads(row["hardware"]) == saved["hardware"]
        assert float(row["latency_seconds"]) == record["latency_seconds"] > 0
        assert record["positions_per_second"] == record["batch_size"] / record["latency_seconds"]
        assert record["parameter_count"] > 0 and record["peak_allocated_mps_bytes"] is None
        assert len(record["wall_samples_seconds"]) == 1
    with pytest.raises(FileExistsError):
        run_benchmarks("bench", config, root=tmp_path, devices=("cpu",), warmup=1, iterations=1)


def test_gab_gradient_matches_central_finite_difference():
    gab = GeometricAttentionBias(tiny(layers=1)).double()
    x = torch.randn(1, 64, 16, dtype=torch.float64, requires_grad=True)
    template = torch.randn(4096, 3, dtype=torch.float64, requires_grad=True)
    loss = gab(x, template).square().mean()
    input_gradient, template_gradient = torch.autograd.grad(loss, (x, template))
    for tensor, location, gradient in (
        (x, (0, 7, 3), input_gradient),
        (template, (17, 1), template_gradient),
    ):
        epsilon = 1e-6
        with torch.no_grad():
            original = tensor[location].item()
            tensor[location] = original + epsilon
            plus = gab(x, template).square().mean().item()
            tensor[location] = original - epsilon
            minus = gab(x, template).square().mean().item()
            tensor[location] = original
        numerical = (plus - minus) / (2 * epsilon)
        assert abs(numerical) > 1e-8
        assert gradient[location].item() == pytest.approx(numerical, rel=1e-5, abs=1e-8)


def test_whole_encoder_zero_gab_matches_standard_with_zero_absolute_embedding():
    gab_model = DrewBot(tiny()).eval()
    standard = DrewBot(tiny(gab_enabled=False)).eval()
    common = {
        name: value
        for name, value in gab_model.state_dict().items()
        if name in standard.state_dict()
    }
    standard.load_state_dict(common, strict=False)
    with torch.no_grad():
        gab_model.templates.zero_()
        standard.square_embedding.zero_()
    x, a = torch.randn(2, 64, 96), torch.randn(2, 72)
    biased, plain = gab_model(x, a), standard(x, a)
    torch.testing.assert_close(biased.policy_logits, plain.policy_logits, rtol=0, atol=0)
    torch.testing.assert_close(biased.value_logits, plain.value_logits, rtol=0, atol=0)
