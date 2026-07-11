from __future__ import annotations

import builtins
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _execute_pickle_marker(path: str, payload_kind: str) -> object:
    Path(path).write_text("pickle executed", encoding="utf-8")
    if payload_kind == "data":
        return {
            "train": [{"example_id": "e0", "input_ids": [0, 1, 2]}],
            "eval": [{"example_id": "eval0", "input_ids": [0, 1, 2]}],
        }
    return {}


class _PickleMarkerPayload:
    def __init__(self, path: Path, payload_kind: str) -> None:
        self.path = str(path)
        self.payload_kind = payload_kind

    def __reduce__(self):
        return _execute_pickle_marker, (self.path, self.payload_kind)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qwen_backend_import_never_imports_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("research.kmd2_ablation.qwen_backend", None)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("qwen_backend imported Transformers eagerly")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("research.kmd2_ablation.qwen_backend")
    assert hasattr(module, "load_qwen_arm")


def test_qwen_training_import_is_transformers_lazy_and_exposes_runner_entrypoint(
) -> None:
    import subprocess

    script = """
import sys
from importlib.abc import MetaPathFinder

class RejectTransformers(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] == 'transformers':
            raise AssertionError('qwen_training imported Transformers eagerly')
        return None

sys.meta_path.insert(0, RejectTransformers())
from research.kmd2_ablation import qwen_training
assert callable(qwen_training.run_job)
assert callable(qwen_training.build_job_dispatcher)
assert 'transformers' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.memory = torch.nn.Linear(2, 2, bias=False)


def _asset(name: str, path: Path):
    from research.kmd2_ablation.qwen_backend import ExternalAssetIdentity

    return ExternalAssetIdentity(
        name=name,
        path=path,
        kind="file",
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def test_qwen_arm_loader_validates_assets_orders_install_and_freezes_exact_names(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.json"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.jsonl"
    cache_resume = tmp_path / "resume.pt"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"examples")
    cache_resume.write_bytes(b"resume")

    events: list[object] = []
    model = _FakeQwen()

    def base_loader(path: Path, **kwargs: object) -> _FakeQwen:
        events.append(("base", path, kwargs))
        return model

    def manager_factory(received: object, config: object) -> object:
        assert received is model
        events.append(("manager", config))
        return SimpleNamespace(name="manager")

    def cache_installer(**kwargs: object) -> tuple[int, ...]:
        events.append(
            (
                "install",
                kwargs["native_checkpoint"],
                kwargs["cache_resume"],
                kwargs["expected_job_id"],
            )
        )
        assert kwargs["model"] is model
        assert getattr(kwargs["manager"], "name") == "manager"
        model.register_parameter(
            "cache_amplitude", torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        )
        return (1, 3)

    spec = QwenArmLoadSpec(
        arm="surprise",
        job_id="job-surprise",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=_asset("cache_resume", cache_resume),
        trainable_names=("memory.weight", "cache_amplitude"),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
        model_loader_kwargs={"torch_dtype": "bfloat16"},
    )
    loaded = load_qwen_arm(
        spec,
        model_config=SimpleNamespace(name="cfg"),
        cache_config=SimpleNamespace(score="exact_outer"),
        base_model_loader=base_loader,
        manager_factory=manager_factory,
        cache_installer=cache_installer,
    )

    assert [event[0] for event in events] == ["base", "manager", "install"]
    assert events[0][1] == model_asset.resolve()
    assert events[0][2] == {"torch_dtype": "bfloat16"}
    assert events[2][1:] == (
        native_checkpoint.resolve(),
        cache_resume.resolve(),
        "job-surprise",
    )
    assert loaded.model is model
    assert loaded.arm == "surprise"
    assert loaded.upgraded_indices == (1, 3)
    assert loaded.trainable_names == ("cache_amplitude", "memory.weight")
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == {
        "cache_amplitude": True,
        "backbone.bias": False,
        "backbone.weight": False,
        "memory.weight": True,
    }
    assert tuple(asset.name for asset in loaded.assets) == (
        "cache_resume",
        "data",
        "model",
        "native_checkpoint",
    )


def test_qwen_recency_arm_uses_real_default_install_and_runs_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = torch.nn.Linear(2, 2)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.model = Backbone()

    class Manager:
        def __init__(self, model: Model) -> None:
            self.model = model

        def apply_upgrade(self) -> list[int]:
            assert os.environ["GDN3_KMD2_NATIVE"] == "1"
            self.model.model.layers[0].linear_attn = KMD2NativeAttn(
                config,
                layer_idx=0,
            )
            return [0]

    model_asset = tmp_path / "model.bin"
    checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.jsonl"
    model_asset.write_bytes(b"model")
    torch.save({}, checkpoint)
    data_asset.write_bytes(b"examples")
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    model = Model()
    spec = QwenArmLoadSpec(
        arm="recency",
        job_id="job-recency-real-install",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=(
            "model.layers.0.linear_attn.cache_amplitude",
        ),
        pre_replacement_checkpoint_sha256=_sha256(checkpoint),
    )

    loaded = load_qwen_arm(
        spec,
        model_config=None,
        cache_config=CacheConfig(
            width=2,
            block_size=2,
            score="recency",
            read="rmsnorm",
            storage_dtype="fp32",
        ),
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=lambda received, _config: Manager(received),
    )

    layer = loaded.model.model.layers[0].linear_attn
    assert type(layer).__name__ == "KMD2RecencyCacheAttn"
    assert layer.cache_config.score == "recency"
    torch.manual_seed(1201)
    output = layer(torch.randn(2, 6, 12))
    assert output.shape == (2, 6, 12)
    assert bool(torch.isfinite(output).all())
    diagnostics = layer.last_cache_diagnostics
    assert diagnostics is not None
    torch.testing.assert_close(
        diagnostics.update_scores,
        torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1).expand(2, 6, 2),
    )
    torch.testing.assert_close(
        diagnostics.final_selected_positions,
        torch.tensor([5, 4], dtype=torch.int64).view(1, 1, 2).expand(2, 2, 2),
    )


@pytest.mark.parametrize("arm", ["native", "recency", "surprise"])
@pytest.mark.parametrize(
    ("checkpoint_dtype", "expect_success"),
    [(torch.bfloat16, True), (torch.float32, False)],
)
def test_qwen_bfloat16_install_aligns_inherited_dtype_and_enforces_checkpoint_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    checkpoint_dtype: torch.dtype,
    expect_success: bool,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = torch.nn.Linear(2, 2).to(torch.bfloat16)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(13, 12).to(torch.bfloat16)
            self.model = Backbone()

    class Manager:
        def __init__(self, model: Model) -> None:
            self.model = model

        def apply_upgrade(self) -> list[int]:
            self.model.model.layers[0].linear_attn = KMD2NativeAttn(
                config, layer_idx=0
            )
            return [0]

    model_asset = tmp_path / f"{arm}-model.bin"
    checkpoint = tmp_path / f"{arm}-{checkpoint_dtype}.pt"
    data_asset = tmp_path / f"{arm}-data.jsonl"
    model_asset.write_bytes(b"model")
    torch.save(
        {
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.full(
                (22, 12), 0.125, dtype=checkpoint_dtype
            )
        },
        checkpoint,
    )
    data_asset.write_bytes(b"examples")
    model = Model()
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    spec = QwenArmLoadSpec(
        arm=arm,
        job_id=f"job-{arm}-bf16",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=(
            f"model.layers.0.linear_attn.{('in_proj_qkv.weight' if arm == 'native' else 'cache_amplitude')}",
        ),
        pre_replacement_checkpoint_sha256=_sha256(checkpoint),
        model_loader_kwargs={"torch_dtype": torch.bfloat16},
    )
    cache_config = None
    if arm != "native":
        cache_config = CacheConfig(
            width=2,
            block_size=2,
            score="recency" if arm == "recency" else "exact_outer",
            read="rmsnorm",
            storage_dtype="fp32",
        )

    if not expect_success:
        with pytest.raises(ValueError, match="dtype"):
            load_qwen_arm(
                spec,
                model_config=None,
                cache_config=cache_config,
                base_model_loader=lambda *_args, **_kwargs: model,
                manager_factory=lambda received, _config: Manager(received),
            )
        return

    loaded = load_qwen_arm(
        spec,
        model_config=None,
        cache_config=cache_config,
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=lambda received, _config: Manager(received),
    )
    layer = loaded.model.model.layers[0].linear_attn
    cache_names = {
        "cache_gamma_q",
        "cache_gamma_k",
        "cache_sink_logit",
        "cache_amplitude",
    }
    for name, parameter in layer.named_parameters():
        expected = torch.float32 if name in cache_names else torch.bfloat16
        assert parameter.dtype == expected, name
    output = layer(torch.randn(1, 4, 12, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output.float()).all())


def test_qwen_arm_loader_rejects_asset_identity_before_loading(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        AssetIdentityError,
        ExternalAssetIdentity,
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    calls: list[str] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=ExternalAssetIdentity(
            name="model",
            path=model_asset,
            kind="file",
            size_bytes=model_asset.stat().st_size,
            sha256="0" * 64,
        ),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    with pytest.raises(AssetIdentityError, match="asset_hash_mismatch") as error:
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: calls.append("load"),
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (),
        )
    assert error.value.code == "asset_hash_mismatch"
    assert calls == []


def test_qwen_arm_loader_rejects_unknown_trainables_transactionally(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    model = _FakeQwen()
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("missing.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    with pytest.raises(KeyError, match="declared trainable"):
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: model,
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (0,),
        )
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def test_qwen_heal_load_spec_requires_a_native_checkpoint(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec

    model_asset = tmp_path / "model.bin"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    data_asset.write_bytes(b"data")

    with pytest.raises(ValueError, match="native_checkpoint_required"):
        QwenArmLoadSpec(
            arm="native",
            job_id="native-job",
            model_asset=_asset("model", model_asset),
            native_checkpoint=None,
            data_asset=_asset("data", data_asset),
            cache_resume=None,
            trainable_names=("memory.weight",),
            pre_replacement_checkpoint_sha256="a" * 64,
        )


def test_qwen_arm_loader_cross_checks_measured_pre_replacement_checkpoint_digest(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        AssetIdentityError,
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    calls: list[str] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256="f" * 64,
    )

    with pytest.raises(AssetIdentityError, match="checkpoint_identity_mismatch") as error:
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: calls.append("load"),
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (0,),
        )
    assert error.value.code == "checkpoint_identity_mismatch"
    assert calls == []


def test_qwen_arm_loader_uses_loaded_model_config_when_execution_passes_none(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    model = _FakeQwen()
    model.config = SimpleNamespace(name="loaded-config")
    seen: list[object] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    def manager_factory(_model: object, config: object) -> object:
        seen.append(config)
        return object()

    def native_installer(**kwargs: object) -> tuple[int, ...]:
        seen.append(kwargs["model_config"])
        return (0,)

    load_qwen_arm(
        spec,
        model_config=None,
        cache_config=None,
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=manager_factory,
        native_installer=native_installer,
    )
    assert seen == [model.config, model.config]


def _heal_arm(arm: str):
    from research.kmd2_ablation.qwen_backend import QwenHealArmContract

    return QwenHealArmContract(
        arm=arm,
        job_id=f"job-{arm}",
        seed=17,
        pre_replacement_checkpoint_sha256="a" * 64,
        data_sha256="c" * 64,
        example_ids=("ruler-000", "ruler-001", "ruler-002"),
        token_budget=12_288,
        update_budget=3,
        curriculum=(64, 128, 256),
        optimizer={"name": "adamw", "lr_memory": 2.0e-5, "betas": [0.9, 0.95]},
        schedule={"name": "cosine", "warmup_updates": 1},
        stopping={"max_nonfinite": 0, "early_stopping": False},
        eval_cells=("512:4q", "16K:4q", "32K:8q"),
        cache_match=(
            None
            if arm == "native"
            else {
                "width": 64,
                "block_size": 256,
                "read": "rmsnorm",
                "read_init": "gamma_one_sink_zero_amplitude_zero",
                "storage_dtype": "bf16",
                "lr_cache": 2.0e-3,
            }
        ),
        selection_policy=(
            None if arm == "native" else "recency" if arm == "recency" else "exact_outer"
        ),
    )


def test_three_arm_pairing_is_order_invariant_and_has_independent_canonical_id() -> None:
    from research.kmd2_ablation.qwen_backend import validate_three_arm_pairing

    native = _heal_arm("native")
    recency = _heal_arm("recency")
    surprise = _heal_arm("surprise")
    paired = validate_three_arm_pairing((surprise, native, recency))
    repeated = validate_three_arm_pairing((recency, surprise, native))

    expected_payload = {
        "cache_match": dict(recency.cache_match or {}),
        "curriculum": [64, 128, 256],
        "eval_cells": ["512:4q", "16K:4q", "32K:8q"],
        "data_sha256": "c" * 64,
        "example_ids": ["ruler-000", "ruler-001", "ruler-002"],
        "optimizer": dict(native.optimizer),
        "policies": {"native": None, "recency": "recency", "surprise": "exact_outer"},
        "pre_replacement_checkpoint_sha256": "a" * 64,
        "schedule": dict(native.schedule),
        "seed": 17,
        "stopping": dict(native.stopping),
        "token_budget": 12_288,
        "update_budget": 3,
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert paired.pairing_id == expected
    assert repeated.pairing_id == expected
    assert tuple(item.arm for item in paired.arms) == ("native", "recency", "surprise")
    assert paired.canonical_bytes == repeated.canonical_bytes
    assert paired.example_ids == native.example_ids


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 18),
        ("pre_replacement_checkpoint_sha256", "b" * 64),
        ("data_sha256", "d" * 64),
        ("example_ids", ("ruler-001", "ruler-000", "ruler-002")),
        ("token_budget", 12_287),
        ("update_budget", 4),
        ("curriculum", (64, 256)),
        ("optimizer", {"name": "adamw", "lr_memory": 3.0e-5}),
        ("schedule", {"name": "constant", "warmup_updates": 1}),
        ("stopping", {"max_nonfinite": 1, "early_stopping": False}),
        ("eval_cells", ("512:4q", "32K:8q")),
    ],
)
def test_three_arm_pairing_rejects_every_shared_contract_mismatch(
    field: str, replacement: object
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    jobs = [_heal_arm("native"), _heal_arm("recency"), _heal_arm("surprise")]
    jobs[2] = dataclasses.replace(jobs[2], **{field: replacement})
    with pytest.raises(PairingContractError, match="pairing_mismatch") as error:
        validate_three_arm_pairing(tuple(jobs))
    assert error.value.code == "pairing_mismatch"
    assert field in str(error.value)


@pytest.mark.parametrize(
    "changed_cache",
    [
        {"width": 32},
        {"block_size": 128},
        {"read": "unit_l2"},
        {"read_init": "different"},
        {"storage_dtype": "fp32"},
        {"lr_cache": 1.0e-3},
    ],
)
def test_three_arm_pairing_requires_capacity_read_gate_and_budget_matched_cache(
    changed_cache: dict[str, object],
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    jobs = [_heal_arm("native"), _heal_arm("recency"), _heal_arm("surprise")]
    altered = dict(jobs[2].cache_match or {})
    altered.update(changed_cache)
    jobs[2] = dataclasses.replace(jobs[2], cache_match=altered)
    with pytest.raises(PairingContractError, match="cache_match_mismatch") as error:
        validate_three_arm_pairing(tuple(jobs))
    assert error.value.code == "cache_match_mismatch"


def test_three_arm_pairing_requires_exactly_one_preregistered_arm() -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    with pytest.raises(PairingContractError, match="pairing_arm_set"):
        validate_three_arm_pairing(
            (_heal_arm("native"), _heal_arm("recency"), _heal_arm("recency"))
        )


def test_qwen_heal_causal_ce_matches_independent_shifted_fixture() -> None:
    from research.kmd2_ablation.qwen_training import (
        causal_cross_entropy,
    )

    student_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.2, 1.3, -0.7], [1.0, -0.5, 0.4], [0.1, 0.2, 0.3]]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[0, 1, -100, 2]])
    expected_ce = F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(causal_cross_entropy(student_logits, labels), expected_ce)


def test_qwen_heal_kl_matches_canonical_full_logit_numeric_fixture() -> None:
    from research.kmd2_ablation.qwen_training import distillation_kl

    student_logits = torch.tensor(
        [
            [[2.0, -1.0, 0.5], [0.2, 1.3, -0.7]],
            [[-0.5, 0.7, 1.4], [1.1, -0.4, 0.3]],
        ],
        dtype=torch.float64,
    )
    teacher_logits = torch.tensor(
        [
            [[1.5, -0.2, 0.1], [0.8, 0.3, -0.1]],
            [[0.4, 0.1, 1.0], [0.3, 0.7, -0.4]],
        ],
        dtype=torch.float64,
    )
    temperature = 1.7
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    expected_kl = (
        F.kl_div(
            student_log,
            teacher_log,
            reduction="batchmean",
            log_target=True,
        )
        * temperature**2
        / student_logits.shape[1]
    )

    assert torch.allclose(
        distillation_kl(student_logits, teacher_logits, temperature=temperature),
        expected_kl,
    )


def test_qwen_heal_layerwise_matches_canonical_normalized_residual_fixture() -> None:
    from research.kmd2_ablation.qwen_training import layerwise_alignment_loss

    student_hidden = (
        torch.tensor([[[99.0, -99.0], [50.0, -50.0]]], dtype=torch.float64),
        torch.tensor([[[2.0, 1.0], [6.0, 2.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], dtype=torch.float64),
    )
    teacher_hidden = (
        torch.zeros((1, 2, 2), dtype=torch.float64),
        torch.tensor([[[1.0, 1.0], [3.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[[2.0, 2.0], [4.0, 8.0]]], dtype=torch.float64),
    )
    expected_layers = []
    for student, teacher in zip(student_hidden[1:], teacher_hidden[1:]):
        student = student.float()
        teacher = teacher.float()
        expected_layers.append(
            (student - teacher).square().mean()
            / teacher.square().mean().clamp_min(1.0e-8)
        )
    expected_layerwise = torch.stack(expected_layers).mean()

    assert torch.allclose(
        layerwise_alignment_loss(student_hidden, teacher_hidden),
        expected_layerwise,
    )


class _HealModel(torch.nn.Module):
    def __init__(self, *, nan_output: bool = False) -> None:
        super().__init__()
        self.memory_weight = torch.nn.Parameter(
            torch.tensor(
                [[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2], [0.2, 0.1, -0.3]],
                dtype=torch.float32,
            )
        )
        self.cache_amplitude = torch.nn.Parameter(torch.tensor([0.25]))
        self.backbone_weight = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.nan_output = nan_output
        self.gradient_checkpointing_calls = 0
        self.forward_example_inputs: list[torch.Tensor] = []

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_calls += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        assert use_cache is False
        self.forward_example_inputs.append(input_ids.detach().clone())
        one_hot = F.one_hot(input_ids, num_classes=3).to(torch.float32)
        logits = one_hot @ self.memory_weight
        logits = logits + self.cache_amplitude.view(1, 1, 1) * one_hot
        if self.nan_output:
            logits = logits * torch.tensor(float("nan"))
        return SimpleNamespace(logits=logits, hidden_states=(one_hot, logits))


class _HealTeacher(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        assert use_cache is False
        one_hot = F.one_hot(input_ids, num_classes=3).to(torch.float32)
        logits = one_hot.roll(1, dims=-1) * 0.4
        return SimpleNamespace(logits=logits, hidden_states=(one_hot * 0.9, logits))


class _GuardProbeHealModel(_HealModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        return super().forward(
            input_ids,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )


def _training_config(**changes: object):
    from research.kmd2_ablation.qwen_training import QwenHealTrainingConfig

    values: dict[str, object] = {
        "objective": "language_model_heal",
        "ce_weight": 1.0,
        "kl_weight": 0.2,
        "layerwise_weight": 0.1,
        "temperature": 1.5,
        "accumulation_steps": 2,
        "max_updates": 1,
        "max_tokens": 6,
        "gradient_checkpointing": True,
    }
    values.update(changes)
    return QwenHealTrainingConfig(**values)


def _batch(example_id: str, tokens: tuple[int, int, int]) -> dict[str, object]:
    input_ids = torch.tensor([tokens], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": (example_id,),
    }


def _optimizer_and_scheduler(model: _HealModel):
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("memory_weight",),
        cache_parameter_names=("cache_amplitude",),
        learning_rate=0.05,
        lr_cache=0.1,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
    )
    return optimizer, scheduler


def test_qwen_heal_optimizer_groups_are_exact_named_and_zero_decay_cache() -> None:
    from research.kmd2_ablation.qwen_training import (
        build_qwen_heal_optimizer,
        project_cache_amplitudes_,
    )

    model = _HealModel()
    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("memory_weight",),
        cache_parameter_names=("cache_amplitude",),
        learning_rate=2.0e-5,
        lr_cache=2.0e-3,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    assert [group["name"] for group in optimizer.param_groups] == ["memory", "cache"]
    assert [group["parameter_names"] for group in optimizer.param_groups] == [
        ("memory_weight",),
        ("cache_amplitude",),
    ]
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 2.0e-3
    with torch.no_grad():
        model.cache_amplitude.fill_(1.7)
    projected = project_cache_amplitudes_(model)
    assert projected == ("cache_amplitude",)
    assert model.cache_amplitude.item() == 1.0


def test_qwen_heal_one_update_accumulates_fixed_windows_projects_and_logs() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    torch.manual_seed(123)
    model = _HealModel()
    teacher = _HealTeacher()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = model.memory_weight.detach().clone()
    log = trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    )

    assert trainer.step == 1
    assert trainer.tokens_seen == 6
    assert trainer.example_cursor == 2
    assert model.gradient_checkpointing_calls == 1
    assert not torch.equal(model.memory_weight, before)
    assert 0.0 <= model.cache_amplitude.item() <= 1.0
    assert scheduler.last_epoch == 1
    record = log.as_dict()
    assert record["job_id"] == "job-surprise"
    assert record["pairing_id"] == "f" * 64
    assert record["arm"] == "surprise"
    assert record["update"] == 1
    assert record["tokens_seen"] == 6
    assert record["example_ids"] == ["e0", "e1"]
    assert record["microbatches"] == 2
    assert record["skipped_steps"] == 0
    assert set(record["losses"]) == {"total", "ce", "kl", "layerwise"}
    assert all(torch.isfinite(torch.tensor(value)) for value in record["losses"].values())
    assert record["learning_rates"] == {
        "cache": pytest.approx(0.05),
        "memory": pytest.approx(0.025),
    }

    with pytest.raises(RuntimeError, match="update_budget_exhausted"):
        trainer.train_update(
            (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
        )


@pytest.mark.parametrize(
    ("extra_inputs", "expected_code"),
    [
        ({"attention_mask": torch.tensor([[1, 1, 0]])}, "padding_unsupported"),
        ({"position_ids": torch.tensor([[0, 1, 0]])}, "position_reset"),
    ],
)
def test_qwen_heal_trainer_guards_padding_and_position_resets_before_forward(
    extra_inputs: dict[str, torch.Tensor], expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import FullRecomputeCallError
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    model = _GuardProbeHealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(
            objective="synthetic_only",
            kl_weight=0.0,
            layerwise_weight=0.0,
            accumulation_steps=1,
            max_tokens=3,
            gradient_checkpointing=False,
        ),
        job_id="guarded-train",
        pairing_id="a" * 64,
        arm="surprise",
        expected_example_windows=(("e0",),),
    )
    batch = _batch("e0", (0, 1, 2))
    batch.update(extra_inputs)

    with pytest.raises(FullRecomputeCallError) as caught:
        trainer.train_update((batch,))

    assert caught.value.code == expected_code
    assert model.forward_example_inputs == []
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0


def test_qwen_heal_routes_teacher_inputs_to_the_explicit_teacher_device() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    class RecordingTeacher(_HealTeacher):
        def __init__(self) -> None:
            super().__init__()
            self.input_devices: list[torch.device] = []

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            self.input_devices.append(input_ids.device)
            return super().forward(input_ids, **kwargs)

    model = _HealModel()
    teacher = RecordingTeacher()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
        teacher_device=torch.device("cpu"),
    )
    trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    )
    assert teacher.input_devices == [torch.device("cpu"), torch.device("cpu")]


def test_qwen_heal_rejects_mismatched_window_before_forward_or_mutation() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer, QwenTrainingError

    model = _HealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=_HealTeacher(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-recency",
        pairing_id="f" * 64,
        arm="recency",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenTrainingError, match="example_window_mismatch") as error:
        trainer.train_update(
            (_batch("e1", (2, 1, 0)), _batch("e0", (0, 1, 2)))
        )
    assert error.value.code == "example_window_mismatch"
    assert model.forward_example_inputs == []
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0


def test_qwen_heal_nonfinite_loss_is_a_skipped_failure_without_optimizer_step() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer, QwenTrainingError

    model = _HealModel(nan_output=True)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=_HealTeacher(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenTrainingError, match="nonfinite_loss") as error:
        trainer.train_update(
            (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
        )
    assert error.value.code == "nonfinite_loss"
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0
    assert trainer.skipped_steps == 1
    assert scheduler.last_epoch == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    ("interruption_type", "interruption_value"),
    [(KeyboardInterrupt, "scheduler interrupted"), (SystemExit, 37)],
)
def test_qwen_heal_base_exception_after_optimizer_step_rolls_back_exactly(
    interruption_type: type[BaseException], interruption_value: object
) -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    model = _HealModel()
    optimizer, _ = _optimizer_and_scheduler(model)
    interruption = interruption_type(interruption_value)

    class InterruptingScheduler:
        def __init__(self) -> None:
            self.optimizer = optimizer
            self.progress = 0

        def state_dict(self) -> dict[str, int]:
            return {"progress": self.progress}

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.progress = state["progress"]

        def step(self) -> None:
            self.progress = 1
            raise interruption

    scheduler = InterruptingScheduler()
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(
            objective="synthetic_only",
            kl_weight=0.0,
            layerwise_weight=0.0,
            accumulation_steps=1,
            max_tokens=3,
            gradient_checkpointing=False,
        ),
        job_id="transactional-train",
        pairing_id="b" * 64,
        arm="surprise",
        expected_example_windows=(("e0",),),
    )
    parameter_snapshot = copy.deepcopy(model.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(interruption_type) as caught:
        trainer.train_update((_batch("e0", (0, 1, 2)),))

    assert caught.value is interruption
    _assert_nested_equal(model.state_dict(), parameter_snapshot)
    _assert_nested_equal(optimizer.state_dict(), optimizer_snapshot)
    _assert_nested_equal(scheduler.state_dict(), scheduler_snapshot)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0
    assert trainer.skipped_steps == 1


def test_qwen_heal_teacher_is_required_except_explicit_synthetic_only() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenHealTrainer,
        TeacherRequiredError,
        validate_teacher_requirement,
    )

    ordinary = _training_config()
    with pytest.raises(TeacherRequiredError, match="teacher_required") as preflight:
        validate_teacher_requirement(ordinary, teacher_present=False, phase="preflight")
    assert preflight.value.code == "teacher_required"

    model = _HealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    with pytest.raises(TeacherRequiredError, match="teacher_required") as runtime:
        QwenHealTrainer(
            model=model,
            teacher=None,
            optimizer=optimizer,
            scheduler=scheduler,
            config=ordinary,
            job_id="job-native",
            pairing_id="f" * 64,
            arm="native",
            expected_example_windows=(("e0",), ("e1",)),
        )
    assert runtime.value.code == "teacher_required"

    synthetic = _training_config(
        objective="synthetic_only", kl_weight=0.0, layerwise_weight=0.0
    )
    validate_teacher_requirement(synthetic, teacher_present=False, phase="preflight")
    synthetic_optimizer, synthetic_scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=synthetic_optimizer,
        scheduler=synthetic_scheduler,
        config=synthetic,
        job_id="job-native",
        pairing_id="f" * 64,
        arm="native",
        expected_example_windows=(("e0",), ("e1",)),
    )
    assert trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    ).as_dict()["losses"]["kl"] == 0.0


class _CheckpointLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.memory = torch.nn.Parameter(
            torch.tensor([[offset, offset + 1.0], [offset + 2.0, offset + 3.0]])
        )
        self.cache_amplitude = torch.nn.Parameter(torch.tensor([0.2 + offset / 10.0]))
        self.register_buffer("native_buffer", torch.tensor([int(offset)], dtype=torch.int64))


class _CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.layer0 = _CheckpointLayer(0.0)
        self.layer1 = _CheckpointLayer(1.0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)


def _checkpoint_parts():
    from research.kmd2_ablation.qwen_checkpoint import QwenCheckpointMetadata
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    model = _CheckpointModel()
    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("layer0.memory", "layer1.memory"),
        cache_parameter_names=("layer0.cache_amplitude", "layer1.cache_amplitude"),
        learning_rate=0.01,
        lr_cache=0.02,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    metadata = QwenCheckpointMetadata(
        job_id="job-surprise",
        pairing_id="d" * 64,
        arm="surprise",
        step=1,
        tokens_seen=6,
        source_hashes={
            "gdn3/kmd2_native.py": "1" * 64,
            "research/kmd2_ablation/qwen_exact_cache.py": "2" * 64,
        },
        data_identity={"sha256": "3" * 64, "row_count": 3},
        example_ids=("e0", "e1", "e2"),
        promotion_config={"width": 64, "policy": "exact_outer", "min_gate_mean": 0.005},
    )
    return model, optimizer, scheduler, metadata


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_qwen_checkpoint_is_atomic_complete_and_records_exact_manifests(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import save_qwen_checkpoint

    random.seed(444)
    torch.manual_seed(555)
    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "schema_version",
        "metadata",
        "target_module_names",
        "model_state",
        "tensor_manifest",
        "optimizer_parameter_names",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "amplitude_range",
    }
    assert payload["schema_version"] == 1
    assert payload["target_module_names"] == ["layer0", "layer1"]
    assert tuple(payload["model_state"]) == (
        "layer0.cache_amplitude",
        "layer0.memory",
        "layer0.native_buffer",
        "layer1.cache_amplitude",
        "layer1.memory",
        "layer1.native_buffer",
    )
    assert not any(name.startswith("backbone") for name in payload["model_state"])
    assert all(tensor.device.type == "cpu" for tensor in payload["model_state"].values())
    assert payload["tensor_manifest"] == [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in payload["model_state"].items()
    ]
    assert payload["optimizer_parameter_names"] == [
        ["layer0.memory", "layer1.memory"],
        ["layer0.cache_amplitude", "layer1.cache_amplitude"],
    ]
    assert payload["metadata"]["job_id"] == "job-surprise"
    assert payload["metadata"]["step"] == 1
    assert payload["metadata"]["tokens_seen"] == 6
    assert payload["amplitude_range"][0] >= 0.0
    assert payload["amplitude_range"][1] <= 1.0
    assert set(payload["rng_state"]) == {"python", "torch_cpu", "torch_cuda"}
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


def test_qwen_checkpoint_interrupted_save_preserves_destination_and_cleans_temp(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import save_qwen_checkpoint

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")

    def interrupted_save(_payload: object, temp_path: Path) -> None:
        temp_path.write_bytes(b"partial")
        raise OSError("simulated interruption")

    with pytest.raises(OSError, match="simulated interruption"):
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=interrupted_save,
        )
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


@pytest.mark.parametrize(
    ("writer_output", "expected_code"),
    [
        ("truncated", "checkpoint_decode_failed"),
        ("different_job", "resume_identity_mismatch"),
        ("different_pair", "resume_identity_mismatch"),
        ("different_model_state", "checkpoint_serialization_mismatch"),
        ("in_place_model_state", "checkpoint_serialization_mismatch"),
        ("different_optimizer_state", "checkpoint_serialization_mismatch"),
    ],
)
def test_qwen_checkpoint_save_rejects_corrupt_or_different_serialized_candidate(
    tmp_path: Path, writer_output: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    known_good = path.read_bytes()

    def corrupting_writer(payload: object, temporary: Path) -> None:
        if writer_output == "truncated":
            temporary.write_bytes(b"truncated torch payload")
            return
        assert isinstance(payload, dict)
        if writer_output == "in_place_model_state":
            payload["model_state"]["layer0.memory"].add_(0.125)
            torch.save(payload, temporary)
            return
        candidate = copy.deepcopy(payload)
        if writer_output == "different_job":
            candidate["metadata"]["job_id"] = "other-job"
        elif writer_output == "different_pair":
            candidate["metadata"]["pairing_id"] = "e" * 64
        elif writer_output == "different_model_state":
            candidate["model_state"]["layer0.memory"].add_(0.125)
        else:
            slot = next(iter(candidate["optimizer_state"]["state"].values()))
            slot["exp_avg"].add_(0.125)
        torch.save(candidate, temporary)

    with pytest.raises(QwenCheckpointError) as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=corrupting_writer,
        )
    assert caught.value.code == expected_code
    assert path.read_bytes() == known_good
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("metadata_progress", "optimizer_state_invalid"),
        ("scheduler_progress", "scheduler_state_invalid"),
    ],
)
def test_qwen_checkpoint_save_self_validates_progress_before_publish(
    tmp_path: Path, corruption: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    if corruption == "metadata_progress":
        metadata = dataclasses.replace(metadata, step=2, tokens_seen=12)
    else:
        scheduler.last_epoch = 2
        scheduler._step_count = 3
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")
    writer_calls: list[Path] = []

    def recording_save(payload: object, temporary: Path) -> None:
        writer_calls.append(temporary)
        torch.save(payload, temporary)

    with pytest.raises(QwenCheckpointError) as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=recording_save,
        )
    assert caught.value.code == expected_code
    assert writer_calls == []
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


def test_qwen_checkpoint_save_rejects_optimizer_parameters_outside_targets_before_publish(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")
    writer_calls: list[Path] = []

    def recording_save(payload: object, temporary: Path) -> None:
        writer_calls.append(temporary)
        torch.save(payload, temporary)

    with pytest.raises(QwenCheckpointError, match="optimizer_target_coverage") as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0",),
            save_function=recording_save,
        )
    assert caught.value.code == "optimizer_target_coverage"
    assert writer_calls == []
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


def test_qwen_checkpoint_load_rejects_optimizer_parameters_outside_targets_without_mutation(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    partial = tmp_path / "partial.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    payload["target_module_names"] = ["layer0"]
    payload["model_state"] = {
        name: tensor
        for name, tensor in payload["model_state"].items()
        if name.startswith("layer0.")
    }
    payload["tensor_manifest"] = [
        item
        for item in payload["tensor_manifest"]
        if item["name"].startswith("layer0.")
    ]
    amplitude = payload["model_state"]["layer0.cache_amplitude"]
    payload["amplitude_range"] = [float(amplitude.min()), float(amplitude.max())]
    torch.save(payload, partial)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_python_rng = random.getstate()
    before_torch_rng = torch.get_rng_state().clone()
    with pytest.raises(QwenCheckpointError, match="optimizer_target_coverage") as caught:
        load_qwen_checkpoint(
            partial,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0",),
        )
    assert caught.value.code == "optimizer_target_coverage"
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)
    assert random.getstate() == before_python_rng
    assert torch.equal(torch.get_rng_state(), before_torch_rng)


def test_qwen_checkpoint_safe_loader_rejects_pickle_execution_without_mutation(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    marker = tmp_path / "checkpoint-pickle-executed.txt"
    checkpoint = tmp_path / "malicious.pt"
    torch.save(_PickleMarkerPayload(marker, "checkpoint"), checkpoint)
    model_snapshot = copy.deepcopy(model.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(QwenCheckpointError) as caught:
        load_qwen_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )

    assert caught.value.code == "checkpoint_decode_failed"
    assert not marker.exists()
    _assert_nested_equal(model.state_dict(), model_snapshot)
    _assert_nested_equal(optimizer.state_dict(), optimizer_snapshot)
    _assert_nested_equal(scheduler.state_dict(), scheduler_snapshot)


def test_qwen_checkpoint_resume_restores_model_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    random.seed(1234)
    torch.manual_seed(4321)
    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_model = copy.deepcopy(payload["model_state"])
    expected_optimizer = copy.deepcopy(payload["optimizer_state"])
    expected_scheduler = copy.deepcopy(payload["scheduler_state"])
    expected_python_rng = payload["rng_state"]["python"]
    expected_torch_rng = payload["rng_state"]["torch_cpu"]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(9.0)
    optimizer.param_groups[0]["lr"] = 9.0
    scheduler.last_epoch = 99
    random.seed(999)
    torch.manual_seed(999)

    resumed = load_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expectation=QwenResumeExpectation.from_metadata(metadata),
        target_module_names=("layer0", "layer1"),
    )
    assert resumed.step == 1
    assert resumed.tokens_seen == 6
    assert resumed.job_id == "job-surprise"
    selected = {name: model.state_dict()[name].cpu() for name in expected_model}
    _assert_nested_equal(selected, expected_model)
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler)
    assert random.getstate() == expected_python_rng
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)


@pytest.mark.parametrize(
    ("interruption_type", "interruption_value"),
    [(KeyboardInterrupt, "stop-now"), (SystemExit, 17)],
)
def test_qwen_checkpoint_resume_rolls_back_exactly_before_reraising_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
    interruption_value: object,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    with torch.no_grad():
        model.layer0.memory.add_(7.0)
        model.layer1.cache_amplitude.mul_(0.5)
        for slot in optimizer.state.values():
            slot["exp_avg"].add_(3.0)
    optimizer.param_groups[0]["lr"] = 0.31
    optimizer.param_groups[1]["lr"] = 0.47
    scheduler.last_epoch = 11
    scheduler._step_count = 12
    scheduler._last_lr = [0.31, 0.47]
    random.seed(9087)
    torch.manual_seed(8709)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7890)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_python_rng = random.getstate()
    before_torch_rng = torch.get_rng_state().clone()
    before_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    interruption = interruption_type(interruption_value)
    scheduler_type = type(scheduler)
    original_load_state_dict = scheduler_type.load_state_dict
    interrupted = False

    def interrupt_once(scheduler_self: object, state_dict: object) -> object:
        nonlocal interrupted
        result = original_load_state_dict(scheduler_self, state_dict)
        if scheduler_self is scheduler and not interrupted:
            interrupted = True
            random.random()
            torch.rand(4)
            if torch.cuda.is_available():
                torch.rand(4, device="cuda")
            raise interruption
        return result

    monkeypatch.setattr(scheduler_type, "load_state_dict", interrupt_once)
    with pytest.raises(interruption_type) as caught:
        load_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    assert caught.value is interruption
    assert interrupted is True
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)
    assert random.getstate() == before_python_rng
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), before_cuda_rng, strict=True
        )
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("job_id", "other-job"),
        ("pairing_id", "e" * 64),
        ("arm", "recency"),
        ("source_hashes", {"gdn3/kmd2_native.py": "4" * 64}),
        ("data_identity", {"sha256": "5" * 64, "row_count": 3}),
        ("example_ids", ("e1", "e0", "e2")),
        ("promotion_config", {"width": 32, "policy": "exact_outer"}),
    ],
)
def test_qwen_checkpoint_resume_rejects_every_identity_mismatch_without_mutation(
    tmp_path: Path, field: str, replacement: object
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    expectation = dataclasses.replace(
        QwenResumeExpectation.from_metadata(metadata), **{field: replacement}
    )
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(QwenCheckpointError, match="resume_identity_mismatch") as error:
        load_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=expectation,
            target_module_names=("layer0", "layer1"),
        )
    assert error.value.code == "resume_identity_mismatch"
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)


@pytest.mark.parametrize("corruption", ["missing_name", "shape", "dtype", "amplitude"])
def test_qwen_checkpoint_rejects_tensor_corruption_before_mutation(
    tmp_path: Path, corruption: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    bad = tmp_path / "bad.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    if corruption == "missing_name":
        del payload["model_state"]["layer1.memory"]
    elif corruption == "shape":
        payload["model_state"]["layer1.memory"] = torch.zeros(3, 2)
    elif corruption == "dtype":
        payload["model_state"]["layer1.memory"] = payload["model_state"][
            "layer1.memory"
        ].double()
    else:
        payload["model_state"]["layer1.cache_amplitude"].fill_(1.1)
    torch.save(payload, bad)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenCheckpointError):
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    _assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("group_parameter_id", "optimizer_parameter_mismatch"),
        ("group_parameter_order", "optimizer_parameter_mismatch"),
        ("missing_slot", "optimizer_state_invalid"),
        ("foreign_slot", "optimizer_state_invalid"),
        ("moment_shape", "optimizer_state_invalid"),
        ("moment_dtype", "optimizer_state_invalid"),
        ("moment_nonfinite", "optimizer_state_invalid"),
        ("parameter_step", "optimizer_state_invalid"),
        ("group_hyperparameter", "optimizer_state_invalid"),
        ("scheduler_static", "scheduler_state_invalid"),
        ("scheduler_progress", "scheduler_state_invalid"),
        ("scheduler_group_lr", "scheduler_state_invalid"),
        ("coordinated_group_lr", "scheduler_state_invalid"),
    ],
)
def test_qwen_checkpoint_strictly_rejects_optimizer_and_scheduler_corruption(
    tmp_path: Path, corruption: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    bad = tmp_path / f"{corruption}.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    optimizer_state = payload["optimizer_state"]
    groups = optimizer_state["param_groups"]
    slots = optimizer_state["state"]
    first_parameter_id = groups[0]["params"][0]
    if corruption == "group_parameter_id":
        groups[0]["params"][0] = 999
    elif corruption == "group_parameter_order":
        groups[0]["params"] = list(reversed(groups[0]["params"]))
    elif corruption == "missing_slot":
        del slots[first_parameter_id]
    elif corruption == "foreign_slot":
        slots[999] = copy.deepcopy(slots[first_parameter_id])
    elif corruption == "moment_shape":
        slots[first_parameter_id]["exp_avg"] = torch.zeros(3, 2)
    elif corruption == "moment_dtype":
        slots[first_parameter_id]["exp_avg_sq"] = slots[first_parameter_id][
            "exp_avg_sq"
        ].double()
    elif corruption == "moment_nonfinite":
        slots[first_parameter_id]["exp_avg"].fill_(float("inf"))
    elif corruption == "parameter_step":
        slots[first_parameter_id]["step"].fill_(2.0)
    elif corruption == "group_hyperparameter":
        groups[0]["betas"] = (0.5, 0.5)
    elif corruption == "scheduler_static":
        payload["scheduler_state"]["gamma"] = 0.75
    elif corruption == "scheduler_progress":
        payload["scheduler_state"]["last_epoch"] = 7
    elif corruption == "scheduler_group_lr":
        payload["scheduler_state"]["_last_lr"][0] *= 0.5
    else:
        arbitrary_rates = [0.123, 0.456]
        for group, rate in zip(groups, arbitrary_rates, strict=True):
            group["lr"] = rate
        payload["scheduler_state"]["_last_lr"] = arbitrary_rates
    torch.save(payload, bad)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(QwenCheckpointError) as caught:
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    assert caught.value.code == expected_code
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)


def test_qwen_checkpoint_validates_production_lambda_schedule_from_base_lrs(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, _step_scheduler, metadata = _checkpoint_parts()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
    )
    optimizer._opt_called = True
    scheduler.step()
    good = tmp_path / "good-lambda.pt"
    bad = tmp_path / "bad-lambda.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    arbitrary_rates = [0.321, 0.654]
    for group, rate in zip(
        payload["optimizer_state"]["param_groups"], arbitrary_rates, strict=True
    ):
        group["lr"] = rate
    payload["scheduler_state"]["_last_lr"] = arbitrary_rates
    torch.save(payload, bad)

    with pytest.raises(QwenCheckpointError, match="scheduler_state_invalid"):
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )


def test_qwen_checkpoint_save_rejects_out_of_range_amplitude(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    with torch.no_grad():
        model.layer0.cache_amplitude.fill_(-0.01)
    with pytest.raises(QwenCheckpointError, match="amplitude_out_of_range"):
        save_qwen_checkpoint(
            tmp_path / "bad.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
        )


def _qwen_adapter_job(
    checkpoint_sha256: str, data_sha256: str = "c" * 64
) -> dict[str, object]:
    from research.kmd2_ablation.qwen_training import derive_three_arm_pairing

    job: dict[str, object] = {
        "job_id": "job-surprise",
        "experiment_id": "experiment-qwen",
        "seed": 17,
        "stage": "qwen_heal",
        "backend": "qwen",
        "arm_id": "exact_cache.selector.exact_outer",
        "canonical_config": {
            "backend": "qwen",
            "qwen": {"run_mode": "heal"},
            "budget": {"updates": 1, "tokens": 6},
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.05,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
            "schedule": {"name": "cosine", "warmup_updates": 0},
            "lengths": {"curriculum": [3], "extrapolation": [3, 6]},
            "evaluation": {
                "primary_metric": "token_accuracy",
                "direction": "maximize",
            },
            "cache": {
                "width": 2,
                "block_size": 2,
                "score": "exact_outer",
                "read": "rmsnorm",
                "read_init": "gamma_one_sink_zero_amplitude_zero",
                "storage_dtype": "fp32",
                "compute_dtype": "fp32",
                "lr_cache": 0.1,
                "weight_decay_cache": 0.0,
            },
            "promotion": {"min_gate_mean": 0.005},
            "task": {
                "name": "ruler",
                "params": {
                    "objective": "language_model_heal",
                    "ce_weight": 1.0,
                    "kl_weight": 0.2,
                    "layerwise_weight": 0.1,
                    "temperature": 1.5,
                    "accumulation_steps": 2,
                    "gradient_checkpointing": True,
                    "example_ids": ["e0", "e1"],
                    "memory_parameter_names": ["memory_weight"],
                    "cache_parameter_names": ["cache_amplitude"],
                    "stopping": {
                        "max_nonfinite": 0,
                        "early_stopping": False,
                    },
                },
            },
        },
    }
    pairing = derive_three_arm_pairing(
        job,
        example_ids=("e0", "e1"),
        pre_replacement_checkpoint_sha256=checkpoint_sha256,
        data_sha256=data_sha256,
    )
    job["pairing_id"] = pairing.pairing_id
    return job


def _exact_cache_result_diagnostics() -> dict[str, object]:
    return {
        "width": 2,
        "block_size": 2,
        "compute_dtype": "fp32",
        "storage_dtype": "fp32",
        "coordinate_frame": "rotated_recurrence",
        "inclusive_causality": True,
        "tie_policy": "score_desc_position_desc",
        "score_definition": "exact_outer",
        "amplitude_initial": [0.25],
        "amplitude_final": [0.3],
        "selected_index_digest": "1" * 64,
        "score_digest": "2" * 64,
        "selected_index_sample": [0, 1],
        "score_statistics": {"count": 2, "min": 0.1, "max": 0.3, "mean": 0.2},
        "retention_count": 2,
        "eviction_count": 1,
        "persistent_bytes": 64,
        "block_bytes": 64,
        "persistent_hit_rate": 1.0,
        "conditional_read_accuracy": 1.0,
        "sink_mass": 0.1,
        "top1_mass": 0.9,
        "stale_occupancy": 0.0,
        "stale_error": 0.0,
        "attention_entropy": 0.2,
        "cache_output_norm": 0.4,
        "state_output_norm": 0.8,
        "implementation_paths": {
            "scan": "reference_full_recompute",
            "score": "exact_outer",
            "selection": "stable_topk",
            "read": "rmsnorm_sink",
        },
    }


def test_qwen_source_hashes_cover_the_exact_semantic_execution_graph() -> None:
    from research.kmd2_ablation.qwen_training import _source_hashes

    root = Path(__file__).resolve().parents[2]
    expected = {
        "research/kmd2_ablation/config.py",
        "research/kmd2_ablation/exact_cache.py",
        "research/kmd2_ablation/qwen_backend.py",
        "research/kmd2_ablation/qwen_checkpoint.py",
        "research/kmd2_ablation/qwen_exact_cache.py",
        "research/kmd2_ablation/qwen_training.py",
        "research/kmd2_ablation/qwen_variants.py",
        "research/kmd2_ablation/results.py",
        "research/kmd2_ablation/runner.py",
        "research/kmd2_ablation/tasks/ruler.py",
        "research/kmd2_ablation/variants.py",
        "gdn3/_reference_recurrence.py",
        "gdn3/gdn3_upgrade.py",
        "gdn3/kmd2_fast_scan.py",
        "gdn3/kmd2_native.py",
    }
    actual = _source_hashes()
    assert set(actual) == expected
    for relative, digest in actual.items():
        assert digest == hashlib.sha256((root / relative).read_bytes()).hexdigest()


def test_qwen_source_hashes_change_only_ruler_digest_when_ruler_source_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    from research.kmd2_ablation.qwen_training import _source_hashes

    root = Path(__file__).resolve().parents[2]
    ruler_path = root / "research/kmd2_ablation/tasks/ruler.py"
    ruler_source = ruler_path.read_bytes()
    baseline = _source_hashes()
    original_open = Path.open

    def open_with_ruler_mutation(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ):
        if path == ruler_path and mode == "rb":
            return BytesIO(ruler_source + b"\n# semantic mutation probe\n")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_ruler_mutation)
    mutated = _source_hashes()

    assert set(mutated) == set(baseline)
    assert {
        relative for relative in baseline if mutated[relative] != baseline[relative]
    } == {"research/kmd2_ablation/tasks/ruler.py"}


class _RunnerQwenHealConfig:
    def __init__(
        self,
        canonical_config: dict[str, object],
        *,
        seeds: tuple[int, ...] = (101, 202, 303),
    ) -> None:
        self._canonical_config = canonical_config
        self.backend = "qwen"
        self.required_stage = "qwen_heal"
        self.mechanism = "exact_cache"
        self.seeds = seeds
        self.task = SimpleNamespace(params=canonical_config["task"]["params"])

    def semantic_dict(self) -> dict[str, object]:
        return copy.deepcopy(self._canonical_config)


def test_qwen_heal_runner_expands_exact_three_arm_jobs_with_strong_pairing() -> None:
    from research.kmd2_ablation.runner import _expand_jobs

    checkpoint_digest = "a" * 64
    data_digest = "c" * 64
    base_job = _qwen_adapter_job(checkpoint_digest, data_digest)
    config = _RunnerQwenHealConfig(base_job["canonical_config"])
    jobs = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": data_digest},
    )

    assert len(jobs) == 9
    assert {job["arm_id"] for job in jobs} == {"native", "recency", "surprise"}
    assert {job["seed"] for job in jobs} == {101, 202, 303}
    for seed in config.seeds:
        paired = [job for job in jobs if job["seed"] == seed]
        assert len(paired) == 3
        assert len({job["pairing_id"] for job in paired}) == 1

    changed_checkpoint = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": "b" * 64, "data": data_digest},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_checkpoint}
    )

    changed_data = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": "d" * 64},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_data}
    )

    reordered = copy.deepcopy(base_job["canonical_config"])
    reordered["task"]["params"]["example_ids"] = ["e1", "e0"]
    changed_examples = _expand_jobs(
        _RunnerQwenHealConfig(reordered),
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": data_digest},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_examples}
    )


@pytest.mark.parametrize(
    "seeds",
    [
        (101,),
        (101, 202),
        (101, 202, 303, 404),
        (101, 202, 101),
    ],
)
def test_qwen_heal_runner_requires_exactly_three_unique_seeds_before_expansion(
    seeds: tuple[int, ...],
) -> None:
    from research.kmd2_ablation.runner import PreflightCheckError, _expand_jobs

    checkpoint_digest = "a" * 64
    data_digest = "c" * 64
    config = _RunnerQwenHealConfig(
        _qwen_adapter_job(checkpoint_digest, data_digest)["canonical_config"],
        seeds=seeds,
    )
    with pytest.raises(PreflightCheckError) as caught:
        _expand_jobs(
            config,
            "exact_cache.selector.exact_outer",
            asset_hashes={
                "checkpoint": checkpoint_digest,
                "data": data_digest,
            },
        )
    assert caught.value.code == "qwen_seed_matrix_invalid"


@pytest.mark.parametrize(
    "seeds",
    [[101], [101, 202], [101, 202, 303, 404], [101, 202, 101]],
)
def test_qwen_heal_raw_preflight_reports_one_stable_seed_matrix_code(
    seeds: list[int],
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    job = _qwen_adapter_job("a" * 64)
    raw = copy.deepcopy(job["canonical_config"])
    raw.update(
        {
            "mechanism": "exact_cache",
            "variant": "top_surprise",
            "required_stage": "qwen_heal",
            "seeds": seeds,
        }
    )
    codes = validate_raw_scientific_config(raw, backend="qwen", mode="heal")
    assert codes.count("qwen_seed_matrix_invalid") == 1


@pytest.mark.parametrize(
    "params",
    [
        {"synthetic_only": True},
        {"objective": "synthetic_only", "synthetic_only": True},
    ],
)
def test_qwen_preflight_rejects_legacy_synthetic_only_boolean_declaration(
    params: dict[str, object],
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    raw = {
        "backend": "qwen",
        "mechanism": "exact_cache",
        "variant": "top_surprise",
        "required_stage": "qwen_heal",
        "seeds": [101, 202, 303],
        "qwen": {
            "run_mode": "heal",
            "streaming": False,
            "decode": False,
            "packing": False,
            "padding": "none",
            "attention_mask": "none",
        },
        "task": {"params": {**params, "example_ids": ["e0"]}},
        "cache": {"width": 2, "block_size": 2},
        "lengths": {"curriculum": [4]},
        "model": {"ffn_dim": 8, "ffn_match_lower": 8, "ffn_match_upper": 8},
    }
    codes = validate_raw_scientific_config(raw, backend="qwen", mode="heal")
    assert codes.count("qwen_synthetic_only_declaration_invalid") == 1


def test_qwen_synthetic_only_objective_is_the_only_teacher_omission_signal(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.runner import PreflightCheckError, _external_asset_paths

    options = SimpleNamespace(
        model=tmp_path / "model",
        tokenizer=None,
        checkpoint=tmp_path / "checkpoint.pt",
        data=tmp_path / "data.pt",
        teacher_model=None,
    )

    def config(params: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            backend="qwen",
            qwen=SimpleNamespace(run_mode="heal"),
            task=SimpleNamespace(params=params),
        )

    accepted = _external_asset_paths(
        options,
        config({"objective": "synthetic_only"}),
    )
    assert set(accepted) == {"model", "checkpoint", "data"}

    with pytest.raises(PreflightCheckError) as caught:
        _external_asset_paths(options, config({"synthetic_only": True}))
    assert caught.value.code == "asset_missing"
    assert "teacher_model" in str(caught.value)


@pytest.mark.parametrize(
    "example_ids",
    [None, [], ["e0", "e0"], ["e0", ""], ["e0", 7]],
)
def test_qwen_heal_preflight_requires_ordered_unique_preregistered_example_ids(
    example_ids: object,
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    raw = {
        "backend": "qwen",
        "mechanism": "exact_cache",
        "variant": "top_surprise",
        "required_stage": "qwen_heal",
        "qwen": {"run_mode": "heal"},
        "task": {"params": {"example_ids": example_ids}},
        "cache": {"width": 2, "block_size": 2},
        "lengths": {"curriculum": [4]},
        "model": {"ffn_dim": 8, "ffn_match_lower": 8, "ffn_match_upper": 8},
    }
    assert "qwen_example_ids_invalid" in validate_raw_scientific_config(
        raw, backend="qwen", mode="heal"
    )


def test_qwen_runtime_data_windows_must_match_preregistered_example_order() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _validate_job_data,
    )

    config = _qwen_adapter_job("a" * 64)["canonical_config"]
    reordered = QwenJobData(
        train_microbatches=(
            _batch("e1", (2, 1, 0)),
            _batch("e0", (0, 1, 2)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": "c" * 64},
    )
    with pytest.raises(
        QwenRuntimeConfigurationError, match="example_window_mismatch"
    ) as caught:
        _validate_job_data(reordered, config=config)
    assert caught.value.code == "example_window_mismatch"


def test_qwen_run_job_requires_bound_runtime_but_is_runner_discoverable() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        run_job,
    )
    from research.kmd2_ablation.runner import load_backend_dispatcher

    assert load_backend_dispatcher("qwen") is run_job
    with pytest.raises(QwenRuntimeConfigurationError, match="runtime_required") as caught:
        run_job({"backend": "qwen"})
    assert caught.value.code == "runtime_required"


def test_bound_qwen_dispatcher_orchestrates_heal_resume_checkpoint_and_diagnostics(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        build_job_dispatcher,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    from research.kmd2_ablation.runner import _expand_jobs

    base_job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    expanded = _expand_jobs(
        _RunnerQwenHealConfig(base_job["canonical_config"], seeds=(17, 19, 23)),
        "exact_cache.selector.exact_outer",
        asset_hashes={
            "checkpoint": hashes["checkpoint"],
            "data": hashes["data"],
        },
    )
    job = next(
        item
        for item in expanded
        if item["arm_id"] == "surprise" and item["seed"] == 17
    )
    job_before = copy.deepcopy(job)
    runtime = {
        **paths,
        "output": tmp_path / "results",
        "student_device": "cpu",
        "teacher_device": "cpu",
        "dtype": "float32",
        "asset_hashes": hashes,
        "resume": True,
    }
    resume_path = (
        runtime["output"] / "checkpoints" / job["job_id"] / "latest.pt"
    )
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"resume checkpoint marker")
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"], "example_count": 2},
    )
    events: list[object] = []
    saved_metadata: list[object] = []

    def load_data(**kwargs: object) -> QwenJobData:
        events.append(("data", kwargs["asset"].sha256))
        return data

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        events.append(
            (
                "arm",
                spec.arm,
                spec.pre_replacement_checkpoint_sha256,
                spec.trainable_names,
            )
        )
        model = _HealModel()
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(spec.trainable_names)),
            assets=(),
        )

    def load_teacher(**kwargs: object) -> torch.nn.Module:
        events.append(("teacher", kwargs["asset"].sha256))
        return _HealTeacher()

    def save_checkpoint(path: Path, **kwargs: object) -> Path:
        saved_metadata.append(kwargs["metadata"])
        events.append(
            (
                "checkpoint",
                path,
                kwargs["metadata"].step,
                kwargs["metadata"].tokens_seen,
            )
        )
        return path

    def load_checkpoint(path: Path, **_kwargs: object) -> SimpleNamespace:
        events.append(("resume", path))
        return SimpleNamespace(
            job_id=job["job_id"],
            pairing_id=job["pairing_id"],
            arm=job["arm_id"],
            step=0,
            tokens_seen=0,
        )

    def evaluate(**kwargs: object) -> dict[str, object]:
        events.append(("evaluate", kwargs["loaded_arm"].arm))
        return {
            "metrics": {"token_accuracy": 1.0, "eval_loss": 0.25},
            "recurrent_state": {"elements": 9, "bytes": 36},
            "exact_cache": _exact_cache_result_diagnostics(),
        }

    def reset_peak_vram(device: str) -> None:
        events.append(("reset_peak_vram", device))

    ticks = iter((10.0, 12.0))
    dispatcher = build_job_dispatcher(
        runtime,
        dependencies={
            "load_data": load_data,
            "load_arm": load_arm,
            "load_teacher": load_teacher,
            "load_checkpoint": load_checkpoint,
            "save_checkpoint": save_checkpoint,
            "evaluate": evaluate,
            "monotonic": lambda: next(ticks),
            "reset_peak_vram": reset_peak_vram,
            "peak_vram_bytes": lambda _device: 0,
        },
    )
    payload = dispatcher(job)

    assert job == job_before
    assert [event[0] for event in events] == [
        "data",
        "arm",
        "teacher",
        "resume",
        "reset_peak_vram",
        "checkpoint",
        "evaluate",
    ]
    assert events[3] == ("resume", resume_path)
    assert events[4] == ("reset_peak_vram", "cpu")
    assert events[1] == (
        "arm",
        "surprise",
        hashes["checkpoint"],
        ("memory_weight", "cache_amplitude"),
    )
    assert payload["metrics"] == {"token_accuracy": 1.0, "eval_loss": 0.25}
    assert payload["counts"] == {
        "nonfinite_loss": 0,
        "nonfinite_gradient": 0,
        "skipped_steps": 0,
    }
    assert len(payload["loss_curves"]["total"]) == 1
    assert payload["parameters"] == {"trainable": 10, "total": 11}
    assert payload["recurrent_state"] == {"elements": 9, "bytes": 36}
    assert payload["performance"]["wall_time_seconds"] == 2.0
    assert payload["performance"]["tokens_per_second"] == 3.0
    assert payload["identities"]["checkpoint"]["sha256"] == hashes["checkpoint"]
    assert payload["identities"]["data"]["sha256"] == hashes["data"]
    assert payload["identities"]["paired_starts"] == {
        "native": hashes["checkpoint"],
        "recency": hashes["checkpoint"],
        "surprise": hashes["checkpoint"],
    }
    assert saved_metadata[0].source_hashes["asset:model"] == hashes["model"]
    assert saved_metadata[0].source_hashes["asset:checkpoint"] == hashes["checkpoint"]
    assert saved_metadata[0].source_hashes["asset:data"] == hashes["data"]
    assert saved_metadata[0].source_hashes["asset:teacher_model"] == hashes[
        "teacher_model"
    ]
    assert payload["exact_cache"] == _exact_cache_result_diagnostics()
    assert "runtime" not in payload
    assert str(tmp_path) not in json.dumps(payload)


def test_qwen_dispatcher_scopes_paired_python_and_torch_rng_on_success_and_interrupt(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        build_job_dispatcher,
        derive_three_arm_pairing,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.pt",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    runtime = {
        **paths,
        "output": tmp_path / "results",
        "student_device": "cpu",
        "dtype": "float32",
        "asset_hashes": hashes,
        "resume": False,
    }
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"]},
    )

    def job_for(arm: str, seed: int) -> dict[str, object]:
        job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
        job["job_id"] = f"rng-{arm}-{seed}"
        job["seed"] = seed
        job["arm_id"] = arm
        params = job["canonical_config"]["task"]["params"]
        params.update(
            {
                "objective": "synthetic_only",
                "kl_weight": 0.0,
                "layerwise_weight": 0.0,
            }
        )
        pairing = derive_three_arm_pairing(
            job,
            example_ids=("e0", "e1"),
            pre_replacement_checkpoint_sha256=hashes["checkpoint"],
            data_sha256=hashes["data"],
        )
        job["pairing_id"] = pairing.pairing_id
        return job

    observed: list[tuple[float, tuple[float, ...]]] = []

    def load_data(**_kwargs: object) -> QwenJobData:
        observed.append(
            (
                random.random(),
                tuple(float(value) for value in torch.rand(3).tolist()),
            )
        )
        return data

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        model = _HealModel()
        declared = set(spec.trainable_names)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in declared)
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(declared)),
            assets=(),
        )

    def evaluate(**kwargs: object) -> dict[str, object]:
        result: dict[str, object] = {
            "metrics": {"eval_loss": 0.25, "token_accuracy": 1.0},
            "recurrent_state": {"elements": 9, "bytes": 36},
        }
        if kwargs["loaded_arm"].arm != "native":
            result["exact_cache"] = _exact_cache_result_diagnostics()
        return result

    dispatcher = build_job_dispatcher(
        runtime,
        dependencies={
            "load_data": load_data,
            "load_arm": load_arm,
            "save_checkpoint": lambda path, **_kwargs: path,
            "evaluate": evaluate,
            "monotonic": lambda: 1.0,
            "peak_vram_bytes": lambda _device: 0,
        },
    )
    random.seed(9917)
    torch.manual_seed(9917)
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    for arm in ("native", "recency", "surprise"):
        dispatcher(job_for(arm, 41))
    dispatcher(job_for("native", 42))

    assert observed[0] == observed[1] == observed[2]
    assert observed[3] != observed[0]
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)

    def interrupted_load_data(**_kwargs: object) -> QwenJobData:
        random.random()
        torch.rand(2)
        raise KeyboardInterrupt("interrupt after RNG use")

    interrupted = build_job_dispatcher(
        runtime,
        dependencies={"load_data": interrupted_load_data},
    )
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    with pytest.raises(KeyboardInterrupt, match="interrupt after RNG use"):
        interrupted(job_for("native", 43))
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_bound_qwen_dispatcher_validates_runtime_asset_hashes_before_loading(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import AssetIdentityError
    from research.kmd2_ablation.qwen_training import build_job_dispatcher

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    calls: list[str] = []
    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": tmp_path / "results",
            "student_device": "cpu",
            "teacher_device": "cpu",
            "dtype": "float32",
            "asset_hashes": {**hashes, "checkpoint": "0" * 64},
            "resume": False,
        },
        dependencies={"load_arm": lambda *_args, **_kwargs: calls.append("load")},
    )

    with pytest.raises(AssetIdentityError, match="asset_hash_mismatch"):
        dispatcher(job)
    assert calls == []


def test_default_qwen_pt_data_loader_rejects_pickle_execution(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        _default_load_data,
    )

    marker = tmp_path / "data-pickle-executed.txt"
    data_path = tmp_path / "malicious-windows.pt"
    torch.save(_PickleMarkerPayload(marker, "data"), data_path)
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _default_load_data(asset=asset)

    assert caught.value.code == "data_window_invalid"
    assert not marker.exists()


def test_default_qwen_data_loader_normalizes_empty_sparse_stale_positions(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import _default_load_data

    data_path = tmp_path / "empty-stale-positions.json"
    data_path.write_text(
        json.dumps(
            {
                "train": [{"example_id": "e0", "input_ids": [0, 1, 2]}],
                "eval": [
                    {
                        "example_id": "eval0",
                        "input_ids": [0, 1, 2],
                        "query_mask": [False, False, True],
                        "source_spans": [[-1, -1], [-1, -1], [0, 1]],
                        "stale_positions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    data = _default_load_data(asset=asset)

    stale_positions = data.eval_microbatches[0]["stale_positions"]
    assert isinstance(stale_positions, torch.Tensor)
    assert stale_positions.dtype == torch.int64
    assert stale_positions.shape == (0, 3)


def test_bound_qwen_dispatcher_rejects_inconsistent_resume_identity(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        build_job_dispatcher,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    output = tmp_path / "results"
    resume_path = output / "checkpoints" / "job-surprise" / "latest.pt"
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"resume")
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"], "example_count": 2},
    )

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        return LoadedQwenArm(
            model=_HealModel(),
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(spec.trainable_names)),
            assets=(),
        )

    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": output,
            "student_device": "cpu",
            "teacher_device": "cpu",
            "dtype": "float32",
            "asset_hashes": hashes,
            "resume": True,
        },
        dependencies={
            "load_data": lambda **_kwargs: data,
            "load_arm": load_arm,
            "load_teacher": lambda **_kwargs: _HealTeacher(),
            "load_checkpoint": lambda *_args, **_kwargs: SimpleNamespace(
                job_id="wrong-job",
                pairing_id=job["pairing_id"],
                arm="surprise",
                step=1,
                tokens_seen=6,
            ),
            "evaluate": lambda **_kwargs: {
                "metrics": {"token_accuracy": 1.0},
                "recurrent_state": {"elements": 9, "bytes": 36},
                "exact_cache": _exact_cache_result_diagnostics(),
            },
        },
    )

    with pytest.raises(
        QwenRuntimeConfigurationError, match="resume_identity_mismatch"
    ) as caught:
        dispatcher(job)
    assert caught.value.code == "resume_identity_mismatch"


def test_default_qwen_dependencies_complete_two_layer_annotated_ruler_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import (
        LoadedQwenArm,
        _recency_cache_type,
    )
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn
    from research.kmd2_ablation.qwen_training import (
        build_job_dispatcher,
        derive_three_arm_pairing,
    )
    from research.kmd2_ablation.results import _EXACT_CACHE_FIELDS
    from research.kmd2_ablation.summarize import _normalize_evaluation
    from research.kmd2_ablation.tasks.ruler import RulerCell, RulerEpisode

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    layer_config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )
    exact_config = CacheConfig(
        width=2,
        block_size=64,
        score="exact_outer",
        read="rmsnorm",
        storage_dtype="fp32",
    )

    class Block(torch.nn.Module):
        def __init__(self, linear_attn: torch.nn.Module) -> None:
            super().__init__()
            self.linear_attn = linear_attn

    class Backbone(torch.nn.Module):
        def __init__(self, linear_attn: tuple[torch.nn.Module, ...]) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [Block(module) for module in linear_attn]
            )

    class RulerModel(torch.nn.Module):
        def __init__(self, arm: str) -> None:
            super().__init__()
            self.config = layer_config
            linear_attn: list[torch.nn.Module] = []
            for layer_index in range(2):
                native = KMD2NativeAttn(layer_config, layer_idx=layer_index)
                installed: torch.nn.Module = native
                if arm != "native":
                    exact = KMD2ExactCacheAttn.from_native(
                        native,
                        model_config=layer_config,
                        cache_config=exact_config,
                    )
                    if arm == "recency":
                        exact.__class__ = _recency_cache_type()
                        exact.cache_config = dataclasses.replace(
                            exact_config,
                            score="recency",
                        )
                    installed = exact
                linear_attn.append(installed)
            self.embedding = torch.nn.Embedding(13, 12)
            self.model = Backbone(tuple(linear_attn))
            self.lm_head = torch.nn.Linear(12, 13)
            self.embedding.requires_grad_(False)
            self.lm_head.requires_grad_(False)

        def gradient_checkpointing_enable(self) -> None:
            return None

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
        ) -> SimpleNamespace:
            assert use_cache is False
            hidden = self.embedding(input_ids)
            memory = hidden
            for layer in self.model.layers:
                memory = layer.linear_attn(memory)
            logits = self.lm_head(memory)
            hidden_states = (hidden, memory) if output_hidden_states else None
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    prototype = RulerModel("surprise")
    cache_basenames = {
        "cache_gamma_q",
        "cache_gamma_k",
        "cache_sink_logit",
        "cache_amplitude",
    }
    memory_names = tuple(
        name
        for name, parameter in prototype.named_parameters()
        if parameter.requires_grad and name.rsplit(".", 1)[-1] not in cache_basenames
    )
    cache_names = tuple(
        name
        for name, parameter in prototype.named_parameters()
        if parameter.requires_grad and name.rsplit(".", 1)[-1] in cache_basenames
    )
    assert memory_names and set(name.rsplit(".", 1)[-1] for name in cache_names) == cache_basenames

    cell = RulerCell(context_length=512, needles=16, queries=1)
    tokens = tuple(index % 13 for index in range(514))
    answer_token = tokens[513]
    episode = RulerEpisode(
        episode_id="e" * 64,
        seed=41,
        example_index=0,
        cell=cell,
        input_ids=tokens,
        prompt_end=513,
        answers=(str(answer_token),),
        answer_token_ids=((answer_token,),),
        answer_spans=((513, 514),),
        source_spans=((1, 2),),
        depth_strata=("early",),
        query_keys=("key",),
    )
    target_digest = hashlib.sha256(
        json.dumps([[answer_token]], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    query_mask = torch.zeros(1, len(tokens), dtype=torch.bool)
    query_mask[0, 513] = True
    source_spans = torch.full((1, len(tokens), 2), -1, dtype=torch.int64)
    source_spans[0, 513] = torch.tensor([1, 2])
    stale_positions = torch.tensor([[0, 513, 3]], dtype=torch.int64)
    ruler_metadata = (
        {
            "cell_id": cell.cell_id,
            "context_length": cell.context_length,
            "needles": cell.needles,
            "queries": cell.queries,
            "depth_stratum": "early",
            "example_id": "eval0",
            "episode_id": episode.episode_id,
            "evaluation_mode": "teacher_forced",
            "evidence_scope": "feasibility",
            "seed": episode.seed,
            "example_index": episode.example_index,
            "prompt_end": episode.prompt_end,
            "answers": episode.answers,
            "answer_token_ids": episode.answer_token_ids,
            "answer_spans": episode.answer_spans,
            "source_spans": episode.source_spans,
            "depth_strata": episode.depth_strata,
            "query_keys": episode.query_keys,
            "target_digest": target_digest,
            "paired_interval": {
                "kind": "paired_seed_interval",
                "status": "feasibility_only",
            },
        },
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "windows.pt",
    }
    paths["model"].write_bytes(b"model")
    paths["checkpoint"].write_bytes(b"checkpoint")
    torch.save(
        {
            "train": [
                {"example_id": "e0", "input_ids": [0, 1, 2]},
                {"example_id": "e1", "input_ids": [2, 1, 0]},
            ],
            "eval": [
                {
                    "example_id": "eval0",
                    "input_ids": list(tokens),
                    "labels": list(tokens),
                    "query_mask": query_mask,
                    "source_spans": source_spans,
                    "stale_positions": stale_positions,
                    "ruler_metadata": ruler_metadata,
                }
            ],
        },
        paths["data"],
    )
    hashes = {name: _sha256(path) for name, path in paths.items()}

    def job_for(arm: str) -> dict[str, object]:
        job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
        job["job_id"] = f"ruler-{arm}"
        job["seed"] = 41
        job["arm_id"] = arm
        config = job["canonical_config"]
        config["cache"].update(
            {
                "width": 2,
                "block_size": 64,
            }
        )
        params = config["task"]["params"]
        params.update(
            {
                "objective": "synthetic_only",
                "ce_weight": 1.0,
                "kl_weight": 0.0,
                "layerwise_weight": 0.0,
                "memory_parameter_names": list(memory_names),
                "cache_parameter_names": list(cache_names),
            }
        )
        pairing = derive_three_arm_pairing(
            job,
            example_ids=("e0", "e1"),
            pre_replacement_checkpoint_sha256=hashes["checkpoint"],
            data_sha256=hashes["data"],
        )
        job["pairing_id"] = pairing.pairing_id
        return job

    loaded_models: dict[str, RulerModel] = {}

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        model = RulerModel(spec.arm)
        loaded_models[spec.arm] = model
        declared = set(spec.trainable_names)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in declared)
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0, 1),
            trainable_names=tuple(sorted(declared)),
            assets=(),
        )

    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": tmp_path / "results",
            "student_device": "cpu",
            "dtype": "float32",
            "asset_hashes": hashes,
            "resume": False,
        },
        dependencies={
            "load_arm": load_arm,
            "save_checkpoint": lambda path, **_kwargs: path,
            "monotonic": lambda: 1.0,
            "peak_vram_bytes": lambda _device: 0,
        },
    )

    for arm in ("native", "recency", "surprise"):
        job = job_for(arm)
        payload = dispatcher(job)
        assert len(payload["evaluations"]) == 1
        row = payload["evaluations"][0]
        normalized = _normalize_evaluation(
            row,
            record={"job_id": job["job_id"], "seed": 41, "arm_id": arm},
            index=0,
        )
        assert normalized["evidence_scope"] == "feasibility"
        assert normalized["source_spans"] == [[1, 2]]
        assert normalized["target_digest"] == target_digest
        assert normalized["denominator"] == 1
        assert normalized["episode_exact"] == (normalized["numerator"] == 1)
        assert normalized["seed"] == 41
        assert normalized["arm_id"] == arm
        assert isinstance(normalized["cache_diagnostics"], dict)
        assert normalized["paired_interval"] == {
            "kind": "paired_seed_interval",
            "status": "feasibility_only",
        }
        if arm == "native":
            assert "exact_cache" not in payload
            assert normalized["cache_diagnostics"] == {"active": False}
        else:
            assert set(payload["exact_cache"]) == _EXACT_CACHE_FIELDS
            assert normalized["cache_diagnostics"]["active"] is True
            diagnostics = [
                layer.linear_attn.last_cache_diagnostics
                for layer in loaded_models[arm].model.layers
            ]
            assert all(item is not None for item in diagnostics)
            assert sum(item.persistent_bytes for item in diagnostics) == 328
            assert all(not hasattr(item, "blocks") for item in diagnostics)
            assert all(not hasattr(item, "update_scores") for item in diagnostics)
            assert payload["exact_cache"]["persistent_bytes"] == 328
            assert payload["exact_cache"]["block_bytes"] == 10_496
            assert payload["exact_cache"]["score_statistics"]["count"] == 2_056


def test_default_qwen_cache_evaluator_streams_sparse_32k_without_quadratic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_exact_cache import (
        CacheBlockObservation,
        KMD2ExactCacheAttn,
        QwenBoundedCacheDiagnostics,
    )
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _cache_amplitudes,
        _default_evaluate,
        _validate_evaluation_annotations,
    )

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    steps = 32_768
    block_size = 4_096
    model_config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )
    cache_config = CacheConfig(
        width=1,
        block_size=block_size,
        score="exact_outer",
        read="rmsnorm",
        storage_dtype="fp32",
    )

    class StreamingProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            native = KMD2NativeAttn(model_config, layer_idx=0)
            self.cache_layer = KMD2ExactCacheAttn.from_native(
                native,
                model_config=model_config,
                cache_config=cache_config,
            )
            self.emitted_blocks = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
        ) -> SimpleNamespace:
            assert output_hidden_states is False
            assert use_cache is False
            assert self.cache_layer._retain_full_cache_diagnostics is False
            observer = self.cache_layer._cache_diagnostic_observer
            assert callable(observer)
            batch_size, sequence_length = input_ids.shape
            heads = self.cache_layer.H
            width = self.cache_layer.cache_config.width
            persistent_positions = torch.full(
                (batch_size, heads, width), 5, dtype=torch.int64
            )
            persistent_scores = torch.ones(
                batch_size, heads, width, dtype=torch.float32
            )
            persistent_valid = torch.ones(
                batch_size, heads, width, dtype=torch.bool
            )
            persistent_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    persistent_positions,
                    persistent_scores,
                    persistent_valid,
                )
            )
            self.emitted_blocks = 0
            for block_start in range(0, sequence_length, block_size):
                block_stop = min(sequence_length, block_start + block_size)
                block_length = block_stop - block_start
                top1_positions = torch.full(
                    (batch_size, block_length, heads), 5, dtype=torch.int64
                )
                candidate_positions = top1_positions.unsqueeze(-1)
                candidate_valid = torch.ones_like(
                    candidate_positions, dtype=torch.bool
                )
                update_scores = torch.ones(
                    batch_size, block_length, heads, dtype=torch.float32
                )
                unit_metric = torch.ones_like(update_scores)
                zero_metric = torch.zeros_like(update_scores)
                block_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        top1_positions,
                        candidate_positions,
                        candidate_valid,
                        update_scores,
                        unit_metric,
                        zero_metric,
                    )
                )
                observer(
                    CacheBlockObservation(
                        block_start=block_start,
                        block_stop=block_stop,
                        candidate_positions=candidate_positions,
                        candidate_valid=candidate_valid,
                        attention_weights=torch.empty(0),
                        persistent_selected_positions=persistent_positions,
                        top1_positions=top1_positions,
                        attention_entropy=zero_metric,
                        top1_mass=unit_metric,
                        sink_mass=zero_metric,
                        update_scores=update_scores,
                        state_output_norm=unit_metric,
                        cache_output_norm=unit_metric,
                        persistent_bytes=persistent_bytes,
                        block_bytes=block_bytes,
                    )
                )
                self.emitted_blocks += 1
            self.cache_layer.last_cache_diagnostics = QwenBoundedCacheDiagnostics(
                blocks_processed=self.emitted_blocks,
                final_selected_positions=persistent_positions,
                final_selected_scores=persistent_scores,
                final_selected_valid=persistent_valid,
                persistent_bytes=persistent_bytes,
            )
            logits = F.one_hot(input_ids, num_classes=3).to(torch.float32)
            return SimpleNamespace(logits=logits)

    input_ids = (torch.arange(steps, dtype=torch.int64) % 3).unsqueeze(0)
    query_mask = torch.zeros((1, steps), dtype=torch.bool)
    query_mask[0, -1] = True
    source_spans = torch.full((1, steps, 2), -1, dtype=torch.int64)
    source_spans[0, -1] = torch.tensor([5, 6])
    sparse_batch: dict[str, object] = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": ("sparse-32k",),
        "query_mask": query_mask,
        "source_spans": source_spans,
        "stale_positions": torch.tensor([[0, steps - 1, 7]], dtype=torch.int64),
    }
    assert all(
        value.ndim < 2 or tuple(value.shape[-2:]) != (steps, steps)
        for value in sparse_batch.values()
        if isinstance(value, torch.Tensor)
    )
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(sparse_batch,),
        data_identity={"sha256": "a" * 64},
    )
    job = _qwen_adapter_job("a" * 64)
    job["canonical_config"]["task"]["name"] = "bounded-32k-probe"
    model = StreamingProbeModel()
    loaded = LoadedQwenArm(
        model=model,
        arm="surprise",
        job_id="bounded-32k-probe",
        upgraded_indices=(0,),
        trainable_names=(),
        assets=(),
    )

    result = _default_evaluate(
        loaded_arm=loaded,
        data=data,
        job=job,
        runtime={"student_device": "cpu"},
        amplitude_initial=_cache_amplitudes(model),
    )

    diagnostics = model.cache_layer.last_cache_diagnostics
    assert isinstance(diagnostics, QwenBoundedCacheDiagnostics)
    assert diagnostics.blocks_processed == steps // block_size
    assert not hasattr(diagnostics, "blocks")
    assert not hasattr(diagnostics, "update_scores")
    assert sum(
        tensor.numel()
        for tensor in (
            diagnostics.final_selected_positions,
            diagnostics.final_selected_scores,
            diagnostics.final_selected_valid,
        )
    ) == 6
    exact_cache = result["exact_cache"]
    assert exact_cache["score_statistics"]["count"] == steps * 2
    assert len(exact_cache["selected_index_sample"]) == 2
    assert max(
        len(value) for value in exact_cache.values() if isinstance(value, list)
    ) <= 32
    assert "observation_logs" not in exact_cache
    assert model.emitted_blocks == 8

    dense_steps = 4_097
    dense_query_mask = torch.zeros((1, dense_steps), dtype=torch.bool)
    dense_query_mask[0, -1] = True
    dense_source_spans = torch.full((1, dense_steps, 2), -1, dtype=torch.int64)
    dense_source_spans[0, -1] = torch.tensor([0, 1])
    dense_batch = {
        "input_ids": torch.zeros((1, dense_steps), dtype=torch.int64),
        "query_mask": dense_query_mask,
        "source_spans": dense_source_spans,
        # A meta tensor proves rejection is shape-based without allocating T^2 bytes.
        "stale_mask": torch.empty(
            (1, dense_steps, dense_steps), dtype=torch.bool, device="meta"
        ),
    }
    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _validate_evaluation_annotations(
            dense_batch,
            job=job,
            require_cache=True,
        )
    assert caught.value.code == "cache_annotations_invalid"
    assert "use stale_positions" in str(caught.value)


def test_default_qwen_cache_evaluator_rejects_missing_annotations_actionably(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _default_evaluate,
    )

    loaded = LoadedQwenArm(
        model=_HealModel(),
        arm="surprise",
        job_id="missing-annotations",
        upgraded_indices=(0,),
        trainable_names=("memory_weight", "cache_amplitude"),
        assets=(),
    )
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(_batch("eval0", (0, 1, 2)),),
        data_identity={"sha256": "a" * 64},
    )
    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _default_evaluate(
            loaded_arm=loaded,
            data=data,
            job=_qwen_adapter_job("a" * 64),
            runtime={"student_device": "cpu"},
            amplitude_initial=[0.25],
        )
    assert caught.value.code == "cache_annotations_missing"
    assert "query_mask" in str(caught.value)


@pytest.mark.parametrize(
    ("extra_inputs", "expected_code"),
    [
        ({"attention_mask": torch.tensor([[1, 1, 0]])}, "padding_unsupported"),
        ({"position_ids": torch.tensor([[0, 1, 0]])}, "position_reset"),
    ],
)
def test_default_qwen_evaluator_guards_padding_and_position_resets_before_forward(
    extra_inputs: dict[str, torch.Tensor], expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_exact_cache import FullRecomputeCallError
    from research.kmd2_ablation.qwen_training import QwenJobData, _default_evaluate

    class EvalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            del attention_mask, position_ids
            assert output_hidden_states is False
            assert use_cache is False
            self.forward_calls += 1
            logits = F.one_hot(input_ids, num_classes=3).to(torch.float32)
            return SimpleNamespace(logits=logits)

    model = EvalModel()
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    source_spans = torch.full((1, 3, 2), -1, dtype=torch.int64)
    source_spans[0, 2] = torch.tensor([0, 1])
    batch: dict[str, object] = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": ("eval0",),
        "query_mask": torch.tensor([[False, False, True]]),
        "source_spans": source_spans,
        "stale_mask": torch.zeros((1, 3, 3), dtype=torch.bool),
        **extra_inputs,
    }
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(batch,),
        data_identity={"sha256": "a" * 64},
    )
    job = _qwen_adapter_job("a" * 64)
    job["canonical_config"]["task"]["name"] = "guard-probe"
    loaded = LoadedQwenArm(
        model=model,
        arm="native",
        job_id="guarded-eval",
        upgraded_indices=(0,),
        trainable_names=(),
        assets=(),
    )

    with pytest.raises(FullRecomputeCallError) as caught:
        _default_evaluate(
            loaded_arm=loaded,
            data=data,
            job=job,
            runtime={"student_device": "cpu"},
        )

    assert caught.value.code == expected_code
    assert model.forward_calls == 0
