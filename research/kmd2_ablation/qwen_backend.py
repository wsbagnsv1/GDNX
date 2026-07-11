"""Strict, injectable Qwen loading for paired KMD-2 heal experiments.

Importing this module never imports Transformers or loads external assets.  The
heavy model loader and the production upgrade manager are resolved only when a
real execution calls :func:`load_qwen_arm`; tests can inject small fakes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import torch


_ARMS = ("native", "recency", "surprise")


class AssetIdentityError(ValueError):
    """An external asset does not match its preregistered identity."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ExternalAssetIdentity:
    """Expected identity for one external file or directory tree."""

    name: str
    path: Path | str | os.PathLike[str]
    kind: str
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("asset name must be a nonempty string")
        if self.kind not in {"file", "directory"}:
            raise ValueError("asset kind must be 'file' or 'directory'")
        try:
            path = Path(self.path)
        except TypeError as error:
            raise TypeError("asset path must be path-like") from error
        object.__setattr__(self, "path", path)
        if self.size_bytes is not None and (
            type(self.size_bytes) is not int or self.size_bytes < 0
        ):
            raise ValueError("asset size_bytes must be a nonnegative integer or None")
        if self.sha256 is not None:
            if (
                type(self.sha256) is not str
                or len(self.sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.sha256)
            ):
                raise ValueError("asset sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class ValidatedAssetIdentity:
    """Resolved measured identity passed to manifests and checkpoints."""

    name: str
    path: Path
    kind: str
    size_bytes: int
    sha256: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_identity(path: Path) -> tuple[int, str]:
    entries: list[tuple[str, int, str]] = []
    total = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise AssetIdentityError(
                "asset_symlink_unsupported",
                f"directory asset {path} contains symlink {child}",
            )
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        file_digest = _hash_file(child)
        entries.append((relative, size, file_digest))
        total += size
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return total, hashlib.sha256(encoded).hexdigest()


def validate_external_assets(
    assets: Sequence[ExternalAssetIdentity],
) -> tuple[ValidatedAssetIdentity, ...]:
    """Validate all identities before any model loader is allowed to execute."""
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
        raise TypeError("assets must be a sequence of ExternalAssetIdentity records")
    seen: set[str] = set()
    validated: list[ValidatedAssetIdentity] = []
    for asset in assets:
        if not isinstance(asset, ExternalAssetIdentity):
            raise TypeError("assets must contain ExternalAssetIdentity records")
        if asset.name in seen:
            raise ValueError(f"duplicate external asset name: {asset.name}")
        seen.add(asset.name)
        path = asset.path.expanduser().resolve()
        if not path.exists():
            raise AssetIdentityError(
                "asset_missing", f"external asset {asset.name!r} is missing: {path}"
            )
        if asset.kind == "file":
            if not path.is_file():
                raise AssetIdentityError(
                    "asset_kind_mismatch",
                    f"external asset {asset.name!r} must be a file: {path}",
                )
            size = path.stat().st_size
            digest = _hash_file(path)
        else:
            if not path.is_dir():
                raise AssetIdentityError(
                    "asset_kind_mismatch",
                    f"external asset {asset.name!r} must be a directory: {path}",
                )
            size, digest = _directory_identity(path)
        if asset.size_bytes is not None and size != asset.size_bytes:
            raise AssetIdentityError(
                "asset_size_mismatch",
                f"external asset {asset.name!r} expected {asset.size_bytes} bytes, got {size}",
            )
        if asset.sha256 is not None and digest != asset.sha256:
            raise AssetIdentityError(
                "asset_hash_mismatch",
                f"external asset {asset.name!r} SHA-256 does not match",
            )
        validated.append(
            ValidatedAssetIdentity(
                name=asset.name,
                path=path,
                kind=asset.kind,
                size_bytes=size,
                sha256=digest,
            )
        )
    return tuple(sorted(validated, key=lambda item: item.name))


@dataclass(frozen=True)
class QwenArmLoadSpec:
    """Execution-only inputs needed to construct one paired Qwen arm."""

    arm: str
    job_id: str
    model_asset: ExternalAssetIdentity
    native_checkpoint: ExternalAssetIdentity | None
    data_asset: ExternalAssetIdentity
    cache_resume: ExternalAssetIdentity | None
    trainable_names: tuple[str, ...]
    pre_replacement_checkpoint_sha256: str
    model_loader_kwargs: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of: {', '.join(_ARMS)}")
        if type(self.job_id) is not str or not self.job_id:
            raise ValueError("job_id must be a nonempty string")
        for name in ("model_asset", "data_asset"):
            if not isinstance(getattr(self, name), ExternalAssetIdentity):
                raise TypeError(f"{name} must be an ExternalAssetIdentity")
        for name in ("native_checkpoint", "cache_resume"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ExternalAssetIdentity):
                raise TypeError(f"{name} must be an ExternalAssetIdentity or None")
        if self.native_checkpoint is None:
            raise ValueError(
                "native_checkpoint_required: Qwen heal arms require a native checkpoint"
            )
        if self.arm == "native" and self.cache_resume is not None:
            raise ValueError("native continuation cannot load a cache resume")
        if type(self.trainable_names) is not tuple or not self.trainable_names:
            raise ValueError("trainable_names must be a nonempty tuple")
        if any(type(name) is not str or not name for name in self.trainable_names):
            raise ValueError("trainable_names must contain nonempty strings")
        if len(set(self.trainable_names)) != len(self.trainable_names):
            raise ValueError("trainable_names must not contain duplicates")
        digest = self.pre_replacement_checkpoint_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "pre_replacement_checkpoint_sha256 must be lowercase SHA-256"
            )
        if not isinstance(self.model_loader_kwargs, Mapping):
            raise TypeError("model_loader_kwargs must be a mapping")
        frozen_kwargs = MappingProxyType(dict(self.model_loader_kwargs))
        if any(type(key) is not str or not key for key in frozen_kwargs):
            raise ValueError("model_loader_kwargs keys must be nonempty strings")
        object.__setattr__(self, "model_loader_kwargs", frozen_kwargs)


@dataclass(frozen=True)
class LoadedQwenArm:
    """A constructed arm plus the exact identity and trainability record."""

    model: torch.nn.Module
    arm: str
    job_id: str
    upgraded_indices: tuple[int, ...]
    trainable_names: tuple[str, ...]
    assets: tuple[ValidatedAssetIdentity, ...]


class PairingContractError(ValueError):
    """The three Qwen heal arms are not a mechanically paired comparison."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _freeze_json(value: object, context: str) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain nonfinite values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if type(key) is not str or not key:
                raise ValueError(f"{context} keys must be nonempty strings")
            frozen[key] = _freeze_json(value[key], f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{context} must contain only JSON-compatible values")


@dataclass(frozen=True)
class QwenHealArmContract:
    """All scientific identity fields for one arm of a paired Qwen heal."""

    arm: str
    job_id: str
    seed: int
    pre_replacement_checkpoint_sha256: str
    data_sha256: str
    example_ids: tuple[str, ...]
    token_budget: int
    update_budget: int
    curriculum: tuple[int, ...]
    optimizer: Mapping[str, object]
    schedule: Mapping[str, object]
    stopping: Mapping[str, object]
    eval_cells: tuple[str, ...]
    cache_match: Mapping[str, object] | None
    selection_policy: str | None

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of: {', '.join(_ARMS)}")
        if type(self.job_id) is not str or not self.job_id:
            raise ValueError("job_id must be a nonempty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        digest = self.pre_replacement_checkpoint_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "pre_replacement_checkpoint_sha256 must be lowercase SHA-256"
            )
        data_digest = self.data_sha256
        if (
            type(data_digest) is not str
            or len(data_digest) != 64
            or any(character not in "0123456789abcdef" for character in data_digest)
        ):
            raise ValueError("data_sha256 must be lowercase SHA-256")
        for field_name in ("example_ids", "eval_cells"):
            value = getattr(self, field_name)
            if type(value) is not tuple or not value:
                raise ValueError(f"{field_name} must be a nonempty tuple")
            if any(type(item) is not str or not item for item in value):
                raise ValueError(f"{field_name} must contain nonempty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{field_name} must not contain duplicates")
        for field_name in ("token_budget", "update_budget"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            type(self.curriculum) is not tuple
            or not self.curriculum
            or any(type(length) is not int or length < 1 for length in self.curriculum)
            or tuple(sorted(set(self.curriculum))) != self.curriculum
        ):
            raise ValueError("curriculum must be a strictly increasing tuple of lengths")
        for field_name in ("optimizer", "schedule", "stopping"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{field_name} must be a nonempty mapping")
            object.__setattr__(self, field_name, _freeze_json(value, field_name))

        if self.arm == "native":
            if self.cache_match is not None or self.selection_policy is not None:
                raise ValueError("native arm cannot declare a cache policy")
            return
        if not isinstance(self.cache_match, Mapping) or not self.cache_match:
            raise ValueError("cache arms require a nonempty cache_match mapping")
        frozen_cache = _freeze_json(self.cache_match, "cache_match")
        assert isinstance(frozen_cache, Mapping)
        required = {
            "width",
            "block_size",
            "read",
            "read_init",
            "storage_dtype",
            "lr_cache",
        }
        missing = sorted(required - set(frozen_cache))
        if missing:
            raise ValueError(
                "cache_match is missing matched settings: " + ", ".join(missing)
            )
        object.__setattr__(self, "cache_match", frozen_cache)
        if self.arm == "recency" and self.selection_policy != "recency":
            raise ValueError("recency arm requires selection_policy='recency'")
        if self.arm == "surprise" and (
            type(self.selection_policy) is not str
            or not self.selection_policy
            or self.selection_policy == "recency"
        ):
            raise ValueError("surprise arm requires a non-recency selection policy")


@dataclass(frozen=True)
class PairedQwenHealContract:
    """Validated native/recency/surprise comparison with a shared hash ID."""

    pairing_id: str
    arms: tuple[QwenHealArmContract, ...]
    canonical_bytes: bytes
    example_ids: tuple[str, ...]


def validate_three_arm_pairing(
    arms: Sequence[QwenHealArmContract],
) -> PairedQwenHealContract:
    """Require exact byte/checkpoint/data/budget pairing across all three arms."""
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise TypeError("arms must be a sequence of QwenHealArmContract records")
    if len(arms) != 3 or any(not isinstance(arm, QwenHealArmContract) for arm in arms):
        raise PairingContractError(
            "pairing_arm_set", "pairing requires exactly three Qwen arm records"
        )
    by_arm = {arm.arm: arm for arm in arms}
    if len(by_arm) != 3 or set(by_arm) != set(_ARMS):
        raise PairingContractError(
            "pairing_arm_set", "pairing requires native, recency, and surprise once each"
        )
    ordered = tuple(by_arm[name] for name in _ARMS)
    native, recency, surprise = ordered
    shared_fields = (
        "seed",
        "pre_replacement_checkpoint_sha256",
        "data_sha256",
        "example_ids",
        "token_budget",
        "update_budget",
        "curriculum",
        "optimizer",
        "schedule",
        "stopping",
        "eval_cells",
    )
    for field_name in shared_fields:
        expected = getattr(native, field_name)
        mismatched = [
            arm.arm for arm in ordered[1:] if getattr(arm, field_name) != expected
        ]
        if mismatched:
            raise PairingContractError(
                "pairing_mismatch",
                f"{field_name} differs for arm(s): {', '.join(mismatched)}",
            )
    if recency.cache_match != surprise.cache_match:
        raise PairingContractError(
            "cache_match_mismatch",
            "capacity/read/gate/cache optimizer settings differ between recency and surprise",
        )
    payload = {
        "cache_match": dict(recency.cache_match or {}),
        "curriculum": list(native.curriculum),
        "eval_cells": list(native.eval_cells),
        "example_ids": list(native.example_ids),
        "optimizer": dict(native.optimizer),
        "policies": {
            arm.arm: arm.selection_policy for arm in ordered
        },
        "pre_replacement_checkpoint_sha256": native.pre_replacement_checkpoint_sha256,
        "data_sha256": native.data_sha256,
        "schedule": dict(native.schedule),
        "seed": native.seed,
        "stopping": dict(native.stopping),
        "token_budget": native.token_budget,
        "update_budget": native.update_budget,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return PairedQwenHealContract(
        pairing_id=hashlib.sha256(canonical).hexdigest(),
        arms=ordered,
        canonical_bytes=canonical,
        example_ids=native.example_ids,
    )


def _default_base_model_loader(path: Path, **kwargs: object) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    return AutoModelForCausalLM.from_pretrained(str(path), **kwargs)


def _default_manager_factory(model: torch.nn.Module, model_config: object) -> object:
    from gdn3.gdn3_upgrade import GDN3UpgradeManager

    return GDN3UpgradeManager(model, model_config)


def _intended_model_dtype(
    model: torch.nn.Module, loader_kwargs: Mapping[str, object]
) -> torch.dtype:
    requested = loader_kwargs.get("torch_dtype")
    candidates = [requested, getattr(model, "dtype", None)]
    candidates.extend(
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    )
    for candidate in candidates:
        if isinstance(candidate, torch.dtype) and torch.empty(
            (), dtype=candidate
        ).is_floating_point():
            return candidate
    raise TypeError("loaded Qwen model must expose an intended floating-point dtype")


def _load_tensor_mapping(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if checkpoint is None:
        return {}
    loaded = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("state_dict"), Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise TypeError("native checkpoint must contain a tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if type(name) is not str or not name:
            raise TypeError("native checkpoint names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"native checkpoint value {name!r} is not a tensor")
        result[name] = tensor
    return result


def _validate_indices(model: object, raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("upgrade manager must return a sequence of layer indices")
    indices = tuple(raw)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("upgrade manager returned an invalid layer index")
    if len(set(indices)) != len(indices) or not indices:
        raise ValueError("upgrade manager must return unique upgraded layer indices")
    try:
        layer_count = len(model.model.layers)
    except (AttributeError, TypeError) as error:
        raise TypeError("model must expose model.layers") from error
    if any(index >= layer_count for index in indices):
        raise ValueError("upgrade manager returned an out-of-range layer index")
    return indices


def _default_native_installer(
    *,
    model: torch.nn.Module,
    manager: object,
    model_config: object,
    cache_config: object,
    native_checkpoint: Path | None,
    cache_resume: Path | None,
    expected_job_id: str,
    target_dtype: torch.dtype,
) -> tuple[int, ...]:
    del model_config, cache_config, expected_job_id
    if cache_resume is not None:
        raise ValueError("native continuation cannot load a cache resume")
    prior = os.environ.get("GDN3_KMD2_NATIVE")
    os.environ["GDN3_KMD2_NATIVE"] = "1"
    try:
        apply_upgrade = getattr(manager, "apply_upgrade", None)
        if not callable(apply_upgrade):
            raise TypeError("manager must expose apply_upgrade()")
        indices = _validate_indices(model, apply_upgrade())
        from gdn3.kmd2_native import KMD2NativeAttn

        prefixes: list[str] = []
        named_modules = dict(model.named_modules())
        for index in indices:
            module = model.model.layers[index].linear_attn
            if type(module) is not KMD2NativeAttn:
                raise TypeError(f"upgraded layer {index} is not an actual KMD2NativeAttn")
            module.to(dtype=target_dtype)
            names = [name for name, candidate in named_modules.items() if candidate is module]
            if len(names) != 1:
                raise ValueError(f"upgraded layer {index} has no unique model name")
            prefixes.append(names[0] + ".")

        checkpoint = _load_tensor_mapping(native_checkpoint)
        state = model.state_dict()
        targets: dict[str, torch.Tensor] = {}
        for name, tensor in checkpoint.items():
            if not name.startswith(tuple(prefixes)) or name not in state:
                raise KeyError(
                    f"native checkpoint key {name!r} does not target an upgraded layer"
                )
            target = state[name]
            if target.shape != tensor.shape or target.dtype != tensor.dtype:
                raise ValueError(
                    f"native checkpoint tensor {name!r} shape/dtype does not match"
                )
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"native checkpoint tensor {name!r} is nonfinite")
            targets[name] = target
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(checkpoint[name].to(device=target.device))
        return indices
    finally:
        if prior is None:
            os.environ.pop("GDN3_KMD2_NATIVE", None)
        else:
            os.environ["GDN3_KMD2_NATIVE"] = prior


_RECENCY_CACHE_TYPE: type[torch.nn.Module] | None = None


def _recency_cache_type() -> type[torch.nn.Module]:
    global _RECENCY_CACHE_TYPE
    if _RECENCY_CACHE_TYPE is not None:
        return _RECENCY_CACHE_TYPE
    from .qwen_exact_cache import KMD2ExactCacheAttn

    class KMD2RecencyCacheAttn(KMD2ExactCacheAttn):
        """Exact-cache read whose persistent admission order is pure recency."""

        def _native_state_and_scores(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta_e: torch.Tensor,
            beta_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state, _ = KMD2ExactCacheAttn._native_state_and_scores(
                self, q, k, v, g, beta_e, beta_w
            )
            scores = torch.arange(
                1,
                k.shape[1] + 1,
                device=k.device,
                dtype=torch.float32,
            ).view(1, -1, 1)
            return state, scores.expand(k.shape[0], -1, k.shape[2])

    KMD2RecencyCacheAttn.__name__ = "KMD2RecencyCacheAttn"
    KMD2RecencyCacheAttn.__qualname__ = "KMD2RecencyCacheAttn"
    KMD2RecencyCacheAttn.__module__ = __name__
    _RECENCY_CACHE_TYPE = KMD2RecencyCacheAttn
    return _RECENCY_CACHE_TYPE


def _default_cache_installer(
    *,
    arm: str,
    model: torch.nn.Module,
    manager: object,
    model_config: object,
    cache_config: object,
    native_checkpoint: Path | None,
    cache_resume: Path | None,
    expected_job_id: str,
    target_dtype: torch.dtype,
) -> tuple[int, ...]:
    from .config import CacheConfig
    from .qwen_exact_cache import KMD2ExactCacheAttn, load_native_then_install

    if not isinstance(cache_config, CacheConfig):
        raise TypeError("cache_config must be a CacheConfig for cache arms")
    if arm == "surprise":
        if cache_config.score != "exact_outer":
            raise ValueError("the initial Qwen surprise arm requires cache.score=exact_outer")
        return load_native_then_install(
            model,
            manager,
            model_config,
            cache_config,
            native_checkpoint,
            cache_resume,
            expected_job_id=expected_job_id,
            target_dtype=target_dtype,
        )
    if arm != "recency":
        raise ValueError("cache installer supports only recency and surprise arms")
    if cache_config.score != "recency":
        raise ValueError("the recency arm requires cache.score=recency")
    exact_config = replace(cache_config, score="exact_outer")
    indices = load_native_then_install(
        model,
        manager,
        model_config,
        exact_config,
        native_checkpoint,
        cache_resume,
        expected_job_id=expected_job_id,
        target_dtype=target_dtype,
    )
    recency_type = _recency_cache_type()
    for index in indices:
        layer = model.model.layers[index].linear_attn
        if type(layer) is not KMD2ExactCacheAttn:
            raise TypeError("recency conversion expected an exact-cache installation")
        layer.__class__ = recency_type
        layer.cache_config = cache_config
    return indices


def _configure_trainables(
    model: torch.nn.Module, declared: tuple[str, ...]
) -> tuple[str, ...]:
    named = dict(model.named_parameters())
    missing = sorted(set(declared) - set(named))
    if missing:
        raise KeyError("declared trainable parameters are missing: " + ", ".join(missing))
    original = {name: parameter.requires_grad for name, parameter in named.items()}
    try:
        selected = set(declared)
        for name, parameter in named.items():
            parameter.requires_grad_(name in selected)
        actual = tuple(sorted(name for name, parameter in named.items() if parameter.requires_grad))
        expected = tuple(sorted(declared))
        if actual != expected:
            raise RuntimeError("trainable parameter set does not match the declaration")
        return actual
    except Exception:
        for name, parameter in named.items():
            parameter.requires_grad_(original[name])
        raise


def load_qwen_arm(
    spec: QwenArmLoadSpec,
    *,
    model_config: object,
    cache_config: object | None,
    base_model_loader: Callable[..., torch.nn.Module] | None = None,
    manager_factory: Callable[[torch.nn.Module, object], object] | None = None,
    native_installer: Callable[..., Sequence[int]] | None = None,
    cache_installer: Callable[..., Sequence[int]] | None = None,
) -> LoadedQwenArm:
    """Validate assets, construct one arm, then freeze exactly as declared."""
    if not isinstance(spec, QwenArmLoadSpec):
        raise TypeError("spec must be a QwenArmLoadSpec")
    assets = [spec.model_asset, spec.data_asset]
    if spec.native_checkpoint is not None:
        assets.append(spec.native_checkpoint)
    if spec.cache_resume is not None:
        assets.append(spec.cache_resume)
    validated = validate_external_assets(assets)
    paths = {asset.name: asset.path for asset in validated}
    measured = {asset.name: asset for asset in validated}
    assert spec.native_checkpoint is not None
    checkpoint_identity = measured[spec.native_checkpoint.name]
    if checkpoint_identity.sha256 != spec.pre_replacement_checkpoint_sha256:
        raise AssetIdentityError(
            "checkpoint_identity_mismatch",
            "measured native checkpoint identity does not match "
            "pre_replacement_checkpoint_sha256",
        )
    model_path = paths[spec.model_asset.name]
    checkpoint_path = paths[spec.native_checkpoint.name]
    resume_path = (
        None if spec.cache_resume is None else paths[spec.cache_resume.name]
    )

    loader = base_model_loader or _default_base_model_loader
    manager_builder = manager_factory or _default_manager_factory
    model = loader(model_path, **dict(spec.model_loader_kwargs))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("base model loader must return a torch.nn.Module")
    resolved_model_config = (
        getattr(model, "config", None) if model_config is None else model_config
    )
    if resolved_model_config is None:
        raise TypeError(
            "model_config must be supplied or exposed by the loaded model"
        )
    target_dtype = _intended_model_dtype(model, spec.model_loader_kwargs)
    manager = manager_builder(model, resolved_model_config)
    common = {
        "model": model,
        "manager": manager,
        "model_config": resolved_model_config,
        "cache_config": cache_config,
        "native_checkpoint": checkpoint_path,
        "cache_resume": resume_path,
        "expected_job_id": spec.job_id,
        "target_dtype": target_dtype,
    }
    if spec.arm == "native":
        installer = native_installer or _default_native_installer
        raw_indices = installer(**common)
    else:
        installer = cache_installer
        if installer is None:
            raw_indices = _default_cache_installer(arm=spec.arm, **common)
        else:
            raw_indices = installer(**common)
    indices = tuple(raw_indices)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("installer returned invalid upgraded indices")
    if len(set(indices)) != len(indices):
        raise ValueError("installer returned duplicate upgraded indices")
    trainable_names = _configure_trainables(model, spec.trainable_names)
    return LoadedQwenArm(
        model=model,
        arm=spec.arm,
        job_id=spec.job_id,
        upgraded_indices=indices,
        trainable_names=trainable_names,
        assets=validated,
    )


__all__ = [
    "AssetIdentityError",
    "ExternalAssetIdentity",
    "LoadedQwenArm",
    "PairedQwenHealContract",
    "PairingContractError",
    "QwenArmLoadSpec",
    "QwenHealArmContract",
    "ValidatedAssetIdentity",
    "load_qwen_arm",
    "validate_three_arm_pairing",
    "validate_external_assets",
]
