"""Atomic, strict full-state checkpoints for paired Qwen heal arms."""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import torch


QWEN_CHECKPOINT_SCHEMA_VERSION = 1
_PAYLOAD_FIELDS = {
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
_ARMS = {"native", "recency", "surprise"}


class QwenCheckpointError(ValueError):
    """A checkpoint is incomplete, incompatible, corrupt, or unsafe to apply."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _freeze_json(value: object, context: str) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a nonfinite value")
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


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_identity_fields(
    *,
    job_id: object,
    pairing_id: object,
    arm: object,
    source_hashes: object,
    data_identity: object,
    example_ids: object,
    promotion_config: object,
) -> tuple[str, str, str, Mapping[str, object], Mapping[str, object], tuple[str, ...], Mapping[str, object]]:
    if type(job_id) is not str or not job_id:
        raise ValueError("job_id must be a nonempty string")
    pairing = _sha256("pairing_id", pairing_id)
    if arm not in _ARMS:
        raise ValueError("arm must be native, recency, or surprise")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("source_hashes must be a nonempty mapping")
    normalized_hashes: dict[str, str] = {}
    for name in sorted(source_hashes):
        if type(name) is not str or not name:
            raise ValueError("source_hashes names must be nonempty strings")
        normalized_hashes[name] = _sha256(f"source_hashes.{name}", source_hashes[name])
    if not isinstance(data_identity, Mapping) or not data_identity:
        raise ValueError("data_identity must be a nonempty mapping")
    frozen_data = _freeze_json(data_identity, "data_identity")
    assert isinstance(frozen_data, Mapping)
    if type(example_ids) is not tuple or not example_ids:
        raise ValueError("example_ids must be a nonempty tuple")
    if any(type(item) is not str or not item for item in example_ids):
        raise ValueError("example_ids must contain nonempty strings")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_ids must not contain duplicates")
    if not isinstance(promotion_config, Mapping) or not promotion_config:
        raise ValueError("promotion_config must be a nonempty mapping")
    frozen_promotion = _freeze_json(promotion_config, "promotion_config")
    assert isinstance(frozen_promotion, Mapping)
    return (
        job_id,
        pairing,
        arm,
        MappingProxyType(normalized_hashes),
        frozen_data,
        example_ids,
        frozen_promotion,
    )


@dataclass(frozen=True)
class QwenCheckpointMetadata:
    """Run identity and progress stored in every complete checkpoint."""

    job_id: str
    pairing_id: str
    arm: str
    step: int
    tokens_seen: int
    source_hashes: Mapping[str, object]
    data_identity: Mapping[str, object]
    example_ids: tuple[str, ...]
    promotion_config: Mapping[str, object]

    def __post_init__(self) -> None:
        identity = _validate_identity_fields(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            source_hashes=self.source_hashes,
            data_identity=self.data_identity,
            example_ids=self.example_ids,
            promotion_config=self.promotion_config,
        )
        for name, value in zip(
            (
                "job_id",
                "pairing_id",
                "arm",
                "source_hashes",
                "data_identity",
                "example_ids",
                "promotion_config",
            ),
            identity,
        ):
            object.__setattr__(self, name, value)
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(self.tokens_seen) is not int or self.tokens_seen < 0:
            raise ValueError("tokens_seen must be a nonnegative integer")
        if (self.step == 0) != (self.tokens_seen == 0):
            raise ValueError("step and tokens_seen must both be zero or both positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "pairing_id": self.pairing_id,
            "arm": self.arm,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "source_hashes": _thaw_json(self.source_hashes),
            "data_identity": _thaw_json(self.data_identity),
            "example_ids": list(self.example_ids),
            "promotion_config": _thaw_json(self.promotion_config),
        }

    @classmethod
    def from_dict(cls, value: object) -> "QwenCheckpointMetadata":
        if not isinstance(value, Mapping):
            raise QwenCheckpointError("metadata_invalid", "metadata must be a mapping")
        expected = {
            "job_id",
            "pairing_id",
            "arm",
            "step",
            "tokens_seen",
            "source_hashes",
            "data_identity",
            "example_ids",
            "promotion_config",
        }
        if set(value) != expected:
            raise QwenCheckpointError(
                "metadata_invalid", "metadata fields are incomplete or unknown"
            )
        try:
            return cls(
                job_id=value["job_id"],
                pairing_id=value["pairing_id"],
                arm=value["arm"],
                step=value["step"],
                tokens_seen=value["tokens_seen"],
                source_hashes=value["source_hashes"],
                data_identity=value["data_identity"],
                example_ids=tuple(value["example_ids"]),
                promotion_config=value["promotion_config"],
            )
        except (TypeError, ValueError) as error:
            raise QwenCheckpointError("metadata_invalid", str(error)) from error


@dataclass(frozen=True)
class QwenResumeExpectation:
    """Immutable run identity expected by a process before it accepts resume."""

    job_id: str
    pairing_id: str
    arm: str
    source_hashes: Mapping[str, object]
    data_identity: Mapping[str, object]
    example_ids: tuple[str, ...]
    promotion_config: Mapping[str, object]

    def __post_init__(self) -> None:
        identity = _validate_identity_fields(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            source_hashes=self.source_hashes,
            data_identity=self.data_identity,
            example_ids=self.example_ids,
            promotion_config=self.promotion_config,
        )
        for name, value in zip(
            (
                "job_id",
                "pairing_id",
                "arm",
                "source_hashes",
                "data_identity",
                "example_ids",
                "promotion_config",
            ),
            identity,
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_metadata(cls, metadata: QwenCheckpointMetadata) -> "QwenResumeExpectation":
        if not isinstance(metadata, QwenCheckpointMetadata):
            raise TypeError("metadata must be QwenCheckpointMetadata")
        return cls(
            job_id=metadata.job_id,
            pairing_id=metadata.pairing_id,
            arm=metadata.arm,
            source_hashes=metadata.source_hashes,
            data_identity=metadata.data_identity,
            example_ids=metadata.example_ids,
            promotion_config=metadata.promotion_config,
        )


@dataclass(frozen=True)
class QwenResumeState:
    job_id: str
    pairing_id: str
    arm: str
    step: int
    tokens_seen: int


def _validate_target_names(
    model: torch.nn.Module, target_module_names: tuple[str, ...]
) -> tuple[str, ...]:
    if type(target_module_names) is not tuple or not target_module_names:
        raise ValueError("target_module_names must be a nonempty tuple")
    if any(type(name) is not str or not name for name in target_module_names):
        raise ValueError("target_module_names must contain nonempty strings")
    if len(set(target_module_names)) != len(target_module_names):
        raise ValueError("target_module_names must not contain duplicates")
    if tuple(sorted(target_module_names)) != target_module_names:
        raise ValueError("target_module_names must be in canonical sorted order")
    for left in target_module_names:
        for right in target_module_names:
            if left != right and right.startswith(left + "."):
                raise ValueError("target_module_names must not overlap")
    modules = dict(model.named_modules())
    missing = [name for name in target_module_names if name not in modules]
    if missing:
        raise KeyError("target modules are missing: " + ", ".join(missing))
    return target_module_names


def _selected_state(
    model: torch.nn.Module, target_module_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    targets = _validate_target_names(model, target_module_names)
    state = model.state_dict()
    selected = {
        name: tensor
        for name, tensor in state.items()
        if any(name == target or name.startswith(target + ".") for target in targets)
    }
    for target in targets:
        if not any(name == target or name.startswith(target + ".") for name in selected):
            raise ValueError(f"target module {target!r} has no checkpoint tensors")
    return {name: selected[name] for name in sorted(selected)}


def _cpu_state(selected: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in selected.items():
        detached = tensor.detach()
        if detached.is_floating_point() and not bool(torch.isfinite(detached).all()):
            raise QwenCheckpointError(
                "nonfinite_tensor", f"target tensor {name!r} is nonfinite"
            )
        result[name] = detached.to(device="cpu").clone().contiguous()
    return result


def _tensor_manifest(state: Mapping[str, torch.Tensor]) -> list[dict[str, object]]:
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in state.items()
    ]


def _amplitude_range(state: Mapping[str, torch.Tensor]) -> list[float] | None:
    amplitudes = [
        tensor.float().reshape(-1)
        for name, tensor in state.items()
        if name == "cache_amplitude" or name.endswith(".cache_amplitude")
    ]
    if not amplitudes:
        return None
    values = torch.cat(amplitudes)
    if not bool(torch.isfinite(values).all()):
        raise QwenCheckpointError(
            "amplitude_out_of_range", "cache amplitude contains a nonfinite value"
        )
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0.0 or maximum > 1.0:
        raise QwenCheckpointError(
            "amplitude_out_of_range",
            f"cache amplitude range [{minimum}, {maximum}] is outside [0,1]",
        )
    return [minimum, maximum]


def _optimizer_parameter_names(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in by_identity:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer references a parameter outside the model",
                )
            if identity in seen:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer parameter appears in more than one group",
                )
            seen.add(identity)
            names.append(by_identity[identity])
        declared = group.get("parameter_names")
        if declared is not None and tuple(names) != tuple(declared):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                "optimizer parameter_names do not match parameter identity/order",
            )
        groups.append(names)
    return groups


def _validate_optimizer_target_coverage(
    optimizer_names: list[list[str]], selected_state: Mapping[str, torch.Tensor]
) -> None:
    missing = sorted(
        name
        for group in optimizer_names
        for name in group
        if name not in selected_state
    )
    if missing:
        raise QwenCheckpointError(
            "optimizer_target_coverage",
            "optimizer-owned parameters are outside checkpoint targets: "
            + ", ".join(missing),
        )


def _rng_state() -> dict[str, object]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in cuda],
    }


def _restore_rng(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    cuda = value["torch_cuda"]
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _validate_rng(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "python",
        "torch_cpu",
        "torch_cuda",
    }:
        raise QwenCheckpointError("rng_state_invalid", "RNG state fields are invalid")
    cpu = value["torch_cpu"]
    cuda = value["torch_cuda"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise QwenCheckpointError("rng_state_invalid", "torch CPU RNG state is invalid")
    if not isinstance(cuda, list) or any(
        not isinstance(item, torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1
        for item in cuda
    ):
        raise QwenCheckpointError("rng_state_invalid", "torch CUDA RNG state is invalid")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda) != expected_cuda:
        raise QwenCheckpointError(
            "rng_state_invalid", "CUDA RNG device count does not match this process"
        )
    # Validate the opaque Python state without altering the process state.
    probe = random.Random()
    try:
        probe.setstate(value["python"])
    except (TypeError, ValueError) as error:
        raise QwenCheckpointError("rng_state_invalid", "Python RNG state is invalid") from error
    return {"python": value["python"], "torch_cpu": cpu, "torch_cuda": cuda}


def _payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    metadata: QwenCheckpointMetadata,
    target_module_names: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("scheduler must be bound to the supplied optimizer")
    if not isinstance(metadata, QwenCheckpointMetadata):
        raise TypeError("metadata must be QwenCheckpointMetadata")
    names = _validate_target_names(model, target_module_names)
    state = _cpu_state(_selected_state(model, names))
    optimizer_names = _optimizer_parameter_names(model, optimizer)
    _validate_optimizer_target_coverage(optimizer_names, state)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    validated_optimizer_state = _validate_optimizer_resume_state(
        optimizer_state,
        model=model,
        optimizer=optimizer,
        expected_names=optimizer_names,
        step=metadata.step,
    )
    _validate_scheduler_resume_state(
        scheduler_state,
        scheduler=scheduler,
        optimizer_state=validated_optimizer_state,
        step=metadata.step,
    )
    return {
        "schema_version": QWEN_CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata.as_dict(),
        "target_module_names": list(names),
        "model_state": state,
        "tensor_manifest": _tensor_manifest(state),
        "optimizer_parameter_names": optimizer_names,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": _rng_state(),
        "amplitude_range": _amplitude_range(state),
    }


def save_qwen_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    metadata: QwenCheckpointMetadata,
    target_module_names: tuple[str, ...],
    save_function: Callable[[object, Path], None] | None = None,
) -> Path:
    """Flush a complete checkpoint beside the destination, then atomically replace."""
    try:
        destination = Path(path)
    except TypeError as error:
        raise TypeError("checkpoint path must be path-like") from error
    if not destination.name:
        raise ValueError("checkpoint path must name a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=target_module_names,
    )
    writer = torch.save if save_function is None else save_function
    if not callable(writer):
        raise TypeError("save_function must be callable or None")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer_payload = payload if save_function is None else copy.deepcopy(payload)
        writer(writer_payload, temporary)
        del writer_payload
        # Windows requires a writable descriptor for ``fsync``.
        try:
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise QwenCheckpointError(
                "checkpoint_candidate_io_failed",
                "could not flush the serialized checkpoint candidate",
            ) from error
        serialized = _decode_checkpoint_payload(temporary)
        _validate_loaded_payload(
            serialized,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=target_module_names,
        )
        if not _values_equal(serialized, payload):
            raise QwenCheckpointError(
                "checkpoint_serialization_mismatch",
                "serialized checkpoint candidate differs from the validated payload",
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _values_equal(left: object, right: object) -> bool:
    """Exact nested equality that is safe for tensor-bearing state dictionaries."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(
                _values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)  # type: ignore[arg-type]
            )
        )
    return left == right


def _validate_optimizer_resume_state(
    value: object,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_names: list[list[str]],
    step: int,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"state", "param_groups"}:
        raise QwenCheckpointError(
            "optimizer_state_invalid", "checkpoint optimizer state is malformed"
        )
    if not isinstance(optimizer, torch.optim.AdamW):
        raise QwenCheckpointError(
            "optimizer_state_invalid", "Qwen heal resume requires AdamW"
        )
    saved_groups = value["param_groups"]
    saved_slots = value["state"]
    template = optimizer.state_dict()
    template_groups = template["param_groups"]
    if (
        type(saved_groups) is not list
        or len(saved_groups) != len(expected_names)
        or len(saved_groups) != len(template_groups)
    ):
        raise QwenCheckpointError(
            "optimizer_state_invalid", "optimizer group count is incompatible"
        )

    parameters_by_id: dict[int, torch.nn.Parameter] = {}
    expected_ids: list[int] = []
    for index, (saved, current, live, names) in enumerate(
        zip(
            saved_groups,
            template_groups,
            optimizer.param_groups,
            expected_names,
            strict=True,
        )
    ):
        if type(saved) is not dict or set(saved) != set(current):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"optimizer group {index} fields are incompatible",
            )
        saved_ids = saved.get("params")
        current_ids = current.get("params")
        if saved_ids != current_ids:
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter IDs/order differ",
            )
        if saved.get("parameter_names") != tuple(names):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter-name mapping differs",
            )
        if not isinstance(saved_ids, list) or len(saved_ids) != len(live["params"]):
            raise QwenCheckpointError(
                "optimizer_parameter_mismatch",
                f"optimizer group {index} parameter count differs",
            )
        for parameter_id, parameter in zip(saved_ids, live["params"], strict=True):
            if type(parameter_id) is not int or parameter_id in parameters_by_id:
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer parameter IDs must be unique integers",
                )
            if not isinstance(parameter, torch.nn.Parameter):
                raise QwenCheckpointError(
                    "optimizer_parameter_mismatch",
                    "optimizer group contains a non-Parameter value",
                )
            parameters_by_id[parameter_id] = parameter
            expected_ids.append(parameter_id)
        for field, current_value in current.items():
            if field in {"params", "lr"}:
                continue
            if not _values_equal(saved[field], current_value):
                raise QwenCheckpointError(
                    "optimizer_state_invalid",
                    f"optimizer group {index} field {field!r} is incompatible",
                )
        learning_rate = saved["lr"]
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) < 0.0
        ):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"optimizer group {index} learning rate is invalid",
            )

    if type(saved_slots) is not dict:
        raise QwenCheckpointError(
            "optimizer_state_invalid", "optimizer slots must be a dictionary"
        )
    expected_slot_ids = set(expected_ids) if step > 0 else set()
    if set(saved_slots) != expected_slot_ids:
        raise QwenCheckpointError(
            "optimizer_state_invalid",
            "optimizer slot parameter IDs do not exactly match active parameters",
        )
    for parameter_id in expected_ids:
        slot = saved_slots[parameter_id]
        parameter = parameters_by_id[parameter_id]
        group = next(
            group for group in saved_groups if parameter_id in group["params"]
        )
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if group["amsgrad"]:
            expected_fields.add("max_exp_avg_sq")
        if type(slot) is not dict or set(slot) != expected_fields:
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"AdamW slot {parameter_id} fields are incompatible",
            )
        saved_step = slot["step"]
        if (
            not isinstance(saved_step, torch.Tensor)
            or saved_step.device.type != "cpu"
            or saved_step.shape != torch.Size([])
            or not saved_step.is_floating_point()
            or not bool(torch.isfinite(saved_step).all())
            or float(saved_step) != float(step)
        ):
            raise QwenCheckpointError(
                "optimizer_state_invalid",
                f"AdamW slot {parameter_id} step does not match global progress",
            )
        for field in expected_fields - {"step"}:
            tensor = slot[field]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise QwenCheckpointError(
                    "optimizer_state_invalid",
                    f"AdamW slot {parameter_id} {field} shape/dtype/finiteness differs",
                )
    return value


def _validate_scheduler_resume_state(
    value: object,
    *,
    scheduler: object,
    optimizer_state: Mapping[str, object],
    step: int,
) -> Mapping[str, object]:
    template = scheduler.state_dict()
    if not isinstance(value, Mapping) or set(value) != set(template):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler fields are incompatible"
        )
    dynamic = {"last_epoch", "_step_count", "_last_lr"}
    for field, current_value in template.items():
        if field not in dynamic and not _values_equal(value[field], current_value):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                f"checkpoint scheduler field {field!r} is incompatible",
            )
    if value.get("last_epoch") != step or value.get("_step_count") != step + 1:
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler progress is inconsistent"
        )
    rates = value.get("_last_lr")
    groups = optimizer_state["param_groups"]
    if (
        type(rates) is not list
        or not isinstance(groups, list)
        or len(rates) != len(groups)
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "checkpoint scheduler LR groups are incompatible"
        )
    base_lrs = getattr(scheduler, "base_lrs", None)
    if (
        type(base_lrs) is not list
        or len(base_lrs) != len(groups)
        or any(
            type(rate) not in (int, float)
            or not math.isfinite(float(rate))
            or float(rate) < 0.0
            for rate in base_lrs
        )
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "configured scheduler base LRs are invalid"
        )
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        lambdas = scheduler.lr_lambdas
        if len(lambdas) != len(base_lrs) or any(not callable(item) for item in lambdas):
            raise QwenCheckpointError(
                "scheduler_state_invalid", "configured scheduler lambdas are invalid"
            )
        configured_rates = [
            float(base_rate) * float(multiplier(step))
            for base_rate, multiplier in zip(base_lrs, lambdas, strict=True)
        ]
    else:
        probe = copy.deepcopy(scheduler)
        closed_form = getattr(probe, "_get_closed_form_lr", None)
        if not callable(closed_form):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                "configured scheduler cannot derive learning rates from progress",
            )
        probe.last_epoch = step
        configured_rates = list(closed_form())
    if (
        len(configured_rates) != len(groups)
        or any(not math.isfinite(rate) or rate < 0.0 for rate in configured_rates)
    ):
        raise QwenCheckpointError(
            "scheduler_state_invalid", "configured scheduler learning rates are invalid"
        )
    for index, (rate, group, base_rate, configured_rate) in enumerate(
        zip(rates, groups, base_lrs, configured_rates, strict=True)
    ):
        if (
            type(rate) not in (int, float)
            or not math.isfinite(float(rate))
            or float(rate) < 0.0
            or not isinstance(group, Mapping)
            or float(rate) != float(group["lr"])
            or float(group.get("initial_lr", -1.0)) != float(base_rate)
            or float(rate) != configured_rate
        ):
            raise QwenCheckpointError(
                "scheduler_state_invalid",
                f"checkpoint scheduler LR for group {index} is inconsistent",
            )
    return value


def _validate_loaded_payload(
    payload: object,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expectation: QwenResumeExpectation,
    target_module_names: tuple[str, ...],
) -> tuple[QwenCheckpointMetadata, dict[str, torch.Tensor], dict[str, object]]:
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        raise QwenCheckpointError(
            "checkpoint_fields_invalid", "checkpoint fields are incomplete or unknown"
        )
    if payload["schema_version"] != QWEN_CHECKPOINT_SCHEMA_VERSION:
        raise QwenCheckpointError(
            "checkpoint_schema_mismatch", "checkpoint schema version is incompatible"
        )
    metadata = QwenCheckpointMetadata.from_dict(payload["metadata"])
    identity_fields = (
        "job_id",
        "pairing_id",
        "arm",
        "source_hashes",
        "data_identity",
        "example_ids",
        "promotion_config",
    )
    mismatched = [
        name
        for name in identity_fields
        if getattr(metadata, name) != getattr(expectation, name)
    ]
    if mismatched:
        raise QwenCheckpointError(
            "resume_identity_mismatch",
            "checkpoint identity differs for: " + ", ".join(mismatched),
        )
    names = _validate_target_names(model, target_module_names)
    if payload["target_module_names"] != list(names):
        raise QwenCheckpointError(
            "target_module_mismatch", "checkpoint target modules do not match"
        )
    current = _selected_state(model, names)
    loaded_state = payload["model_state"]
    if not isinstance(loaded_state, Mapping) or tuple(loaded_state) != tuple(current):
        raise QwenCheckpointError(
            "tensor_name_mismatch", "checkpoint tensor names/order do not match"
        )
    validated_state: dict[str, torch.Tensor] = {}
    for name, target in current.items():
        tensor = loaded_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise QwenCheckpointError(
                "tensor_type_mismatch", f"checkpoint value {name!r} is not a tensor"
            )
        if tensor.device.type != "cpu":
            raise QwenCheckpointError(
                "tensor_device_mismatch", f"checkpoint tensor {name!r} is not CPU portable"
            )
        if tensor.shape != target.shape:
            raise QwenCheckpointError(
                "tensor_shape_mismatch", f"checkpoint tensor {name!r} shape differs"
            )
        if tensor.dtype != target.dtype:
            raise QwenCheckpointError(
                "tensor_dtype_mismatch", f"checkpoint tensor {name!r} dtype differs"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise QwenCheckpointError(
                "nonfinite_tensor", f"checkpoint tensor {name!r} is nonfinite"
            )
        validated_state[name] = tensor
    if payload["tensor_manifest"] != _tensor_manifest(validated_state):
        raise QwenCheckpointError(
            "tensor_manifest_mismatch", "checkpoint tensor manifest is stale or corrupt"
        )
    if payload["amplitude_range"] != _amplitude_range(validated_state):
        raise QwenCheckpointError(
            "amplitude_manifest_mismatch", "checkpoint amplitude range is stale"
        )
    current_optimizer_names = _optimizer_parameter_names(model, optimizer)
    _validate_optimizer_target_coverage(current_optimizer_names, current)
    if payload["optimizer_parameter_names"] != current_optimizer_names:
        raise QwenCheckpointError(
            "optimizer_parameter_mismatch", "checkpoint optimizer parameter order differs"
        )
    optimizer_state = _validate_optimizer_resume_state(
        payload["optimizer_state"],
        model=model,
        optimizer=optimizer,
        expected_names=current_optimizer_names,
        step=metadata.step,
    )
    _validate_scheduler_resume_state(
        payload["scheduler_state"],
        scheduler=scheduler,
        optimizer_state=optimizer_state,
        step=metadata.step,
    )
    rng = _validate_rng(payload["rng_state"])
    return metadata, validated_state, rng


def _decode_checkpoint_payload(checkpoint_path: Path) -> object:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise QwenCheckpointError(
            "checkpoint_decode_failed", f"could not decode {checkpoint_path}"
        ) from error


def load_qwen_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expectation: QwenResumeExpectation,
    target_module_names: tuple[str, ...],
) -> QwenResumeState:
    """Prevalidate every field, then transactionally restore all dynamic state."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("scheduler must be bound to the supplied optimizer")
    if not isinstance(expectation, QwenResumeExpectation):
        raise TypeError("expectation must be QwenResumeExpectation")
    try:
        checkpoint_path = Path(path)
    except TypeError as error:
        raise TypeError("checkpoint path must be path-like") from error
    payload = _decode_checkpoint_payload(checkpoint_path)
    metadata, loaded_state, loaded_rng = _validate_loaded_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expectation=expectation,
        target_module_names=target_module_names,
    )
    current = _selected_state(model, target_module_names)
    model_snapshot = {name: tensor.detach().clone() for name, tensor in current.items()}
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())
    rng_snapshot = _rng_state()
    try:
        with torch.no_grad():
            for name, target in current.items():
                target.copy_(loaded_state[name].to(device=target.device))
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
        scheduler.load_state_dict(copy.deepcopy(payload["scheduler_state"]))
        _restore_rng(loaded_rng)
    except BaseException as error:
        with torch.no_grad():
            for name, target in current.items():
                target.copy_(model_snapshot[name])
        optimizer.load_state_dict(optimizer_snapshot)
        scheduler.load_state_dict(scheduler_snapshot)
        _restore_rng(rng_snapshot)
        if not isinstance(error, Exception):
            raise
        raise QwenCheckpointError(
            "resume_apply_failed", "checkpoint application failed and was rolled back"
        ) from error
    return QwenResumeState(
        job_id=metadata.job_id,
        pairing_id=metadata.pairing_id,
        arm=metadata.arm,
        step=metadata.step,
        tokens_seen=metadata.tokens_seen,
    )


__all__ = [
    "QWEN_CHECKPOINT_SCHEMA_VERSION",
    "QwenCheckpointError",
    "QwenCheckpointMetadata",
    "QwenResumeExpectation",
    "QwenResumeState",
    "load_qwen_checkpoint",
    "save_qwen_checkpoint",
]
