"""Small, deterministic Qwen heal losses and one-update training adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class QwenTrainingError(RuntimeError):
    """Typed failure that must invalidate, rather than alter, a paired run."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class TeacherRequiredError(QwenTrainingError):
    """Ordinary Qwen heal was requested without a frozen teacher."""


class QwenRuntimeConfigurationError(QwenTrainingError):
    """Runtime-only execution bindings are absent, stale, or incompatible."""


def _finite_real(name: str, value: object, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


@dataclass(frozen=True)
class QwenHealTrainingConfig:
    """Fixed objective and stopping contract shared by all paired arms."""

    objective: str
    ce_weight: float
    kl_weight: float
    layerwise_weight: float
    temperature: float
    accumulation_steps: int
    max_updates: int
    max_tokens: int
    gradient_checkpointing: bool

    def __post_init__(self) -> None:
        if type(self.objective) is not str or not self.objective:
            raise ValueError("objective must be a nonempty string")
        for name in ("ce_weight", "kl_weight", "layerwise_weight"):
            object.__setattr__(self, name, _finite_real(name, getattr(self, name)))
        if self.ce_weight + self.kl_weight + self.layerwise_weight <= 0.0:
            raise ValueError("at least one heal loss weight must be positive")
        temperature = _finite_real("temperature", self.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(self, "temperature", temperature)
        for name in ("accumulation_steps", "max_updates", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be boolean")
        if self.objective == "synthetic_only" and (
            self.kl_weight != 0.0 or self.layerwise_weight != 0.0
        ):
            raise ValueError(
                "synthetic_only without a teacher requires zero KL and layerwise weights"
            )


def validate_teacher_requirement(
    config: QwenHealTrainingConfig,
    *,
    teacher_present: bool,
    phase: str,
) -> None:
    """Apply the same teacher guard in preflight and at runtime."""
    if not isinstance(config, QwenHealTrainingConfig):
        raise TypeError("config must be a QwenHealTrainingConfig")
    if type(teacher_present) is not bool:
        raise TypeError("teacher_present must be boolean")
    if type(phase) is not str or not phase:
        raise ValueError("phase must be a nonempty string")
    if not teacher_present and config.objective != "synthetic_only":
        raise TeacherRequiredError(
            "teacher_required",
            f"{phase}: objective {config.objective!r} requires a teacher model",
        )


def _validate_logits(name: str, logits: object) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError(f"{name} must have shape [batch, time>=2, vocab>=2]")
    return logits


def _validate_labels(labels: object, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise TypeError("labels must be a torch.long tensor")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits batch/time dimensions")
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("labels must contain at least one valid causal target")
    return labels


def causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Teacher-forced next-token CE using the standard one-token shift."""
    logits = _validate_logits("logits", logits)
    labels = _validate_labels(labels, logits)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Canonical full-logit ``KL(teacher || student)`` distillation loss."""
    student = _validate_logits("student_logits", student_logits)
    teacher = _validate_logits("teacher_logits", teacher_logits)
    if teacher.shape != student.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    scale = _finite_real("temperature", temperature)
    if scale <= 0.0:
        raise ValueError("temperature must be positive")
    student_log = F.log_softmax(student.float() / scale, dim=-1)
    teacher_log = F.log_softmax(
        teacher.detach().to(device=student.device, dtype=torch.float32) / scale,
        dim=-1,
    )
    return F.kl_div(
        student_log,
        teacher_log,
        reduction="batchmean",
        log_target=True,
    ) * (scale * scale) / student.shape[1]


def layerwise_alignment_loss(
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Canonical normalized residual MSE over ``hidden_states[1:]``."""
    if isinstance(student_hidden, torch.Tensor) or not isinstance(
        student_hidden, Sequence
    ):
        raise TypeError("student_hidden must be a sequence of tensors")
    if isinstance(teacher_hidden, torch.Tensor) or not isinstance(
        teacher_hidden, Sequence
    ):
        raise TypeError("teacher_hidden must be a sequence of tensors")
    if len(student_hidden) < 2 or len(student_hidden) != len(teacher_hidden):
        raise ValueError("student and teacher hidden-state sequences must align")
    losses: list[torch.Tensor] = []
    for index, (student, teacher) in enumerate(
        zip(student_hidden[1:], teacher_hidden[1:]), start=1
    ):
        if (
            not isinstance(student, torch.Tensor)
            or not isinstance(teacher, torch.Tensor)
            or not student.is_floating_point()
            or not teacher.is_floating_point()
        ):
            raise TypeError(f"hidden layer {index} must contain floating tensors")
        if student.shape != teacher.shape or student.ndim < 2:
            raise ValueError(f"hidden layer {index} shape does not align")
        teacher_float = teacher.detach().to(
            device=student.device, dtype=torch.float32
        )
        difference = student.float() - teacher_float
        losses.append(
            difference.square().mean()
            / teacher_float.square().mean().clamp_min(1.0e-8)
        )
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class HealLossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    kl: torch.Tensor
    layerwise: torch.Tensor


def _output_field(output: object, name: str) -> object:
    if isinstance(output, Mapping):
        if name not in output:
            raise ValueError(f"model output is missing {name}")
        return output[name]
    if not hasattr(output, name):
        raise ValueError(f"model output is missing {name}")
    return getattr(output, name)


def compute_heal_loss(
    student_output: object,
    teacher_output: object | None,
    labels: torch.Tensor,
    config: QwenHealTrainingConfig,
) -> HealLossBreakdown:
    """Compose the three preregistered losses without importing a trainer."""
    if not isinstance(config, QwenHealTrainingConfig):
        raise TypeError("config must be a QwenHealTrainingConfig")
    student_logits = _validate_logits(
        "student_logits", _output_field(student_output, "logits")
    )
    zero = student_logits.sum() * 0.0
    ce = (
        causal_cross_entropy(student_logits, labels)
        if config.ce_weight > 0.0
        else zero
    )
    if config.kl_weight > 0.0 or config.layerwise_weight > 0.0:
        if teacher_output is None:
            raise TeacherRequiredError(
                "teacher_required", "KL/layerwise Qwen heal losses require a teacher"
            )
    if config.kl_weight > 0.0:
        assert teacher_output is not None
        kl = distillation_kl(
            student_logits,
            _output_field(teacher_output, "logits"),
            temperature=config.temperature,
        )
    else:
        kl = zero
    if config.layerwise_weight > 0.0:
        assert teacher_output is not None
        layerwise = layerwise_alignment_loss(
            _output_field(student_output, "hidden_states"),
            _output_field(teacher_output, "hidden_states"),
        )
    else:
        layerwise = zero
    total = (
        config.ce_weight * ce
        + config.kl_weight * kl
        + config.layerwise_weight * layerwise
    )
    return HealLossBreakdown(total=total, ce=ce, kl=kl, layerwise=layerwise)


def _validate_parameter_names(
    model: torch.nn.Module,
    memory_parameter_names: tuple[str, ...],
    cache_parameter_names: tuple[str, ...],
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], tuple[tuple[str, torch.nn.Parameter], ...]]:
    if type(memory_parameter_names) is not tuple or not memory_parameter_names:
        raise ValueError("memory_parameter_names must be a nonempty tuple")
    if type(cache_parameter_names) is not tuple:
        raise TypeError("cache_parameter_names must be a tuple")
    combined = memory_parameter_names + cache_parameter_names
    if any(type(name) is not str or not name for name in combined):
        raise ValueError("optimizer parameter names must be nonempty strings")
    if len(set(combined)) != len(combined):
        raise ValueError("optimizer parameter groups overlap or contain duplicates")
    named = dict(model.named_parameters())
    missing = sorted(set(combined) - set(named))
    if missing:
        raise KeyError("optimizer parameter names are missing: " + ", ".join(missing))
    actual_trainable = {name for name, parameter in named.items() if parameter.requires_grad}
    if actual_trainable != set(combined):
        raise ValueError(
            "optimizer groups must cover exactly the declared trainable parameters"
        )
    memory = tuple((name, named[name]) for name in memory_parameter_names)
    cache = tuple((name, named[name]) for name in cache_parameter_names)
    return memory, cache


def build_qwen_heal_optimizer(
    model: torch.nn.Module,
    *,
    memory_parameter_names: tuple[str, ...],
    cache_parameter_names: tuple[str, ...],
    learning_rate: float,
    lr_cache: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build stable memory/cache AdamW groups with no cache weight decay."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    memory, cache = _validate_parameter_names(
        model, memory_parameter_names, cache_parameter_names
    )
    memory_lr = _finite_real("learning_rate", learning_rate)
    cache_lr = _finite_real("lr_cache", lr_cache)
    epsilon = _finite_real("eps", eps)
    decay = _finite_real("weight_decay", weight_decay)
    if memory_lr <= 0.0 or cache_lr <= 0.0 or epsilon <= 0.0:
        raise ValueError("optimizer learning rates and eps must be positive")
    if (
        type(betas) is not tuple
        or len(betas) != 2
        or any(type(beta) not in (int, float) for beta in betas)
        or any(not math.isfinite(float(beta)) or not 0.0 <= float(beta) < 1.0 for beta in betas)
    ):
        raise ValueError("betas must be two finite values in [0,1)")
    groups: list[dict[str, Any]] = [
        {
            "name": "memory",
            "parameter_names": tuple(name for name, _ in memory),
            "params": [parameter for _, parameter in memory],
            "lr": memory_lr,
            "weight_decay": decay,
        }
    ]
    if cache:
        groups.append(
            {
                "name": "cache",
                "parameter_names": tuple(name for name, _ in cache),
                "params": [parameter for _, parameter in cache],
                "lr": cache_lr,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=(float(betas[0]), float(betas[1])),
        eps=epsilon,
    )


def project_cache_amplitudes_(model: torch.nn.Module) -> tuple[str, ...]:
    """Project every declared cache amplitude to the closed identity-gate range."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    amplitudes = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name == "cache_amplitude" or name.endswith(".cache_amplitude")
    )
    with torch.no_grad():
        for name, parameter in amplitudes:
            if not bool(torch.isfinite(parameter).all()):
                raise QwenTrainingError(
                    "nonfinite_parameter", f"cache amplitude {name} is nonfinite"
                )
            parameter.clamp_(0.0, 1.0)
    return tuple(name for name, _ in amplitudes)


@dataclass(frozen=True)
class HealStepLog:
    job_id: str
    pairing_id: str
    arm: str
    update: int
    tokens_seen: int
    example_ids: tuple[str, ...]
    microbatches: int
    losses: Mapping[str, float]
    learning_rates: Mapping[str, float]
    skipped_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "pairing_id": self.pairing_id,
            "arm": self.arm,
            "update": self.update,
            "tokens_seen": self.tokens_seen,
            "example_ids": list(self.example_ids),
            "microbatches": self.microbatches,
            "losses": dict(sorted(self.losses.items())),
            "learning_rates": dict(sorted(self.learning_rates.items())),
            "skipped_steps": self.skipped_steps,
        }


class QwenHealTrainer:
    """One deterministic paired-heal update path, independent of the main trainer."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        teacher: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        config: QwenHealTrainingConfig,
        job_id: str,
        pairing_id: str,
        arm: str,
        expected_example_windows: tuple[tuple[str, ...], ...],
        teacher_device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if teacher is not None and not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher must be a torch.nn.Module or None")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise ValueError("scheduler must be bound to the supplied optimizer")
        if not isinstance(config, QwenHealTrainingConfig):
            raise TypeError("config must be a QwenHealTrainingConfig")
        validate_teacher_requirement(
            config, teacher_present=teacher is not None, phase="runtime"
        )
        if type(job_id) is not str or not job_id:
            raise ValueError("job_id must be a nonempty string")
        if (
            type(pairing_id) is not str
            or len(pairing_id) != 64
            or any(character not in "0123456789abcdef" for character in pairing_id)
        ):
            raise ValueError("pairing_id must be lowercase SHA-256")
        if arm not in {"native", "recency", "surprise"}:
            raise ValueError("arm must be native, recency, or surprise")
        expected_count = config.max_updates * config.accumulation_steps
        if (
            type(expected_example_windows) is not tuple
            or len(expected_example_windows) != expected_count
        ):
            raise ValueError(
                "expected_example_windows must cover every configured microbatch"
            )
        for window in expected_example_windows:
            if (
                type(window) is not tuple
                or not window
                or any(type(item) is not str or not item for item in window)
            ):
                raise ValueError("each expected example window must be nonempty IDs")
        if config.gradient_checkpointing:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise TypeError(
                    "gradient checkpointing was requested but the model cannot enable it"
                )
            enable()
        if teacher is not None:
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        try:
            resolved_teacher_device = (
                None if teacher_device is None else torch.device(teacher_device)
            )
        except (TypeError, RuntimeError) as error:
            raise ValueError("teacher_device must name a valid torch device") from error
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.job_id = job_id
        self.pairing_id = pairing_id
        self.arm = arm
        self.expected_example_windows = expected_example_windows
        self.teacher_device = resolved_teacher_device
        self.step = 0
        self.tokens_seen = 0
        self.example_cursor = 0
        self.skipped_steps = 0

    def _prevalidate_batches(
        self, microbatches: Sequence[Mapping[str, object]]
    ) -> tuple[int, tuple[str, ...]]:
        if isinstance(microbatches, (str, bytes)) or not isinstance(
            microbatches, Sequence
        ):
            raise TypeError("microbatches must be a sequence of mappings")
        if len(microbatches) != self.config.accumulation_steps:
            raise QwenTrainingError(
                "accumulation_mismatch",
                f"expected {self.config.accumulation_steps} microbatches",
            )
        total_tokens = 0
        flattened_ids: list[str] = []
        for offset, batch in enumerate(microbatches):
            if not isinstance(batch, Mapping):
                raise TypeError("each microbatch must be a mapping")
            missing = {"input_ids", "labels", "example_ids"} - set(batch)
            if missing:
                raise ValueError("microbatch is missing: " + ", ".join(sorted(missing)))
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            ids = batch["example_ids"]
            if (
                not isinstance(input_ids, torch.Tensor)
                or input_ids.dtype != torch.long
                or input_ids.ndim != 2
            ):
                raise TypeError("input_ids must be a rank-2 torch.long tensor")
            if (
                not isinstance(labels, torch.Tensor)
                or labels.dtype != torch.long
                or labels.shape != input_ids.shape
            ):
                raise TypeError("labels must be torch.long and match input_ids")
            if type(ids) is not tuple or len(ids) != input_ids.shape[0]:
                raise ValueError("example_ids must match the microbatch size")
            expected = self.expected_example_windows[self.example_cursor + offset]
            if ids != expected:
                raise QwenTrainingError(
                    "example_window_mismatch",
                    f"expected {expected!r}, received {ids!r}",
                )
            total_tokens += input_ids.numel()
            flattened_ids.extend(ids)
        return total_tokens, tuple(flattened_ids)

    @staticmethod
    def _model_inputs(batch: Mapping[str, object]) -> dict[str, object]:
        inputs = {
            name: value
            for name, value in batch.items()
            if name not in {"labels", "example_ids"}
        }
        for reserved in ("output_hidden_states", "use_cache"):
            if reserved in inputs:
                raise ValueError(f"microbatch cannot override {reserved}")
        inputs["output_hidden_states"] = True
        inputs["use_cache"] = False
        return inputs

    def _fail_nonfinite(self, code: str, message: str) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.skipped_steps += 1
        raise QwenTrainingError(code, message)

    def train_update(
        self, microbatches: Sequence[Mapping[str, object]]
    ) -> HealStepLog:
        if self.step >= self.config.max_updates:
            raise QwenTrainingError(
                "update_budget_exhausted", "configured update budget is exhausted"
            )
        token_count, example_ids = self._prevalidate_batches(microbatches)
        if self.tokens_seen + token_count > self.config.max_tokens:
            raise QwenTrainingError(
                "token_budget_exhausted", "update would exceed the fixed token budget"
            )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "ce": 0.0, "kl": 0.0, "layerwise": 0.0}
        divisor = float(self.config.accumulation_steps)
        for batch in microbatches:
            labels = batch["labels"]
            assert isinstance(labels, torch.Tensor)
            model_inputs = self._model_inputs(batch)
            from .qwen_exact_cache import guarded_model_forward

            student_output = guarded_model_forward(self.model, **model_inputs)
            if self.teacher is None:
                teacher_output = None
            else:
                teacher_inputs = {
                    name: (
                        value.to(self.teacher_device)
                        if self.teacher_device is not None
                        and isinstance(value, torch.Tensor)
                        else value
                    )
                    for name, value in model_inputs.items()
                }
                with torch.no_grad():
                    teacher_output = self.teacher(**teacher_inputs)
            breakdown = compute_heal_loss(
                student_output, teacher_output, labels, self.config
            )
            values = {
                "total": breakdown.total,
                "ce": breakdown.ce,
                "kl": breakdown.kl,
                "layerwise": breakdown.layerwise,
            }
            if any(not bool(torch.isfinite(value.detach()).all()) for value in values.values()):
                self._fail_nonfinite(
                    "nonfinite_loss", "Qwen heal loss contains a nonfinite value"
                )
            (breakdown.total / divisor).backward()
            for name, value in values.items():
                totals[name] += float(value.detach().cpu()) / divisor

        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        if any(parameter.grad is None for parameter in optimizer_parameters):
            self._fail_nonfinite(
                "missing_gradient", "a declared trainable parameter has no gradient"
            )
        if any(
            not bool(torch.isfinite(parameter.grad.detach()).all())
            for parameter in optimizer_parameters
            if parameter.grad is not None
        ):
            self._fail_nonfinite(
                "nonfinite_gradient", "Qwen heal gradients contain a nonfinite value"
            )

        parameter_snapshot = [parameter.detach().clone() for parameter in optimizer_parameters]
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        scheduler_snapshot = copy.deepcopy(self.scheduler.state_dict())
        try:
            self.optimizer.step()
            project_cache_amplitudes_(self.model)
            from .qwen_variants import project_variant_gates_

            project_variant_gates_(self.model)
            if any(
                not bool(torch.isfinite(parameter.detach()).all())
                for parameter in optimizer_parameters
            ):
                raise QwenTrainingError(
                    "nonfinite_parameter", "optimizer produced a nonfinite parameter"
                )
            self.scheduler.step()
        except BaseException:
            with torch.no_grad():
                for parameter, snapshot in zip(optimizer_parameters, parameter_snapshot):
                    parameter.copy_(snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.scheduler.load_state_dict(scheduler_snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            raise
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens_seen += token_count
        self.example_cursor += self.config.accumulation_steps
        rates: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            if type(name) is not str or name in rates:
                raise RuntimeError("optimizer group names must be unique strings")
            rates[name] = float(group["lr"])
        return HealStepLog(
            job_id=self.job_id,
            pairing_id=self.pairing_id,
            arm=self.arm,
            update=self.step,
            tokens_seen=self.tokens_seen,
            example_ids=example_ids,
            microbatches=self.config.accumulation_steps,
            losses=totals,
            learning_rates=rates,
            skipped_steps=self.skipped_steps,
        )


@dataclass(frozen=True)
class QwenJobData:
    """Materialized, identity-bearing train/evaluation windows for one heal job."""

    train_microbatches: tuple[Mapping[str, object], ...]
    eval_microbatches: tuple[Mapping[str, object], ...]
    data_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name in ("train_microbatches", "eval_microbatches"):
            value = getattr(self, field_name)
            if type(value) is not tuple or not value:
                raise ValueError(f"{field_name} must be a nonempty tuple")
            if any(not isinstance(batch, Mapping) for batch in value):
                raise TypeError(f"{field_name} must contain mappings")
        if not isinstance(self.data_identity, Mapping) or not self.data_identity:
            raise ValueError("data_identity must be a nonempty mapping")
        try:
            json.dumps(
                self.data_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("data_identity must be finite JSON") from error


_QWEN_ARM_IDS = {
    "native": "native",
    "recency": "recency",
    "exact_cache.selector.recency": "recency",
    "surprise": "surprise",
    "exact_cache.selector.exact_outer": "surprise",
}


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", f"{name} must be a mapping"
        )
    return value


def _job_config(job: Mapping[str, object]) -> Mapping[str, object]:
    config = job.get("canonical_config")
    if not isinstance(config, Mapping):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.canonical_config must be a mapping"
        )
    if config.get("backend") != "qwen" or job.get("backend") != "qwen":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen dispatcher received a non-Qwen job"
        )
    qwen = _required_mapping(config.get("qwen"), "canonical_config.qwen")
    if qwen.get("run_mode") != "heal":
        raise QwenRuntimeConfigurationError(
            "qwen_heal_required", "paired Qwen adapter supports run_mode='heal' only"
        )
    return config


def _positive_int(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < 1:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must be a positive integer"
        )
    return value


def _string_tuple(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (
        not allow_empty and not value
    ):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must be a string sequence"
        )
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must contain nonempty strings"
        )
    if len(set(result)) != len(result):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", f"{name} must not contain duplicates"
        )
    return result


def _selected_arm(job: Mapping[str, object]) -> str:
    arm_id = job.get("arm_id")
    if arm_id not in _QWEN_ARM_IDS:
        raise QwenRuntimeConfigurationError(
            "qwen_arm_invalid", f"unsupported paired Qwen arm: {arm_id!r}"
        )
    return _QWEN_ARM_IDS[arm_id]


def derive_three_arm_pairing(
    job: Mapping[str, object],
    *,
    example_ids: tuple[str, ...],
    pre_replacement_checkpoint_sha256: str,
    data_sha256: str,
):
    """Derive the native/recency/surprise scientific contract from one job."""
    from .qwen_backend import QwenHealArmContract, validate_three_arm_pairing

    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    config = _job_config(job)
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    optimizer = _required_mapping(
        config.get("optimizer"), "canonical_config.optimizer"
    )
    schedule = _required_mapping(config.get("schedule"), "canonical_config.schedule")
    lengths = _required_mapping(config.get("lengths"), "canonical_config.lengths")
    task = _required_mapping(config.get("task"), "canonical_config.task")
    task_params = _required_mapping(task.get("params"), "canonical_config.task.params")
    cache = _required_mapping(config.get("cache"), "canonical_config.cache")
    seed = job.get("seed")
    if type(seed) is not int or seed < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.seed must be a nonnegative integer"
        )
    curriculum_raw = lengths.get("curriculum")
    if not isinstance(curriculum_raw, (list, tuple)):
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "lengths.curriculum must be a sequence"
        )
    curriculum = tuple(curriculum_raw)
    extrapolation = lengths.get("extrapolation")
    if not isinstance(extrapolation, (list, tuple)) or not extrapolation:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "lengths.extrapolation must be nonempty"
        )
    stopping = task_params.get(
        "stopping", {"max_nonfinite": 0, "early_stopping": False}
    )
    if not isinstance(stopping, Mapping) or not stopping:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "task.params.stopping must be a mapping"
        )
    cache_match = {
        "width": cache.get("width"),
        "block_size": cache.get("block_size"),
        "read": cache.get("read"),
        "read_init": cache.get("read_init"),
        "storage_dtype": cache.get("storage_dtype"),
        "lr_cache": cache.get("lr_cache"),
    }
    surprise_policy = cache.get("score")
    if type(surprise_policy) is not str or surprise_policy in {"", "recency"}:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "cache.score must name the winning surprise policy"
        )
    job_id = job.get("job_id")
    if type(job_id) is not str or not job_id:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.job_id must be a nonempty string"
        )
    contracts = tuple(
        QwenHealArmContract(
            arm=arm,
            job_id=job_id if arm == _selected_arm(job) else f"{job_id}:{arm}",
            seed=seed,
            pre_replacement_checkpoint_sha256=pre_replacement_checkpoint_sha256,
            data_sha256=data_sha256,
            example_ids=example_ids,
            token_budget=_positive_int(budget, "tokens"),
            update_budget=_positive_int(budget, "updates"),
            curriculum=curriculum,
            optimizer=optimizer,
            schedule=schedule,
            stopping=stopping,
            eval_cells=tuple(str(length) for length in extrapolation),
            cache_match=None if arm == "native" else cache_match,
            selection_policy=(
                None if arm == "native" else "recency" if arm == "recency" else surprise_policy
            ),
        )
        for arm in ("native", "recency", "surprise")
    )
    return validate_three_arm_pairing(contracts)


def _batch_example_ids(batch: Mapping[str, object]) -> tuple[str, ...]:
    identifiers = batch.get("example_ids")
    if type(identifiers) is not tuple or not identifiers or any(
        type(item) is not str or not item for item in identifiers
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "every data window requires tuple example_ids"
        )
    return identifiers


def _batch_token_count(batch: Mapping[str, object]) -> int:
    input_ids = batch.get("input_ids")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "input_ids must be rank-2 torch.long"
        )
    labels = batch.get("labels")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.shape != input_ids.shape
    ):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "labels must match input_ids"
        )
    if len(_batch_example_ids(batch)) != input_ids.shape[0]:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "example_ids must match batch size"
        )
    return input_ids.numel()


def _validate_job_data(
    data: QwenJobData,
    *,
    config: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if not isinstance(data, QwenJobData):
        raise TypeError("data loader must return QwenJobData")
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    accumulation = params.get("accumulation_steps", 1)
    if type(accumulation) is not int or accumulation < 1:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "accumulation_steps must be positive"
        )
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    updates = _positive_int(budget, "updates")
    expected_microbatches = updates * accumulation
    if len(data.train_microbatches) != expected_microbatches:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid",
            "training windows do not exactly cover update x accumulation budget",
        )
    token_count = sum(_batch_token_count(batch) for batch in data.train_microbatches)
    if token_count != _positive_int(budget, "tokens"):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "training windows do not exactly match token budget"
        )
    windows = tuple(_batch_example_ids(batch) for batch in data.train_microbatches)
    flattened = tuple(item for window in windows for item in window)
    if len(flattened) != len(set(flattened)):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "paired Qwen example IDs must be globally unique"
        )
    preregistered = _string_tuple(
        params.get("example_ids"), "task.params.example_ids"
    )
    if flattened != preregistered:
        raise QwenRuntimeConfigurationError(
            "example_window_mismatch",
            "runtime data windows do not match preregistered example_ids order",
        )
    for batch in data.eval_microbatches:
        _batch_token_count(batch)
    return flattened, windows


def _digest_string(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", f"asset_hashes.{name} is not SHA-256"
        )
    return value


def _runtime_assets(
    runtime: Mapping[str, object], *, teacher_required: bool
) -> tuple[dict[str, object], dict[str, object]]:
    from .qwen_backend import ExternalAssetIdentity, validate_external_assets

    allowed = {
        "model",
        "tokenizer",
        "checkpoint",
        "data",
        "teacher_model",
        "output",
        "student_device",
        "teacher_device",
        "dtype",
        "asset_hashes",
        "resume",
        "checkpoint_every",
    }
    unknown = set(runtime) - allowed
    if unknown:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "unknown runtime keys: " + ", ".join(sorted(unknown)),
        )
    for name in ("model", "checkpoint", "data", "output", "student_device", "dtype"):
        if runtime.get(name) is None:
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid", f"runtime.{name} is required"
            )
    if teacher_required:
        for name in ("teacher_model", "teacher_device"):
            if runtime.get(name) is None:
                raise TeacherRequiredError(
                    "teacher_required", f"runtime.{name} is required for Qwen heal"
                )
    if type(runtime.get("student_device")) is not str or (
        runtime.get("teacher_device") is not None
        and type(runtime.get("teacher_device")) is not str
    ):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime devices must be strings"
        )
    if runtime.get("dtype") not in {"float32", "bfloat16"}:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.dtype must be float32 or bfloat16"
        )
    if type(runtime.get("resume")) is not bool:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "runtime.resume must be boolean"
        )
    checkpoint_every = runtime.get("checkpoint_every", 1)
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid", "checkpoint_every must be positive"
        )
    hashes = _required_mapping(runtime.get("asset_hashes"), "runtime.asset_hashes")
    asset_paths = {
        name: runtime[name]
        for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
        if runtime.get(name) is not None
    }
    if set(hashes) != set(asset_paths):
        raise QwenRuntimeConfigurationError(
            "runtime_configuration_invalid",
            "asset_hashes must exactly match supplied runtime asset paths",
        )
    specs = []
    for name, raw_path in asset_paths.items():
        try:
            path = Path(raw_path)
        except TypeError as error:
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid", f"runtime.{name} must be path-like"
            ) from error
        kind = "directory" if path.is_dir() else "file"
        specs.append(
            ExternalAssetIdentity(
                name=name,
                path=path,
                kind=kind,
                sha256=_digest_string(name, hashes[name]),
            )
        )
    validated = {asset.name: asset for asset in validate_external_assets(specs)}
    normalized = dict(runtime)
    normalized["output"] = Path(runtime["output"]).expanduser().resolve()
    normalized["checkpoint_every"] = checkpoint_every
    return normalized, validated


def _asset_spec(asset: object):
    from .qwen_backend import ExternalAssetIdentity, ValidatedAssetIdentity

    if not isinstance(asset, ValidatedAssetIdentity):
        raise TypeError("asset must be a ValidatedAssetIdentity")
    return ExternalAssetIdentity(
        name=asset.name,
        path=asset.path,
        kind=asset.kind,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
    )


def _training_parameter_names(
    config: Mapping[str, object], arm: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    memory = _string_tuple(
        params.get("memory_parameter_names"), "task.params.memory_parameter_names"
    )
    cache = _string_tuple(
        params.get("cache_parameter_names", ()),
        "task.params.cache_parameter_names",
        allow_empty=True,
    )
    return memory, cache if arm != "native" else ()


def _training_config(config: Mapping[str, object]) -> QwenHealTrainingConfig:
    task = _required_mapping(config.get("task"), "canonical_config.task")
    params = _required_mapping(task.get("params"), "canonical_config.task.params")
    budget = _required_mapping(config.get("budget"), "canonical_config.budget")
    return QwenHealTrainingConfig(
        objective=str(params.get("objective", "language_model_heal")),
        ce_weight=params.get("ce_weight", 0.1),
        kl_weight=params.get("kl_weight", 1.0),
        layerwise_weight=params.get("layerwise_weight", 0.0),
        temperature=params.get("temperature", 2.0),
        accumulation_steps=params.get("accumulation_steps", 1),
        max_updates=_positive_int(budget, "updates"),
        max_tokens=_positive_int(budget, "tokens"),
        gradient_checkpointing=params.get("gradient_checkpointing", True),
    )


def _default_load_data(*, asset: object, **_kwargs: object) -> QwenJobData:
    """Load a small explicit JSON/JSONL/PT window bundle for production smoke runs."""
    path = asset.path
    if path.is_dir():
        candidates = (path / "qwen_windows.pt", path / "qwen_windows.jsonl")
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", f"no supported Qwen window bundle at {path}"
        )
    if path.suffix == ".pt":
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise QwenRuntimeConfigurationError(
                "data_window_invalid", "Qwen .pt data is not a safe tensor bundle"
            ) from error
    elif path.suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "Qwen data must be .pt, .json, or .jsonl"
        )
    if isinstance(raw, Mapping):
        train_raw = raw.get("train")
        eval_raw = raw.get("eval", train_raw)
    else:
        train_raw = eval_raw = raw
    if not isinstance(train_raw, Sequence) or not isinstance(eval_raw, Sequence):
        raise QwenRuntimeConfigurationError(
            "data_window_invalid", "Qwen data bundle train/eval fields must be sequences"
        )

    def convert(record: object, index: int) -> Mapping[str, object]:
        if not isinstance(record, Mapping):
            raise QwenRuntimeConfigurationError(
                "data_window_invalid", f"data record {index} must be a mapping"
            )
        identifiers = record.get("example_ids")
        if identifiers is None:
            identifier = record.get("example_id", f"window-{index:08d}")
            identifiers = (identifier,)
        identifiers = tuple(identifiers)
        input_ids = torch.as_tensor(record.get("input_ids"), dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        labels_raw = record.get("labels")
        labels = input_ids.clone() if labels_raw is None else torch.as_tensor(
            labels_raw, dtype=torch.long
        )
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        converted: dict[str, object] = {
            "input_ids": input_ids,
            "labels": labels,
            "example_ids": identifiers,
        }
        tensor_annotations = {
            "query_mask": 2,
            "source_spans": 3,
            "stale_mask": 3,
        }
        for name, expected_rank in tensor_annotations.items():
            if name not in record:
                continue
            tensor = torch.as_tensor(record[name])
            if tensor.ndim == expected_rank - 1:
                tensor = tensor.unsqueeze(0)
            converted[name] = tensor
        if "stale_positions" in record:
            stale_positions = torch.as_tensor(
                record["stale_positions"], dtype=torch.int64
            )
            if stale_positions.numel() == 0:
                stale_positions = stale_positions.reshape(0, 3)
            converted["stale_positions"] = stale_positions
        if "ruler_metadata" in record:
            metadata = record["ruler_metadata"]
            if isinstance(metadata, Mapping):
                metadata = (metadata,)
            if (
                isinstance(metadata, (str, bytes, bytearray))
                or not isinstance(metadata, Sequence)
                or any(not isinstance(item, Mapping) for item in metadata)
            ):
                raise QwenRuntimeConfigurationError(
                    "ruler_annotations_invalid",
                    f"data record {index} ruler_metadata must contain mappings",
                )
            converted["ruler_metadata"] = tuple(
                copy.deepcopy(dict(item)) for item in metadata
            )
        return converted

    train = tuple(convert(record, index) for index, record in enumerate(train_raw))
    evaluation = tuple(convert(record, index) for index, record in enumerate(eval_raw))
    return QwenJobData(
        train_microbatches=train,
        eval_microbatches=evaluation,
        data_identity={
            "sha256": asset.sha256,
            "size_bytes": asset.size_bytes,
            "kind": asset.kind,
            "example_count": len(train),
        },
    )


def _default_load_teacher(*, asset: object, runtime: Mapping[str, object], **_kwargs: object):
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    dtype = torch.float32 if runtime["dtype"] == "float32" else torch.bfloat16
    teacher = AutoModelForCausalLM.from_pretrained(
        str(asset.path), torch_dtype=dtype, low_cpu_mem_usage=True
    )
    teacher.to(runtime["teacher_device"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _default_build_scheduler(
    *, optimizer: torch.optim.Optimizer, config: Mapping[str, object], **_kwargs: object
):
    schedule = _required_mapping(config.get("schedule"), "canonical_config.schedule")
    if schedule.get("name") != "cosine":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen heal requires cosine schedule"
        )
    warmup = schedule.get("warmup_updates")
    if type(warmup) is not int or warmup < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "warmup_updates must be nonnegative"
        )
    total = _positive_int(
        _required_mapping(config.get("budget"), "canonical_config.budget"), "updates"
    )

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _move_batch(
    batch: Mapping[str, object], device: str
) -> dict[str, object]:
    return {
        name: (
            value.to(device)
            if isinstance(value, torch.Tensor)
            and name not in _EVALUATION_ANNOTATION_FIELDS
            else value
        )
        for name, value in batch.items()
    }


_EVALUATION_ANNOTATION_FIELDS = {
    "query_mask",
    "source_spans",
    "stale_mask",
    "stale_positions",
    "ruler_metadata",
}

_RULER_METADATA_FIELDS = {
    "cell_id",
    "context_length",
    "needles",
    "queries",
    "depth_stratum",
    "example_id",
    "episode_id",
    "evaluation_mode",
    "evidence_scope",
    "seed",
    "example_index",
    "prompt_end",
    "answers",
    "answer_token_ids",
    "answer_spans",
    "source_spans",
    "depth_strata",
    "query_keys",
    "target_digest",
    "paired_interval",
}


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else math.fsum(values) / len(values)


def _checked_byte_sum(name: str, values: Sequence[object]) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                f"cache {name} byte counts must be nonnegative integers",
            )
        total += value
    return total


class _OnlineMean:
    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, *, count: int = 1) -> None:
        if not math.isfinite(value) or type(count) is not int or count < 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid", "streamed diagnostic value is invalid"
            )
        self.total = math.fsum((self.total, value))
        self.count += count

    def add_tensor(self, value: torch.Tensor) -> None:
        detached = value.detach()
        if detached.numel() == 0:
            return
        if not bool(torch.isfinite(detached).all()):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid", "streamed diagnostic tensor is nonfinite"
            )
        self.add(
            float(detached.double().sum().cpu()),
            count=detached.numel(),
        )

    def mean(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


class _StreamingTensorSequence:
    __slots__ = (
        "_dtype",
        "_label",
        "_hasher",
        "_sample_size",
        "count",
        "sample",
    )

    def __init__(
        self, *, dtype: torch.dtype, label: str, sample_size: int = 0
    ) -> None:
        self._dtype = dtype
        self._label = label
        self._hasher = hashlib.sha256()
        self.count = 0
        self.sample: list[int] = []
        self._sample_size = sample_size

    def add(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.detach().to(device="cpu", dtype=self._dtype).contiguous().reshape(-1)
        if normalized.numel():
            self._hasher.update(normalized.view(torch.uint8).numpy().tobytes())
            remaining = self._sample_size - len(self.sample)
            if remaining > 0:
                self.sample.extend(
                    int(item) for item in normalized[:remaining].tolist()
                )
            self.count += normalized.numel()
        return normalized

    def digest(self) -> str:
        envelope = hashlib.sha256()
        envelope.update(
            f"tensor-sequence-v1:{self._label}:{self._dtype}:{self.count}:".encode(
                "ascii"
            )
        )
        envelope.update(self._hasher.digest())
        return envelope.hexdigest()


class _StreamingScoreStatistics:
    __slots__ = ("sequence", "total", "minimum", "maximum")

    def __init__(self) -> None:
        self.sequence = _StreamingTensorSequence(
            dtype=torch.float32, label="cache-update-scores"
        )
        self.total = _OnlineMean()
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: torch.Tensor) -> None:
        normalized = self.sequence.add(value)
        if normalized.numel() == 0:
            return
        if not bool(torch.isfinite(normalized).all()) or bool((normalized < 0).any()):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache scores must be finite and nonnegative",
            )
        self.total.add_tensor(normalized)
        self.minimum = min(self.minimum, float(normalized.min()))
        self.maximum = max(self.maximum, float(normalized.max()))

    def as_dict(self) -> dict[str, object]:
        if self.sequence.count == 0:
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache evaluation produced no measured scores",
            )
        return {
            "count": self.sequence.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total.mean(),
        }


def _cache_amplitudes(model: torch.nn.Module) -> tuple[float, ...]:
    values: list[float] = []
    for name, parameter in sorted(model.named_parameters()):
        if name == "cache_amplitude" or name.endswith(".cache_amplitude"):
            values.extend(float(value) for value in parameter.detach().float().cpu().flatten())
    return tuple(values)


def _validate_evaluation_annotations(
    batch: Mapping[str, object],
    *,
    job: Mapping[str, object],
    require_cache: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Mapping[tuple[int, int], frozenset[int]],
    tuple[tuple[dict[str, object], object], ...],
]:
    config = _job_config(job)
    task = _required_mapping(config.get("task"), "canonical_config.task")
    is_ruler = task.get("name") == "ruler"
    required = {"query_mask", "source_spans"}
    missing = sorted(required - set(batch))
    stale_fields = {"stale_mask", "stale_positions"} & set(batch)
    if not stale_fields:
        missing.append("stale_mask or stale_positions")
    if missing:
        code = "cache_annotations_missing" if require_cache else "ruler_annotations_missing"
        raise QwenRuntimeConfigurationError(
            code,
            "annotated evaluation windows require: " + ", ".join(missing),
        )
    if len(stale_fields) != 1:
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid",
            "provide exactly one of stale_mask or stale_positions",
        )
    input_ids = batch.get("input_ids")
    query_mask = batch.get("query_mask")
    source_spans = batch.get("source_spans")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "input_ids must be rank-2 for evaluation"
        )
    batch_size, steps = input_ids.shape
    if (
        not isinstance(query_mask, torch.Tensor)
        or query_mask.dtype != torch.bool
        or query_mask.shape != (batch_size, steps)
    ):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "query_mask must be bool [B,T]"
        )
    if (
        not isinstance(source_spans, torch.Tensor)
        or source_spans.dtype != torch.int64
        or source_spans.shape != (batch_size, steps, 2)
    ):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "source_spans must be int64 [B,T,2]"
        )
    if not bool(query_mask.any()):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid", "evaluation requires at least one annotated query"
        )
    if bool((source_spans[~query_mask] != -1).any()):
        raise QwenRuntimeConfigurationError(
            "cache_annotations_invalid",
            "non-query positions require [-1,-1] source spans",
        )
    query_coordinates = tuple(
        (int(batch_index), int(token_index))
        for batch_index, token_index in torch.nonzero(
            query_mask, as_tuple=False
        ).detach().cpu().tolist()
    )
    stale_by_query: dict[tuple[int, int], set[int]] = {
        coordinate: set() for coordinate in query_coordinates
    }
    if "stale_positions" in stale_fields:
        stale_positions = batch.get("stale_positions")
        if (
            not isinstance(stale_positions, torch.Tensor)
            or stale_positions.dtype != torch.int64
            or stale_positions.ndim != 2
            or stale_positions.shape[1] != 3
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "stale_positions must be int64 [N,3] rows of batch/query/stale positions",
            )
        seen: set[tuple[int, int, int]] = set()
        for row in stale_positions.detach().cpu().tolist():
            batch_index, token_index, stale_position = (int(value) for value in row)
            triple = (batch_index, token_index, stale_position)
            coordinate = (batch_index, token_index)
            if triple in seen or coordinate not in stale_by_query:
                raise QwenRuntimeConfigurationError(
                    "cache_annotations_invalid",
                    "stale_positions must uniquely label declared query positions",
                )
            seen.add(triple)
            stale_by_query[coordinate].add(stale_position)
    else:
        stale_mask = batch.get("stale_mask")
        if (
            not isinstance(stale_mask, torch.Tensor)
            or stale_mask.dtype != torch.bool
            or stale_mask.shape != (batch_size, steps, steps)
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid", "stale_mask must be bool [B,T,T]"
            )
        if steps > 4096:
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "dense stale_mask is unsupported above 4096 tokens; use stale_positions",
            )
        if bool(stale_mask[~query_mask].any()):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "non-query positions cannot carry stale labels",
            )
        for coordinate in query_coordinates:
            batch_index, token_index = coordinate
            stale_by_query[coordinate].update(
                int(value)
                for value in torch.nonzero(
                    stale_mask[batch_index, token_index], as_tuple=False
                ).flatten().detach().cpu().tolist()
            )
    for batch_index, token_index in query_coordinates:
        start, stop = (
            int(value) for value in source_spans[batch_index, token_index]
        )
        stale = stale_by_query[(batch_index, token_index)]
        if token_index < 1 or not 0 <= start < stop <= token_index:
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "query source spans must be causal and nonempty",
            )
        if any(
            position < 0
            or position >= token_index
            or start <= position < stop
            for position in stale
        ):
            raise QwenRuntimeConfigurationError(
                "cache_annotations_invalid",
                "stale labels must be causal and disjoint from the gold source span",
            )
    frozen_stale = {
        coordinate: frozenset(values)
        for coordinate, values in stale_by_query.items()
    }

    if not is_ruler:
        return query_mask, source_spans, frozen_stale, ()
    raw_metadata = batch.get("ruler_metadata")
    if (
        not isinstance(raw_metadata, tuple)
        or len(raw_metadata) != batch_size
        or any(not isinstance(item, Mapping) for item in raw_metadata)
    ):
        raise QwenRuntimeConfigurationError(
            "ruler_annotations_missing",
            "RULER evaluation requires one ruler_metadata mapping per example",
        )
    from .tasks.ruler import RULER_DEPTH_STRATA, RulerCell, RulerEpisode

    example_ids = _batch_example_ids(batch)
    validated: list[tuple[dict[str, object], object]] = []
    for batch_index, raw in enumerate(raw_metadata):
        metadata = copy.deepcopy(dict(raw))
        if set(metadata) != _RULER_METADATA_FIELDS:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "ruler_metadata fields must exactly match the production schema",
            )
        if metadata["evaluation_mode"] != "teacher_forced":
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "default Qwen evaluation supports teacher_forced RULER rows only",
            )
        if metadata["evidence_scope"] not in {"feasibility", "promotion"}:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER evidence_scope is invalid"
            )
        if metadata["seed"] != job.get("seed") or metadata["example_id"] != example_ids[batch_index]:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER seed/example identity is mismatched"
            )
        if metadata["depth_stratum"] not in RULER_DEPTH_STRATA:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER depth_stratum is invalid"
            )
        if not isinstance(metadata["paired_interval"], Mapping):
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER paired_interval must be annotated"
            )
        cell = RulerCell(
            metadata["context_length"], metadata["needles"], metadata["queries"]
        )
        if metadata["cell_id"] != cell.cell_id:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER cell identity is inconsistent"
            )
        try:
            episode = RulerEpisode(
                episode_id=metadata["episode_id"],
                seed=metadata["seed"],
                example_index=metadata["example_index"],
                cell=cell,
                input_ids=tuple(int(value) for value in input_ids[batch_index].cpu()),
                prompt_end=metadata["prompt_end"],
                answers=tuple(metadata["answers"]),
                answer_token_ids=tuple(tuple(values) for values in metadata["answer_token_ids"]),
                answer_spans=tuple(tuple(values) for values in metadata["answer_spans"]),
                source_spans=tuple(tuple(values) for values in metadata["source_spans"]),
                depth_strata=tuple(metadata["depth_strata"]),
                query_keys=tuple(metadata["query_keys"]),
            )
        except (TypeError, ValueError) as error:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER episode metadata is malformed"
            ) from error
        expected_target_digest = _json_digest(
            [list(values) for values in episode.answer_token_ids]
        )
        if metadata["target_digest"] != expected_target_digest:
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid", "RULER target_digest is inconsistent"
            )
        expected_queries = torch.zeros(steps, dtype=torch.bool, device=query_mask.device)
        expected_spans = torch.full(
            (steps, 2), -1, dtype=torch.int64, device=source_spans.device
        )
        for answer_span, source_span in zip(episode.answer_spans, episode.source_spans):
            answer_start, answer_stop = answer_span
            expected_queries[answer_start:answer_stop] = True
            expected_spans[answer_start:answer_stop] = torch.tensor(
                source_span, dtype=torch.int64, device=source_spans.device
            )
        if not torch.equal(query_mask[batch_index], expected_queries) or not torch.equal(
            source_spans[batch_index], expected_spans
        ):
            raise QwenRuntimeConfigurationError(
                "ruler_annotations_invalid",
                "RULER query_mask/source_spans do not match answer/source spans",
            )
        validated.append((metadata, episode))
    return query_mask, source_spans, frozen_stale, tuple(validated)


def _default_evaluate(
    *,
    loaded_arm: object,
    data: QwenJobData,
    job: Mapping[str, object],
    runtime: Mapping[str, object],
    amplitude_initial: Sequence[float] = (),
    **_kwargs: object,
) -> dict[str, object]:
    from gdn3.kmd2_native import KMD2NativeAttn
    from .qwen_exact_cache import KMD2ExactCacheAttn, guarded_model_forward
    from .tasks.ruler import score_teacher_forced

    model = loaded_arm.model
    cache_layers = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, KMD2ExactCacheAttn)
    )
    require_cache = loaded_arm.arm != "native"
    native_layers = tuple(
        module for module in model.modules() if isinstance(module, KMD2NativeAttn)
    )
    state_elements = sum(module.H * module.dk * module.dv for module in native_layers)
    if state_elements < 1:
        config = _job_config(job)
        model_config = _required_mapping(config.get("model", {}), "canonical_config.model")
        dimensions = ("num_layers", "num_heads", "state_key_dim", "state_value_dim")
        if all(type(model_config.get(name)) is int for name in dimensions):
            state_elements = math.prod(int(model_config[name]) for name in dimensions)

    losses: list[float] = []
    correct = total = 0
    evaluations: list[dict[str, object]] = []
    selected_indices = _StreamingTensorSequence(
        dtype=torch.int64,
        label="cache-selected-indices",
        sample_size=32,
    )
    scores = _StreamingScoreStatistics()
    persistent_hits = _OnlineMean()
    conditional_correct = _OnlineMean()
    sinks = _OnlineMean()
    entropies = _OnlineMean()
    top1_masses = _OnlineMean()
    stale_flags = _OnlineMean()
    stale_errors = _OnlineMean()
    cache_norms = _OnlineMean()
    state_norms = _OnlineMean()
    retention_count = eviction_count = persistent_bytes = block_bytes = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in data.eval_microbatches:
                query_mask, source_spans, stale_by_query, ruler_rows = _validate_evaluation_annotations(
                    raw_batch,
                    job=job,
                    require_cache=require_cache,
                )
                if require_cache and not cache_layers:
                    raise QwenRuntimeConfigurationError(
                        "cache_diagnostics_unavailable",
                        "cache arm contains no installed KMD2 exact-cache layers",
                    )
                metric_names = (
                    "hits",
                    "conditional",
                    "sink",
                    "entropy",
                    "top1_mass",
                    "stale",
                    "stale_error",
                )
                row_accumulators = [
                    {name: _OnlineMean() for name in metric_names}
                    for _ in range(query_mask.shape[0])
                ]
                stale_error_records: list[tuple[int, int, bool]] = []
                layer_streams: dict[str, dict[str, object]] = {}

                def make_observer(layer_name: str):
                    nonlocal retention_count, eviction_count
                    state: dict[str, object] = {
                        "next_start": 0,
                        "blocks": 0,
                        "block_peak": 0,
                        "pending": None,
                    }
                    layer_streams[layer_name] = state

                    def observe(block: object) -> None:
                        nonlocal retention_count, eviction_count
                        block_start = getattr(block, "block_start", None)
                        block_stop = getattr(block, "block_stop", None)
                        if (
                            type(block_start) is not int
                            or type(block_stop) is not int
                            or block_start != state["next_start"]
                            or block_stop <= block_start
                        ):
                            raise QwenRuntimeConfigurationError(
                                "cache_diagnostics_invalid",
                                f"cache layer {layer_name} streamed block identity drifted",
                            )
                        state["next_start"] = block_stop
                        state["blocks"] = int(state["blocks"]) + 1
                        scores.add(block.update_scores)
                        cache_norms.add_tensor(block.cache_output_norm)
                        state_norms.add_tensor(block.state_output_norm)
                        sinks.add_tensor(block.sink_mass)
                        entropies.add_tensor(block.attention_entropy)
                        top1_masses.add_tensor(block.top1_mass)
                        if type(block.block_bytes) is not int or block.block_bytes < 0:
                            raise QwenRuntimeConfigurationError(
                                "cache_diagnostics_invalid",
                                "cache block byte counts must be nonnegative integers",
                            )
                        state["block_peak"] = max(
                            int(state["block_peak"]), block.block_bytes
                        )
                        prior_valid = int(
                            (block.persistent_selected_positions >= 0).sum()
                        )
                        batch_count, _, head_count = block.top1_positions.shape
                        incoming = (
                            batch_count * (block_stop - block_start) * head_count
                        )
                        pending = state["pending"]
                        if pending is not None:
                            previous_prior, previous_incoming = pending
                            retention_count += prior_valid
                            eviction_count += (
                                previous_prior + previous_incoming - prior_valid
                            )
                        state["pending"] = (prior_valid, incoming)

                        for (batch_index, target_position), stale in stale_by_query.items():
                            read_position = target_position - 1
                            if not block_start <= read_position < block_stop:
                                continue
                            local = read_position - block_start
                            source_start, source_stop = (
                                int(value)
                                for value in source_spans[
                                    batch_index, target_position
                                ]
                            )
                            gold = set(range(source_start, source_stop))
                            persistent_rows = block.persistent_selected_positions[
                                batch_index
                            ].detach().cpu()
                            top1_rows = block.top1_positions[
                                batch_index, local
                            ].detach().cpu()
                            candidate_rows = block.candidate_positions[
                                batch_index, local
                            ].detach().cpu()
                            candidate_valid_rows = block.candidate_valid[
                                batch_index, local
                            ].detach().cpu()
                            sink_rows = block.sink_mass[
                                batch_index, local
                            ].detach().float().cpu()
                            entropy_rows = block.attention_entropy[
                                batch_index, local
                            ].detach().float().cpu()
                            top1_mass_rows = block.top1_mass[
                                batch_index, local
                            ].detach().float().cpu()
                            accumulator = row_accumulators[batch_index]
                            for head in range(top1_rows.shape[0]):
                                persistent = {
                                    int(value)
                                    for value in persistent_rows[head].tolist()
                                    if value >= 0
                                }
                                hit = bool(persistent & gold)
                                top1 = int(top1_rows[head])
                                candidates = [
                                    int(value)
                                    for value in candidate_rows[head][
                                        candidate_valid_rows[head]
                                    ].tolist()
                                    if value >= 0
                                ]
                                stale_count = sum(
                                    value in stale for value in candidates
                                )
                                persistent_hits.add(float(hit))
                                accumulator["hits"].add(float(hit))
                                if hit:
                                    conditional = float(top1 in gold)
                                    conditional_correct.add(conditional)
                                    accumulator["conditional"].add(conditional)
                                stale_flags.add(
                                    float(stale_count), count=len(candidates)
                                )
                                accumulator["stale"].add(
                                    float(stale_count), count=len(candidates)
                                )
                                accumulator["sink"].add(float(sink_rows[head]))
                                accumulator["entropy"].add(
                                    float(entropy_rows[head])
                                )
                                accumulator["top1_mass"].add(
                                    float(top1_mass_rows[head])
                                )
                                stale_error_records.append(
                                    (batch_index, target_position, top1 in stale)
                                )

                    return observe

                for name, layer in cache_layers:
                    layer.set_cache_diagnostic_observer(
                        make_observer(name), retain_full=False
                    )
                try:
                    batch = _move_batch(raw_batch, runtime["student_device"])
                    inputs = {
                        name: value
                        for name, value in batch.items()
                        if name not in {"labels", "example_ids"} | _EVALUATION_ANNOTATION_FIELDS
                    }
                    inputs.update({"output_hidden_states": False, "use_cache": False})
                    output = guarded_model_forward(model, **inputs)
                finally:
                    for _, layer in cache_layers:
                        layer.set_cache_diagnostic_observer(None)
                logits = _validate_logits(
                    "evaluation logits", _output_field(output, "logits")
                )
                labels = batch["labels"]
                assert isinstance(labels, torch.Tensor)
                loss = causal_cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise QwenTrainingError("nonfinite_loss", "evaluation loss is nonfinite")
                losses.append(float(loss.cpu()))
                shifted_predictions = logits[:, :-1].argmax(dim=-1)
                targets = labels[:, 1:]
                valid = targets != -100
                correct += int(((shifted_predictions == targets) & valid).sum().cpu())
                total += int(valid.sum().cpu())
                aligned_predictions = torch.zeros_like(labels)
                aligned_predictions[:, 1:] = shifted_predictions
                layer_persistent_bytes: list[object] = []
                layer_block_bytes: list[object] = []
                for layer_name, layer in cache_layers:
                    diagnostics = layer.last_cache_diagnostics
                    stream = layer_streams[layer_name]
                    if (
                        diagnostics is None
                        or hasattr(diagnostics, "blocks")
                        or getattr(diagnostics, "blocks_processed", None)
                        != stream["blocks"]
                    ):
                        raise QwenRuntimeConfigurationError(
                            "cache_diagnostics_invalid",
                            f"cache layer {layer_name} omitted bounded synchronized diagnostics",
                        )
                    pending = stream["pending"]
                    if pending is None:
                        raise QwenRuntimeConfigurationError(
                            "cache_diagnostics_invalid",
                            f"cache layer {layer_name} streamed no blocks",
                        )
                    previous_prior, previous_incoming = pending
                    final_valid = int(diagnostics.final_selected_valid.sum())
                    retention_count += final_valid
                    eviction_count += previous_prior + previous_incoming - final_valid
                    selected_indices.add(
                        diagnostics.final_selected_positions[
                            diagnostics.final_selected_valid
                        ]
                    )
                    layer_persistent_bytes.append(diagnostics.persistent_bytes)
                    layer_block_bytes.append(stream["block_peak"])

                for batch_index, target_position, top1_is_stale in stale_error_records:
                    wrong = bool(
                        aligned_predictions[batch_index, target_position].cpu()
                        != labels[batch_index, target_position].cpu()
                    )
                    stale_error = float(wrong and top1_is_stale)
                    stale_errors.add(stale_error)
                    row_accumulators[batch_index]["stale_error"].add(stale_error)

                persistent_bytes = max(
                    persistent_bytes,
                    _checked_byte_sum("persistent", layer_persistent_bytes),
                )
                block_bytes = max(
                    block_bytes,
                    _checked_byte_sum("block", layer_block_bytes),
                )

                for batch_index, (metadata, episode) in enumerate(ruler_rows):
                    score = score_teacher_forced(
                        episode,
                        [int(value) for value in aligned_predictions[batch_index].cpu()],
                    )
                    if require_cache:
                        accumulator = row_accumulators[batch_index]
                        cache_diagnostics: dict[str, object] = {
                            "active": True,
                            "persistent_hit": accumulator["hits"].mean(),
                            "persistent_hit_count": accumulator["hits"].count,
                            "conditional_read": accumulator["conditional"].mean(),
                            "conditional_read_count": accumulator["conditional"].count,
                            "sink_mass": accumulator["sink"].mean(),
                            "attention_entropy": accumulator["entropy"].mean(),
                            "top1_mass": accumulator["top1_mass"].mean(),
                            "stale_occupancy": accumulator["stale"].mean(),
                            "stale_error": accumulator["stale_error"].mean(),
                        }
                    else:
                        cache_diagnostics = {"active": False}
                    evaluations.append(
                        {
                            "task": "ruler",
                            "cell_id": metadata["cell_id"],
                            "context_length": metadata["context_length"],
                            "needles": metadata["needles"],
                            "queries": metadata["queries"],
                            "depth_stratum": metadata["depth_stratum"],
                            "example_id": metadata["example_id"],
                            "episode_id": metadata["episode_id"],
                            "evaluation_mode": score.evaluation_mode,
                            "evidence_scope": metadata["evidence_scope"],
                            "numerator": score.numerator,
                            "denominator": score.denominator,
                            "episode_exact": score.episode_exact,
                            "source_spans": [list(span) for span in episode.source_spans],
                            "target_digest": metadata["target_digest"],
                            "cache_diagnostics": cache_diagnostics,
                            "paired_interval": copy.deepcopy(dict(metadata["paired_interval"])),
                            "seed": job["seed"],
                            "arm_id": job["arm_id"],
                        }
                    )
    finally:
        model.train(was_training)

    if not losses:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluation requires at least one batch"
        )
    result: dict[str, object] = {
        "metrics": {
            "eval_loss": math.fsum(losses) / len(losses),
            "token_accuracy": correct / max(1, total),
        },
        "recurrent_state": {"elements": state_elements, "bytes": 4 * state_elements},
    }
    if evaluations:
        result["evaluations"] = evaluations
    if require_cache:
        initial = tuple(float(value) for value in amplitude_initial)
        final = _cache_amplitudes(model)
        if not initial or len(initial) != len(final):
            raise QwenRuntimeConfigurationError(
                "cache_diagnostics_invalid",
                "cache amplitude initial/final measurements are incomplete",
            )
        cache_config = cache_layers[0][1].cache_config
        result["exact_cache"] = {
            "width": cache_config.width,
            "block_size": cache_config.block_size,
            "score_definition": cache_config.score,
            "compute_dtype": cache_config.compute_dtype,
            "storage_dtype": cache_config.storage_dtype,
            "coordinate_frame": cache_config.coordinate_frame,
            "inclusive_causality": cache_config.inclusive,
            "tie_policy": cache_config.tie_policy,
            "amplitude_initial": list(initial),
            "amplitude_final": list(final),
            "selected_index_digest": selected_indices.digest(),
            "selected_index_sample": selected_indices.sample,
            "score_digest": scores.sequence.digest(),
            "score_statistics": scores.as_dict(),
            "retention_count": retention_count,
            "eviction_count": eviction_count,
            "persistent_hit_rate": persistent_hits.mean(),
            "conditional_read_accuracy": conditional_correct.mean(),
            "sink_mass": sinks.mean(),
            "attention_entropy": entropies.mean(),
            "top1_mass": top1_masses.mean(),
            "stale_occupancy": stale_flags.mean(),
            "stale_error": stale_errors.mean(),
            "cache_output_norm": cache_norms.mean(),
            "state_output_norm": state_norms.mean(),
            "persistent_bytes": persistent_bytes,
            "block_bytes": block_bytes,
            "implementation_paths": {
                "scan": "gdn3.kmd2_native.KMD2NativeAttn.forward",
                "score": (
                    "qwen_backend.KMD2RecencyCacheAttn.position"
                    if cache_config.score == "recency"
                    else "qwen_exact_cache.KMD2ExactCacheAttn._native_state_and_scores"
                ),
                "selection": "exact_cache.merge_persistent_cache.deterministic_topw",
                "read": f"exact_cache.cache_read_blocks.{cache_config.read}",
            },
        }
    return result


def _default_peak_vram_bytes(device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(torch.device(device)))
    return 0


def _default_reset_peak_vram(device: str) -> None:
    """Start one job's peak window at its fully loaded resident baseline."""
    if device.startswith("cuda") and torch.cuda.is_available():
        resolved = torch.device(device)
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)


@dataclass(frozen=True)
class QwenExecutionDependencies:
    """Injectable heavy boundaries used by :func:`execute_job`."""

    load_arm: Callable[..., object]
    load_teacher: Callable[..., torch.nn.Module]
    load_data: Callable[..., QwenJobData]
    build_optimizer: Callable[..., torch.optim.Optimizer]
    build_scheduler: Callable[..., object]
    load_checkpoint: Callable[..., object]
    save_checkpoint: Callable[..., Path]
    evaluate: Callable[..., Mapping[str, object]]
    monotonic: Callable[[], float]
    reset_peak_vram: Callable[[str], None]
    peak_vram_bytes: Callable[[str], int]


def _default_dependencies() -> QwenExecutionDependencies:
    from .qwen_backend import load_qwen_arm
    from .qwen_checkpoint import load_qwen_checkpoint, save_qwen_checkpoint

    return QwenExecutionDependencies(
        load_arm=load_qwen_arm,
        load_teacher=_default_load_teacher,
        load_data=_default_load_data,
        build_optimizer=build_qwen_heal_optimizer,
        build_scheduler=_default_build_scheduler,
        load_checkpoint=load_qwen_checkpoint,
        save_checkpoint=save_qwen_checkpoint,
        evaluate=_default_evaluate,
        monotonic=time.monotonic,
        reset_peak_vram=_default_reset_peak_vram,
        peak_vram_bytes=_default_peak_vram_bytes,
    )


def _resolve_dependencies(
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None,
) -> QwenExecutionDependencies:
    defaults = _default_dependencies()
    if dependencies is None:
        return defaults
    if isinstance(dependencies, QwenExecutionDependencies):
        return dependencies
    if not isinstance(dependencies, Mapping):
        raise TypeError("dependencies must be QwenExecutionDependencies, a mapping, or None")
    fields = tuple(QwenExecutionDependencies.__dataclass_fields__)
    unknown = set(dependencies) - set(fields)
    if unknown:
        raise ValueError("unknown Qwen dependencies: " + ", ".join(sorted(unknown)))
    values = {
        name: dependencies.get(name, getattr(defaults, name)) for name in fields
    }
    if any(not callable(value) for value in values.values()):
        raise TypeError("every Qwen execution dependency must be callable")
    return QwenExecutionDependencies(**values)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
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
    )
    result: dict[str, str] = {}
    for name in relative_paths:
        path = root / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def _identity_record(asset: object) -> dict[str, object]:
    return {
        "kind": asset.kind,
        "size_bytes": asset.size_bytes,
        "sha256": asset.sha256,
    }


def _relevant_cuda_rng_devices(runtime: Mapping[str, object]) -> tuple[int, ...]:
    if not torch.cuda.is_available():
        return ()
    devices: set[int] = set()
    for name in ("student_device", "teacher_device"):
        raw = runtime.get(name)
        if type(raw) is not str:
            continue
        try:
            device = torch.device(raw)
        except (TypeError, RuntimeError):
            continue
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise QwenRuntimeConfigurationError(
                "runtime_configuration_invalid",
                f"runtime.{name} names unavailable CUDA device {index}",
            )
        devices.add(index)
    return tuple(sorted(devices))


@contextmanager
def _scoped_paired_rng(seed: int, runtime: Mapping[str, object]):
    """Seed one job transaction and restore caller RNGs on every exit path."""
    python_state = random.getstate()
    cuda_devices = _relevant_cuda_rng_devices(runtime)
    try:
        with torch.random.fork_rng(devices=list(cuda_devices), enabled=True):
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if torch.cuda.is_available():
                for device_index in cuda_devices:
                    torch.cuda.default_generators[device_index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


def execute_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one paired job inside a fully restoring stochastic transaction."""
    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    seed = job.get("seed")
    if type(seed) is not int or seed < 0:
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "job.seed must be a nonnegative integer"
        )
    with _scoped_paired_rng(seed, runtime):
        return _execute_job_seeded(job, runtime=runtime, dependencies=dependencies)


def _execute_job_seeded(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one bound Qwen heal job and return runner-ready diagnostics."""
    from dataclasses import replace

    from .config import CacheConfig
    from .qwen_backend import (
        LoadedQwenArm,
        PairingContractError,
        QwenArmLoadSpec,
    )
    from .qwen_checkpoint import (
        QwenCheckpointMetadata,
        QwenResumeExpectation,
    )

    if not isinstance(job, Mapping):
        raise TypeError("job must be a mapping")
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    config = _job_config(job)
    training = _training_config(config)
    validate_teacher_requirement(
        training,
        teacher_present=runtime.get("teacher_model") is not None,
        phase="preflight",
    )
    runtime_values, assets = _runtime_assets(
        runtime, teacher_required=training.objective != "synthetic_only"
    )
    dependencies_value = _resolve_dependencies(dependencies)
    started = dependencies_value.monotonic()

    data = dependencies_value.load_data(
        asset=assets["data"], job=job, runtime=runtime_values
    )
    example_ids, expected_windows = _validate_job_data(data, config=config)
    if data.data_identity.get("sha256") != assets["data"].sha256:
        raise QwenRuntimeConfigurationError(
            "data_identity_mismatch",
            "loaded data identity does not match the measured runtime asset",
        )
    pairing = derive_three_arm_pairing(
        job,
        example_ids=example_ids,
        pre_replacement_checkpoint_sha256=assets["checkpoint"].sha256,
        data_sha256=assets["data"].sha256,
    )
    if job.get("pairing_id") != pairing.pairing_id:
        raise PairingContractError(
            "pairing_id_mismatch", "job pairing_id does not match three-arm contract"
        )
    paired_starts = {arm.arm: assets["checkpoint"].sha256 for arm in pairing.arms}
    if len(set(paired_starts.values())) != 1:
        raise PairingContractError(
            "checkpoint_identity_mismatch", "three Qwen arms do not share one checkpoint"
        )

    arm = _selected_arm(job)
    memory_names, cache_names = _training_parameter_names(config, arm)
    cache_mapping = _required_mapping(config.get("cache"), "canonical_config.cache")
    cache_config = CacheConfig(**dict(cache_mapping)) if arm != "native" else None
    if arm == "recency":
        assert cache_config is not None
        cache_config = replace(cache_config, score="recency")
    dtype = torch.float32 if runtime_values["dtype"] == "float32" else torch.bfloat16
    spec = QwenArmLoadSpec(
        arm=arm,
        job_id=job["job_id"],
        model_asset=_asset_spec(assets["model"]),
        native_checkpoint=_asset_spec(assets["checkpoint"]),
        data_asset=_asset_spec(assets["data"]),
        cache_resume=None,
        trainable_names=memory_names + cache_names,
        pre_replacement_checkpoint_sha256=assets["checkpoint"].sha256,
        model_loader_kwargs={"torch_dtype": dtype, "low_cpu_mem_usage": True},
    )
    loaded = dependencies_value.load_arm(
        spec, model_config=None, cache_config=cache_config
    )
    if not isinstance(loaded, LoadedQwenArm) or loaded.arm != arm:
        raise TypeError("Qwen arm loader returned an incompatible result")
    loaded.model.to(runtime_values["student_device"])
    amplitude_initial = _cache_amplitudes(loaded.model)

    teacher = None
    if training.objective != "synthetic_only":
        teacher = dependencies_value.load_teacher(
            asset=assets["teacher_model"], job=job, runtime=runtime_values
        )
        if not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher loader must return a torch.nn.Module")

    optimizer_mapping = _required_mapping(
        config.get("optimizer"), "canonical_config.optimizer"
    )
    if optimizer_mapping.get("name") != "adamw":
        raise QwenRuntimeConfigurationError(
            "job_configuration_invalid", "Qwen heal requires AdamW"
        )
    optimizer = dependencies_value.build_optimizer(
        loaded.model,
        memory_parameter_names=memory_names,
        cache_parameter_names=cache_names,
        learning_rate=optimizer_mapping.get("learning_rate"),
        lr_cache=cache_mapping.get("lr_cache"),
        betas=tuple(optimizer_mapping.get("betas", ())),
        eps=optimizer_mapping.get("eps"),
        weight_decay=optimizer_mapping.get("weight_decay"),
    )
    scheduler = dependencies_value.build_scheduler(
        optimizer=optimizer, config=config, job=job
    )
    moved_train = tuple(
        _move_batch(batch, runtime_values["student_device"])
        for batch in data.train_microbatches
    )
    trainer = QwenHealTrainer(
        model=loaded.model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training,
        job_id=job["job_id"],
        pairing_id=pairing.pairing_id,
        arm=arm,
        expected_example_windows=expected_windows,
        teacher_device=(
            runtime_values.get("teacher_device") if teacher is not None else None
        ),
    )
    target_module_names = tuple(
        sorted(f"model.layers.{index}.linear_attn" for index in loaded.upgraded_indices)
    )
    source_hashes = _source_hashes()
    source_hashes.update(
        {f"asset:{name}": asset.sha256 for name, asset in sorted(assets.items())}
    )
    promotion = config.get("promotion")
    if not isinstance(promotion, Mapping) or not promotion:
        promotion = cache_mapping
    metadata_kwargs = {
        "job_id": job["job_id"],
        "pairing_id": pairing.pairing_id,
        "arm": arm,
        "source_hashes": source_hashes,
        "data_identity": data.data_identity,
        "example_ids": example_ids,
        "promotion_config": promotion,
    }
    checkpoint_path = (
        runtime_values["output"] / "checkpoints" / job["job_id"] / "latest.pt"
    )
    if runtime_values["resume"] and checkpoint_path.is_file():
        expectation = QwenResumeExpectation(**metadata_kwargs)
        resumed = dependencies_value.load_checkpoint(
            checkpoint_path,
            model=loaded.model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=expectation,
            target_module_names=target_module_names,
        )
        expected_resume_identity = {
            "job_id": job["job_id"],
            "pairing_id": pairing.pairing_id,
            "arm": arm,
        }
        if any(
            getattr(resumed, name, None) != expected
            for name, expected in expected_resume_identity.items()
        ):
            raise QwenRuntimeConfigurationError(
                "resume_identity_mismatch",
                "checkpoint loader returned inconsistent job/pair/arm identity",
            )
        if (
            type(getattr(resumed, "step", None)) is not int
            or type(getattr(resumed, "tokens_seen", None)) is not int
            or resumed.step < 0
            or resumed.tokens_seen < 0
        ):
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "checkpoint progress is malformed"
            )
        if resumed.step > training.max_updates:
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "resume step exceeds configured update budget"
            )
        cursor = resumed.step * training.accumulation_steps
        prefix_tokens = sum(
            _batch_token_count(batch) for batch in moved_train[:cursor]
        )
        if resumed.tokens_seen != prefix_tokens:
            raise QwenRuntimeConfigurationError(
                "resume_progress_invalid", "resume token progress does not match data windows"
            )
        trainer.step = resumed.step
        trainer.tokens_seen = resumed.tokens_seen
        trainer.example_cursor = cursor

    # Loading, optional resume, and setup are outside the measured peak window;
    # the reset baseline still includes every resident runtime tensor.
    dependencies_value.reset_peak_vram(runtime_values["student_device"])

    logs: list[HealStepLog] = []
    checkpoint_every = runtime_values["checkpoint_every"]
    while trainer.step < training.max_updates:
        start = trainer.example_cursor
        stop = start + training.accumulation_steps
        log = trainer.train_update(moved_train[start:stop])
        logs.append(log)
        if trainer.step % checkpoint_every == 0 or trainer.step == training.max_updates:
            metadata = QwenCheckpointMetadata(
                step=trainer.step,
                tokens_seen=trainer.tokens_seen,
                **metadata_kwargs,
            )
            dependencies_value.save_checkpoint(
                checkpoint_path,
                model=loaded.model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
                target_module_names=target_module_names,
            )

    evaluation = dependencies_value.evaluate(
        loaded_arm=loaded,
        data=data,
        job=job,
        runtime=runtime_values,
        amplitude_initial=amplitude_initial,
    )
    if not isinstance(evaluation, Mapping):
        raise TypeError("Qwen evaluator must return a mapping")
    metrics = evaluation.get("metrics")
    recurrent_state = evaluation.get("recurrent_state")
    if not isinstance(metrics, Mapping) or not metrics:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluator metrics must be nonempty"
        )
    if not isinstance(recurrent_state, Mapping) or set(recurrent_state) != {
        "elements",
        "bytes",
    }:
        raise QwenRuntimeConfigurationError(
            "evaluation_invalid", "Qwen evaluator recurrent_state is incomplete"
        )
    finished = dependencies_value.monotonic()
    wall_time = finished - started
    if not math.isfinite(wall_time) or wall_time < 0.0:
        raise QwenRuntimeConfigurationError(
            "clock_invalid", "monotonic execution duration is invalid"
        )
    duration = max(wall_time, 1.0e-12)
    loss_curves = {
        name: [float(log.losses[name]) for log in logs]
        for name in ("total", "ce", "kl", "layerwise")
    }
    trainable = sum(
        parameter.numel() for parameter in loaded.model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in loaded.model.parameters())
    payload: dict[str, object] = {
        "metrics": dict(metrics),
        "loss_curves": loss_curves,
        "counts": {
            "nonfinite_loss": 0,
            "nonfinite_gradient": 0,
            "skipped_steps": trainer.skipped_steps,
        },
        "parameters": {"trainable": trainable, "total": total_parameters},
        "recurrent_state": dict(recurrent_state),
        "performance": {
            "wall_time_seconds": wall_time,
            "examples_per_second": len(example_ids) / duration,
            "tokens_per_second": trainer.tokens_seen / duration,
            "peak_vram_bytes": dependencies_value.peak_vram_bytes(
                runtime_values["student_device"]
            ),
        },
        "identities": {
            "model": _identity_record(assets["model"]),
            "checkpoint": _identity_record(assets["checkpoint"]),
            "data": _identity_record(assets["data"]),
            "paired_starts": paired_starts,
            **(
                {"teacher_model": _identity_record(assets["teacher_model"])}
                if teacher is not None
                else {}
            ),
        },
    }
    if "evaluations" in evaluation:
        payload["evaluations"] = evaluation["evaluations"]
    if arm != "native":
        exact_cache = evaluation.get("exact_cache")
        if not isinstance(exact_cache, Mapping):
            raise QwenRuntimeConfigurationError(
                "evaluation_invalid", "cache evaluator omitted exact_cache diagnostics"
            )
        payload["exact_cache"] = dict(exact_cache)
    return payload


def run_job(
    job: Mapping[str, object],
    *,
    runtime: Mapping[str, object] | None = None,
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Runner-discoverable entry point; runtime paths must be explicitly bound."""
    if runtime is None:
        raise QwenRuntimeConfigurationError(
            "runtime_required",
            "Qwen run_job requires build_job_dispatcher(runtime, dependencies)",
        )
    return execute_job(job, runtime=runtime, dependencies=dependencies)


def build_job_dispatcher(
    runtime: Mapping[str, object],
    dependencies: QwenExecutionDependencies | Mapping[str, object] | None = None,
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    """Bind non-semantic runtime state into the runner's one-argument protocol."""
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    frozen_runtime = dict(runtime)
    resolved_dependencies = _resolve_dependencies(dependencies)

    def dispatch(job: Mapping[str, object]) -> Mapping[str, object]:
        return run_job(
            job,
            runtime=frozen_runtime,
            dependencies=resolved_dependencies,
        )

    dispatch.__name__ = "run_bound_qwen_job"
    return dispatch


__all__ = [
    "HealLossBreakdown",
    "HealStepLog",
    "QwenExecutionDependencies",
    "QwenHealTrainer",
    "QwenHealTrainingConfig",
    "QwenJobData",
    "QwenRuntimeConfigurationError",
    "QwenTrainingError",
    "TeacherRequiredError",
    "build_job_dispatcher",
    "build_qwen_heal_optimizer",
    "causal_cross_entropy",
    "compute_heal_loss",
    "distillation_kl",
    "derive_three_arm_pairing",
    "execute_job",
    "layerwise_alignment_loss",
    "project_cache_amplitudes_",
    "run_job",
    "validate_teacher_requirement",
]
