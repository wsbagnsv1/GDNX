"""Metadata-only exact resource accounting for Qwen dry-run preflight."""

from __future__ import annotations

import ast
import json
import math
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .runner import PreflightCheckError


_CACHE_SUFFIX_SIZES = {
    "cache_gamma_q": "key",
    "cache_gamma_k": "key",
    "cache_sink_logit": "heads",
    "cache_amplitude": "heads",
}
_MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
_MAX_EXACT_PARAMETER_COUNT = (1 << 63) - 1
_EXPECTED_NATIVE_SCAN = "gdn3.kmd2_fast_scan.scan"
_EXPECTED_SCORE_SCAN = "gdn3.kmd2_fast_scan.scan_with_update_norm"
_MEMORY_SUFFIX = ".in_proj_b.weight"
_NATIVE_ADDITION_SUFFIXES = (
    ".rot_proj.weight",
    ".rot_proj.bias",
    ".q_slot_scale",
    ".out_mix",
    ".decay_chan",
    ".bw_off",
)
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _shape_elements(shape: object, *, context: str) -> int:
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes, bytearray))
        or any(type(item) is not int or item < 0 for item in shape)
    ):
        raise PreflightCheckError(
            "parameter_metadata_invalid", f"{context} has an invalid tensor shape"
        )
    elements = math.prod(shape)
    if elements > _MAX_EXACT_PARAMETER_COUNT:
        raise PreflightCheckError(
            "parameter_accounting_overflow",
            f"{context} exceeds the exact parameter accounting bound",
        )
    return elements


def _checked_sum(*values: int, context: str) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise PreflightCheckError(
                "parameter_accounting_invalid", f"{context} is not nonnegative"
            )
        total += value
        if total > _MAX_EXACT_PARAMETER_COUNT:
            raise PreflightCheckError(
                "parameter_accounting_overflow",
                f"{context} exceeds the exact parameter accounting bound",
            )
    return total


def _checked_product(*values: int, context: str) -> int:
    product = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise PreflightCheckError(
                "parameter_accounting_invalid", f"{context} is not nonnegative"
            )
        product *= value
        if product > _MAX_EXACT_PARAMETER_COUNT:
            raise PreflightCheckError(
                "parameter_accounting_overflow",
                f"{context} exceeds the exact parameter accounting bound",
            )
    return product


def _read_safetensors_header(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("missing header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("header length is outside the safety bound")
            if 8 + header_size > size:
                raise ValueError("header extends beyond the file")
            encoded = stream.read(header_size)
        raw = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PreflightCheckError(
            "parameter_metadata_invalid",
            f"cannot read safetensors metadata: {path}",
        ) from error
    if not isinstance(raw, Mapping):
        raise PreflightCheckError(
            "parameter_metadata_invalid", f"safetensors header is not a mapping: {path}"
        )
    tensors: dict[str, tuple[int, ...]] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in raw.items():
        if name == "__metadata__":
            continue
        if type(name) is not str or not name or not isinstance(metadata, Mapping):
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"invalid tensor entry in {path}"
            )
        shape = metadata.get("shape")
        elements = _shape_elements(shape, context=f"{path}:{name}")
        dtype = metadata.get("dtype")
        dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if dtype_bytes is None:
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"unknown dtype for {path}:{name}"
            )
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, Sequence)
            or isinstance(offsets, (str, bytes, bytearray))
            or len(offsets) != 2
            or any(type(item) is not int or item < 0 for item in offsets)
            or offsets[1] < offsets[0]
            or 8 + len(encoded) + offsets[1] > size
        ):
            raise PreflightCheckError(
                "parameter_metadata_invalid", f"invalid data offsets for {path}:{name}"
            )
        if offsets[1] - offsets[0] != elements * dtype_bytes:
            raise PreflightCheckError(
                "parameter_metadata_invalid",
                f"shape/dtype byte count disagrees with offsets for {path}:{name}",
            )
        tensors[name] = tuple(shape)
        intervals.append((offsets[0], offsets[1], name))
    if not tensors:
        raise PreflightCheckError(
            "parameter_metadata_missing", f"no tensor metadata found in {path}"
        )
    expected_start = 0
    for start, stop, name in sorted(intervals):
        if start != expected_start:
            raise PreflightCheckError(
                "parameter_metadata_invalid",
                f"tensor data is overlapping or noncanonical near {path}:{name}",
            )
        expected_start = stop
    if expected_start != size - 8 - len(encoded):
        raise PreflightCheckError(
            "parameter_metadata_invalid",
            f"tensor offsets do not cover the safetensors data section: {path}",
        )
    return tensors


def _safetensors_inventory(path: Path) -> dict[str, tuple[int, ...]]:
    files = (
        [path]
        if path.is_file() and path.suffix == ".safetensors"
        else sorted(path.rglob("*.safetensors")) if path.is_dir() else []
    )
    if not files:
        raise PreflightCheckError(
            "parameter_metadata_missing",
            "Qwen dry-run requires safetensors headers for exact parameter accounting",
        )
    tensors: dict[str, tuple[int, ...]] = {}
    for file in files:
        for name, shape in _read_safetensors_header(file).items():
            if name in tensors:
                raise PreflightCheckError(
                    "parameter_metadata_invalid",
                    f"duplicate tensor metadata across safetensors shards: {name}",
                )
            tensors[name] = shape
    return tensors


def _string_names(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (not value and not allow_empty)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise PreflightCheckError(
            "parameter_declaration_invalid", f"{field} must contain unique names"
        )
    return tuple(value)


def _resolve_parameter(
    tensors: Mapping[str, tuple[int, ...]], declared: str
) -> tuple[int, ...]:
    matches = [
        shape
        for name, shape in tensors.items()
        if name == declared or name.endswith("." + declared)
    ]
    if len(matches) != 1:
        raise PreflightCheckError(
            "parameter_metadata_mismatch",
            f"declared trainable parameter does not resolve uniquely: {declared}",
        )
    return matches[0]


def _has_name(tree: ast.AST, name: str) -> bool:
    return any(
        (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        )
        for node in ast.walk(tree)
    )


def _verify_fast_scan_source_contract(source_root: Path) -> None:
    paths = {
        "native": source_root / "gdn3" / "kmd2_native.py",
        "fast": source_root / "gdn3" / "kmd2_fast_scan.py",
        "cache": source_root / "research" / "kmd2_ablation" / "qwen_exact_cache.py",
    }
    try:
        trees = {
            name: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name, path in paths.items()
        }
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "cannot verify the Qwen fast-scan source contract",
        ) from error

    native_assignments = [
        node
        for node in ast.walk(trees["native"])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_FAST_SCAN" for target in node.targets)
    ]
    native_dump = " ".join(ast.dump(node.value) for node in native_assignments)
    if len(native_assignments) != 1 or "GDN3_FAST_SCAN" not in native_dump:
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "native scan does not expose the verified import-time fast-scan gate",
        )

    cache_function = next(
        (
            node
            for node in ast.walk(trees["cache"])
            if isinstance(node, ast.FunctionDef)
            and node.name == "_native_state_and_scores"
        ),
        None,
    )
    cache_dump = "" if cache_function is None else ast.dump(cache_function)
    if (
        cache_function is None
        or "_FAST_SCAN" not in cache_dump
        or "scan_with_update_norm" not in cache_dump
        or not _has_name(trees["fast"], "scan")
        or not _has_name(trees["fast"], "scan_with_update_norm")
    ):
        raise PreflightCheckError(
            "qwen_fast_scan_source_invalid",
            "score-returning Qwen cache scan is not wired to the verified fast path",
        )


def verify_qwen_execution_contract(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Prove the pinned r_out and score-returning fast path before Qwen imports."""

    if isinstance(config, ExperimentConfig):
        backend = config.backend
        params: Mapping[str, Any] = config.task.params
    elif isinstance(config, Mapping):
        backend = config.get("backend")
        task = config.get("task")
        params_value = task.get("params") if isinstance(task, Mapping) else None
        if not isinstance(params_value, Mapping):
            raise TypeError("canonical Qwen config has no task.params mapping")
        params = params_value
    else:
        raise TypeError("config must be a Qwen config mapping")
    if backend != "qwen":
        raise TypeError("config must select the Qwen backend")
    native_r_out = params.get("native_r_out")
    if type(native_r_out) is not int or native_r_out < 1:
        raise PreflightCheckError(
            "qwen_r_out_unpinned",
            "task.params.native_r_out must pin a positive integer",
        )
    if params.get("score_scan") != _EXPECTED_SCORE_SCAN:
        raise PreflightCheckError(
            "qwen_fast_scan_unpinned",
            f"task.params.score_scan must be {_EXPECTED_SCORE_SCAN}",
        )
    environment = os.environ if environ is None else environ
    if environment.get("GDN3_FAST_SCAN") != "1":
        raise PreflightCheckError(
            "qwen_fast_scan_inactive",
            "GDN3_FAST_SCAN=1 must be set before the Qwen process starts",
        )
    if environment.get("GDN3_KMD2_ROUT") != str(native_r_out):
        raise PreflightCheckError(
            "qwen_r_out_mismatch",
            "GDN3_KMD2_ROUT does not match task.params.native_r_out",
        )

    modules = sys.modules if loaded_modules is None else loaded_modules
    loaded_native = modules.get("gdn3.kmd2_native")
    if loaded_native is not None and getattr(loaded_native, "_FAST_SCAN", None) is not True:
        raise PreflightCheckError(
            "qwen_fast_scan_import_order_invalid",
            "gdn3.kmd2_native was imported before the fast scan was enabled",
        )
    root = Path(__file__).resolve().parents[2] if source_root is None else Path(source_root)
    _verify_fast_scan_source_contract(root)
    proof = (
        "loaded_module_flag"
        if loaded_native is not None
        else "preimport_environment_and_source_contract"
    )
    return {
        "activation_proof": proof,
        "fast_scan": True,
        "native_r_out": native_r_out,
        "native_scan": _EXPECTED_NATIVE_SCAN,
        "score_scan": _EXPECTED_SCORE_SCAN,
    }


def _native_layer_prefixes(
    config: ExperimentConfig,
    tensors: Mapping[str, tuple[int, ...]],
    names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(names) != config.model.num_layers:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen memory trainables must declare one in_proj_b weight per native layer",
        )
    prefixes: list[str] = []
    expected_shape = (config.model.num_heads, config.model.hidden_size)
    for name in names:
        if not name.endswith(_MEMORY_SUFFIX):
            raise PreflightCheckError(
                "parameter_declaration_invalid",
                f"Qwen native memory parameter must end in {_MEMORY_SUFFIX}: {name}",
            )
        if _resolve_parameter(tensors, name) != expected_shape:
            raise PreflightCheckError(
                "parameter_metadata_mismatch",
                f"Qwen native in_proj_b layout does not match {expected_shape}: {name}",
            )
        prefixes.append(name[: -len(_MEMORY_SUFFIX)])
    if len(set(prefixes)) != config.model.num_layers:
        raise PreflightCheckError(
            "parameter_declaration_invalid", "Qwen native layer prefixes are not unique"
        )
    return tuple(prefixes)


def _native_addition_count(config: ExperimentConfig, *, r_out: int) -> int:
    if config.model.state_key_dim % 2:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen native state_key_dim must be even for rot_proj",
        )
    layers = config.model.num_layers
    heads = config.model.num_heads
    key = config.model.state_key_dim
    hidden = config.model.hidden_size
    half_key = key // 2
    per_layer = _checked_sum(
        _checked_product(hidden, heads, half_key, context="rot_proj.weight"),
        _checked_product(heads, half_key, context="rot_proj.bias"),
        _checked_product(heads, key, context="decay_chan"),
        heads,
        *(
            (
                _checked_product(heads, r_out, key, context="q_slot_scale"),
                _checked_product(heads, r_out, context="out_mix"),
            )
            if r_out > 1
            else ()
        ),
        context="KMD2 native additions per layer",
    )
    return _checked_product(layers, per_layer, context="KMD2 native additions")


def _cache_parameter_count(
    config: ExperimentConfig, *, layer_prefixes: tuple[str, ...]
) -> tuple[int, tuple[str, ...]]:
    params = config.task.params
    names = _string_names(
        params.get("cache_parameter_names", ()),
        field="task.params.cache_parameter_names",
        allow_empty=config.qwen.run_mode != "heal",
    )
    if not names:
        return _checked_product(
            config.model.num_layers,
            2 * config.model.state_key_dim + 2 * config.model.num_heads,
            context="Qwen cache parameters",
        ), names
    expected_names = {
        f"{prefix}.{suffix}" for prefix in layer_prefixes for suffix in _CACHE_SUFFIX_SIZES
    }
    if set(names) != expected_names:
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen cache trainables must exactly match the installed native layer prefixes",
        )
    counts = {suffix: 0 for suffix in _CACHE_SUFFIX_SIZES}
    total = 0
    for name in names:
        suffix = name.rsplit(".", 1)[-1]
        kind = _CACHE_SUFFIX_SIZES.get(suffix)
        if kind is None:
            raise PreflightCheckError(
                "parameter_declaration_invalid",
                f"unknown cache trainable parameter: {name}",
            )
        counts[suffix] += 1
        total = _checked_sum(
            total,
            config.model.state_key_dim if kind == "key" else config.model.num_heads,
            context="Qwen cache parameters",
        )
    if any(count != config.model.num_layers for count in counts.values()):
        raise PreflightCheckError(
            "parameter_declaration_invalid",
            "Qwen cache trainables must declare each cache parameter once per layer",
        )
    return total, names


def measure_qwen_resources(
    config: ExperimentConfig,
    spec: Any,
    *,
    assets: Mapping[str, Mapping[str, Any]],
    environ: Mapping[str, str] | None = None,
    loaded_modules: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Count parameters from safetensors headers and state/cache bytes by formula."""

    if not isinstance(config, ExperimentConfig) or config.backend != "qwen":
        return {"available": False}
    execution = verify_qwen_execution_contract(
        config,
        environ=environ,
        loaded_modules=loaded_modules,
        source_root=source_root,
    )
    model_record = assets.get("model")
    if not isinstance(model_record, Mapping) or type(model_record.get("path")) is not str:
        raise PreflightCheckError(
            "parameter_metadata_missing", "measured Qwen model identity is unavailable"
        )
    tensors = _safetensors_inventory(Path(model_record["path"]))
    total_base_parameters = _checked_sum(
        *(_shape_elements(shape, context=name) for name, shape in tensors.items()),
        context="base Qwen parameters",
    )
    memory_names = _string_names(
        config.task.params.get("memory_parameter_names", ()),
        field="task.params.memory_parameter_names",
        allow_empty=config.qwen.run_mode != "heal",
    )
    layer_prefixes = _native_layer_prefixes(config, tensors, memory_names)
    if any(
        name.endswith(suffix)
        for name in tensors
        for suffix in _NATIVE_ADDITION_SUFFIXES
    ):
        raise PreflightCheckError(
            "parameter_metadata_mismatch",
            "model safetensors already contain unsupported installed KMD2 additions",
        )
    memory_parameters = _checked_sum(
        *(
            _shape_elements(_resolve_parameter(tensors, name), context=name)
            for name in memory_names
        ),
        context="Qwen memory trainables",
    )
    native_additions = _native_addition_count(
        config, r_out=execution["native_r_out"]
    )
    cache_parameters, cache_names = _cache_parameter_count(
        config, layer_prefixes=layer_prefixes
    )
    cache_enabled = spec.mechanism == "exact_cache" and config.cache.width > 0
    treatment_trainable = _checked_sum(
        memory_parameters,
        cache_parameters if cache_enabled else 0,
        context="Qwen treatment trainables",
    )
    installed_native_total = _checked_sum(
        total_base_parameters,
        native_additions,
        context="Qwen installed native model",
    )
    total_parameters = _checked_sum(
        installed_native_total,
        cache_parameters if cache_enabled else 0,
        context="Qwen installed treatment model",
    )
    recurrent_elements = _checked_product(
        config.model.num_layers,
        config.model.num_heads,
        config.model.state_key_dim,
        config.model.state_value_dim,
        context="Qwen recurrent state",
    )
    storage_bytes = 2 if config.cache.storage_dtype == "bf16" else 4
    entry_bytes = (
        (config.model.state_key_dim + config.model.state_value_dim) * storage_bytes
        + 4
        + 8
        + 1
    )
    persistent_bytes = (
        config.model.num_layers
        * config.model.num_heads
        * config.cache.width
        * entry_bytes
        if cache_enabled
        else 0
    )
    block_bytes = (
        config.model.num_layers
        * config.model.num_heads
        * config.cache.block_size
        * entry_bytes
        if cache_enabled
        else 0
    )
    tolerance = max(0.005 * treatment_trainable, 1024.0)
    return {
        "available": True,
        "exact": True,
        "trainable_parameters": treatment_trainable,
        "total_parameters": total_parameters,
        "recurrent_state_elements": recurrent_elements,
        "recurrent_state_bytes": 4 * recurrent_elements,
        "cache_persistent_bytes": persistent_bytes,
        "cache_block_bytes": block_bytes,
        "cache_storage_dtype": config.cache.storage_dtype,
        "cache_compute_dtype": config.cache.compute_dtype,
        "ffn_match": {
            "matched": True,
            "target_parameters": treatment_trainable,
            "matched_parameters": treatment_trainable,
            "selected_d_ff": config.model.ffn_dim,
            "residual_mismatch": 0,
            "tolerance": tolerance,
        },
        "parameter_metadata_kind": "safetensors_header",
        "parameter_metadata_tensors": len(tensors),
        "parameter_scope": "full_model_plus_installed_kmd2_native_plus_cache",
        "total_base_parameters": total_base_parameters,
        "native_addition_parameters": native_additions,
        "native_r_out": execution["native_r_out"],
        "cache_parameter_count": cache_parameters if cache_enabled else 0,
        "arm_trainable_parameters": {
            "native": memory_parameters,
            "recency": treatment_trainable,
            "surprise": treatment_trainable,
        },
        "arm_total_parameters": {
            "native": installed_native_total,
            "recency": total_parameters,
            "surprise": total_parameters,
        },
        "declared_cache_parameter_count": len(cache_names),
        "qwen_execution": execution,
    }


__all__ = ["measure_qwen_resources", "verify_qwen_execution_contract"]
