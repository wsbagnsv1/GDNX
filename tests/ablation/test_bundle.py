from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import unicodedata
import zipfile

import pytest


def test_deterministic_zip_is_sorted_fixed_and_byte_identical(tmp_path) -> None:
    from research.kmd2_ablation.bundle import (
        ZIP_EPOCH,
        BundleEntry,
        write_deterministic_zip,
    )

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    entries = (
        BundleEntry("z-last.txt", b"last\n", 0o644),
        BundleEntry("pkg/caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", "value = 1\n".encode(), 0o644),
        BundleEntry("bin/verify.py", b"#!/usr/bin/env python3\n", 0o755),
    )
    first_sha = write_deterministic_zip(entries, first)
    second_sha = write_deterministic_zip(tuple(reversed(entries)), second)

    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == hashlib.sha256(first.read_bytes()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(entry.name for entry in entries)
        for info in archive.infolist():
            expected = next(entry for entry in entries if entry.name == info.filename)
            assert info.date_time == ZIP_EPOCH
            assert info.create_system == 3
            assert info.compress_type == zipfile.ZIP_DEFLATED
            mode = info.external_attr >> 16
            assert stat.S_ISREG(mode)
            assert stat.S_IMODE(mode) == expected.mode
            if not info.filename.isascii():
                assert info.flag_bits & 0x800
        assert archive.comment == b""


@pytest.mark.parametrize(
    "name",
    [
        "/absolute.txt",
        "C:/drive.txt",
        "C:\\drive.txt",
        "../escape.txt",
        "dir/../escape.txt",
        "dir\\file.txt",
        "dir//file.txt",
        "./file.txt",
        "dir/./file.txt",
        "README.md:private-stream",
        "nested/name:stream.py",
        "NUL",
        "aux.txt",
        "nested/COM1.py",
        "nested/COM¹.py",
        "nested/LPT9.txt",
        "trailing-dot.",
        "nested/trailing-space ",
        "nested/question?.txt",
        "nested/pipe|name.py",
        "nested/angle<name>.txt",
        "",
    ],
)
def test_deterministic_zip_rejects_unsafe_member_names(tmp_path, name: str) -> None:
    from research.kmd2_ablation.bundle import BundleEntry, BundleError, write_deterministic_zip

    with pytest.raises(BundleError, match="unsafe.*member"):
        write_deterministic_zip((BundleEntry(name, b"x"),), tmp_path / "unsafe.zip")


@pytest.mark.parametrize(
    "names",
    [
        ("same.txt", "same.txt"),
        ("README.md", "readme.md"),
        (
            "docs/caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt",
            "docs/" + unicodedata.normalize("NFD", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt"),
        ),
        ("docs/stra\N{LATIN SMALL LETTER SHARP S}e.txt", "docs/strasse.txt"),
    ],
)
def test_deterministic_zip_rejects_duplicate_unicode_and_casefold_collisions(
    tmp_path, names: tuple[str, str]
) -> None:
    from research.kmd2_ablation.bundle import BundleEntry, BundleError, write_deterministic_zip

    entries = tuple(BundleEntry(name, str(index).encode()) for index, name in enumerate(names))
    with pytest.raises(BundleError, match="collision"):
        write_deterministic_zip(entries, tmp_path / "collision.zip")


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, 0o600, 0o777])
def test_deterministic_zip_rejects_noncanonical_permission_modes(
    tmp_path, mode: int
) -> None:
    from research.kmd2_ablation.bundle import BundleEntry, BundleError, write_deterministic_zip

    with pytest.raises(BundleError, match="mode"):
        write_deterministic_zip(
            (BundleEntry("link", b"target", mode),),
            tmp_path / "mode.zip",
        )


def _write_fixture_file(root: Path, name: str, content: str | bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _create_bundle_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "fixture-repo"
    root.mkdir()
    checkout = Path(__file__).resolve().parents[2]
    production_paths = [
        *(checkout / "research/kmd2_ablation").rglob("*.py"),
        *(checkout / "gdn3").glob("*.py"),
        checkout / "tests/ablation/test_tiny_backend.py",
        checkout / "tests/ablation/test_fast_scan_api.py",
        checkout / "tests/ablation/test_qwen_backend.py",
        checkout / "research/kmd2_ablation/README.md",
        checkout / "research/kmd2_ablation/config.schema.json",
        checkout / "research/kmd2_ablation/scripts/run_remote_tiny.sh",
        checkout / "research/kmd2_ablation/scripts/run_remote_qwen.sh",
    ]
    for source in production_paths:
        _write_fixture_file(
            root,
            source.relative_to(checkout).as_posix(),
            source.read_bytes(),
        )
    for name in ("LICENSE", "README.md"):
        _write_fixture_file(root, name, (checkout / name).read_bytes())

    from tests.ablation.test_config import minimal_config_dict

    tiny_config = minimal_config_dict()
    tiny_config.update(
        {
            "backend": "tiny",
            "mechanism": "native",
            "variant": "native",
            "required_stage": "local_correctness",
            "task": {"name": "parity", "params": {}},
            "seeds": [211],
            "budget": {"tokens": 24, "updates": 1},
            "model": {
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 1,
                "state_key_dim": 2,
                "state_value_dim": 2,
                "ffn_dim": 16,
                "ffn_match_lower": 8,
                "ffn_match_upper": 24,
            },
            "lengths": {"curriculum": [2], "extrapolation": [1]},
            "device_preferences": ["cpu"],
            "dtype_preferences": ["float32"],
        }
    )
    tiny_config["schedule"]["warmup_updates"] = 0
    tiny_config["cache"].update(
        {"width": 2, "block_size": 2, "storage_dtype": "fp32"}
    )
    qwen_config = json.loads(json.dumps(tiny_config))
    qwen_config["backend"] = "qwen"
    qwen_memory_name = "model.layers.0.linear_attn.in_proj_b.weight"
    qwen_config["task"]["params"].update(
        {
            "native_r_out": 4,
            "score_scan": "gdn3.kmd2_fast_scan.scan_with_update_norm",
            "memory_parameter_names": [qwen_memory_name],
        }
    )

    task15_files: dict[str, str | bytes] = {
        "tests/__init__.py": "\n",
        "tests/ablation/__init__.py": "\n",
        "research/kmd2_ablation/requirements-tiny.txt": "torch==2.7.1\n",
        "research/kmd2_ablation/requirements-qwen.txt": (
            "torch>=2.4,<3\n"
            "transformers>=5.12.1,<6\n"
            "safetensors>=0.5,<1\n"
            'triton>=3.0,<4; platform_system == "Linux"\n'
        ),
        "research/kmd2_ablation/configs/tiny/a.json": json.dumps(
            tiny_config, sort_keys=True
        )
        + "\n",
        "research/kmd2_ablation/configs/tiny/b.json": json.dumps(
            tiny_config, sort_keys=True
        )
        + "\n",
        "research/kmd2_ablation/configs/qwen/a.json": json.dumps(
            qwen_config, sort_keys=True
        )
        + "\n",
        ".env": "SECRET=do-not-bundle\n",
        "secret.key": "private-key-material\n",
        ".worktrees/noise.txt": "noise\n",
        "research/kmd2_ablation/__pycache__/bad.pyc": b"pyc",
        "models/model.bin": b"model-secret",
        "data/train.bin": b"data-secret",
        "checkpoints/native.pt": b"checkpoint-secret",
        "runs/run.json": "{}\n",
        "outputs/old.zip": b"old-archive",
        "artifacts/large.bin": b"x" * 1024 * 1024,
    }
    for name, content in task15_files.items():
        _write_fixture_file(root, name, content)

    external_root = tmp_path / "external"
    external_root.mkdir()
    asset_kinds = {
        "model": "directory",
        "tokenizer": "directory",
        "checkpoint": "file",
        "data": "directory",
        "teacher_model": "file",
    }
    external_identity: dict[str, dict[str, object]] = {}
    for index, (name, kind) in enumerate(asset_kinds.items()):
        path = external_root / name
        payload = (f"fixture-{name}-metadata-only\n".encode() * 8)[: 120 + index]
        if kind == "directory":
            path.mkdir()
            member = path / "identity.txt"
            member.write_bytes(payload)
            members = [member]
            if name == "model":
                header = json.dumps(
                    {
                        qwen_memory_name: {
                            "dtype": "F32",
                            "shape": [1, 8],
                            "data_offsets": [0, 32],
                        }
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                metadata_only = path / "model.safetensors"
                metadata_only.write_bytes(
                    len(header).to_bytes(8, "little") + header + bytes(32)
                )
                members.append(metadata_only)
            tree_records = [
                [
                    member.relative_to(path).as_posix(),
                    member.stat().st_size,
                    hashlib.sha256(member.read_bytes()).hexdigest(),
                ]
                for member in sorted(members)
            ]
            tree_bytes = json.dumps(
                tree_records,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = {"tree_sha256": hashlib.sha256(tree_bytes).hexdigest()}
            size_bytes = sum(record[1] for record in tree_records)
        else:
            path.write_bytes(payload)
            digest = {"sha256": hashlib.sha256(payload).hexdigest()}
            size_bytes = len(payload)
        external_identity[name] = {
            "argument": "--" + name.replace("_", "-"),
            "path": str(path.resolve()),
            "kind": kind,
            "identity": f"fixture-{name}",
            "size_bytes": size_bytes,
            **digest,
        }

    asset_source = {
        "assets": {
            name: external_identity[name]
            for name in ("model", "tokenizer", "checkpoint", "data", "teacher_model")
        }
    }
    assets_path = tmp_path / "asset-expectations.json"
    assets_path.write_text(
        json.dumps(asset_source, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Bundle Fixture")
    _git(root, "add", "--all")
    _git(
        root,
        "update-index",
        "--chmod=+x",
        "research/kmd2_ablation/scripts/run_remote_tiny.sh",
        "research/kmd2_ablation/scripts/run_remote_qwen.sh",
    )
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "fixture"],
        cwd=root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return (
        root,
        root / "research/kmd2_ablation/configs/tiny/a.json",
        assets_path,
    )


def _entry_map(plan) -> dict[str, object]:
    return {entry.name: entry for entry in plan.entries}


def test_tiny_plan_has_canonical_manifest_complete_configs_and_no_qwen_leakage(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    entries = _entry_map(plan)
    required = {
        "LICENSE",
        "README.md",
        "research/kmd2_ablation/README.md",
        "research/kmd2_ablation/config.schema.json",
        "research/kmd2_ablation/requirements-tiny.txt",
        "research/kmd2_ablation/scripts/run_remote_tiny.sh",
        "research/kmd2_ablation/configs/tiny/a.json",
        "research/kmd2_ablation/configs/tiny/b.json",
        "research/kmd2_ablation/tiny_backend.py",
        "research/kmd2_ablation/tiny_training.py",
        "tests/ablation/test_tiny_backend.py",
        "tests/ablation/test_fast_scan_api.py",
        "verify_bundle.py",
        "MANIFEST.json",
    }
    assert required <= set(entries)
    assert entries["research/kmd2_ablation/scripts/run_remote_tiny.sh"].mode == 0o755
    assert "research/__init__.py" not in entries
    assert not any("qwen" in name.lower() for name in entries)
    assert "research/kmd2_ablation/gate_probes.py" in entries
    assert "research/kmd2_ablation/resource_probes.py" not in entries
    assert "research/kmd2_ablation/scripts/run_remote_qwen.sh" not in entries
    assert {name for name in entries if name.startswith("gdn3/")} == {
        "gdn3/_reference_recurrence.py",
        "gdn3/gdn3_upgrade.py",
        "gdn3/kmd2_fast_scan.py",
        "gdn3/kmd2_native.py",
    }
    assert "research/kmd2_ablation/requirements-qwen.txt" not in entries
    forbidden_fragments = {
        ".git",
        ".worktrees",
        "__pycache__",
        ".pyc",
        ".env",
        ".key",
        "models/",
        "data/",
        "checkpoints/",
        "runs/",
        "outputs/",
        "artifacts/",
    }
    assert not any(fragment in name for name in entries for fragment in forbidden_fragments)

    manifest_bytes = entries["MANIFEST.json"].data
    manifest = json.loads(manifest_bytes)
    assert plan.manifest == manifest
    assert set(manifest) == {
        "schema_version",
        "suite_version",
        "kind",
        "git",
        "config",
        "config_sha256",
        "production_source_sha256",
        "entries",
        "expected_members",
        "provenance",
        "smoke",
        "manifest_convention",
    }
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["suite_version"] == "1.0.0"
    assert manifest["kind"] == "tiny"
    assert len(manifest["git"]["revision"]) == 40
    assert manifest["git"]["dirty"] is False
    assert len(manifest["git"]["diff_sha256"]) == 64
    assert set(manifest["git"]) == {"revision", "dirty", "diff_sha256"}
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert len(manifest["production_source_sha256"]) == 64
    assert str(root.resolve()) not in manifest_bytes.decode("utf-8")
    assert manifest["expected_members"] == sorted(entries)
    assert "MANIFEST.json" not in manifest["entries"]
    assert set(manifest["entries"]) == set(entries) - {"MANIFEST.json"}
    for name, metadata in manifest["entries"].items():
        entry = entries[name]
        assert metadata == {
            "mode": entry.mode,
            "sha256": hashlib.sha256(entry.data).hexdigest(),
            "size": len(entry.data),
        }
    assert manifest["provenance"]["smoke_command"][-4:] == [
        "--job-index",
        "0",
        "--num-jobs",
        "1",
    ]
    assert set(manifest["provenance"]) == {"build_command", "smoke_command"}
    assert manifest["smoke"] == {
        "command": manifest["provenance"]["smoke_command"]
    }


def test_qwen_plan_adds_exact_modules_requirements_and_sanitized_asset_manifest(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, _, assets = _create_bundle_fixture(tmp_path)
    config = root / "research/kmd2_ablation/configs/qwen/a.json"
    plan = plan_bundle(
        kind="qwen",
        repo_root=root,
        config_path=config,
        assets_manifest=assets,
    )
    entries = _entry_map(plan)
    assert {
        "research/kmd2_ablation/requirements-qwen.txt",
        "research/kmd2_ablation/scripts/run_remote_qwen.sh",
        "research/kmd2_ablation/configs/qwen/a.json",
        "research/kmd2_ablation/qwen_backend.py",
        "research/kmd2_ablation/qwen_training.py",
        "research/kmd2_ablation/gate_probes.py",
        "research/kmd2_ablation/resource_probes.py",
        "tests/ablation/test_qwen_backend.py",
        "gdn3/_reference_recurrence.py",
        "gdn3/gdn3_upgrade.py",
        "gdn3/kmd2_fast_scan.py",
        "gdn3/kmd2_native.py",
        "external-assets.json",
    } <= set(entries)
    assert "research/kmd2_ablation/scripts/run_remote_tiny.sh" not in entries
    assert entries["research/kmd2_ablation/scripts/run_remote_qwen.sh"].mode == 0o755
    embedded = json.loads(entries["external-assets.json"].data)
    assert set(embedded["assets"]) == {
        "model",
        "tokenizer",
        "checkpoint",
        "data",
        "teacher_model",
    }
    serialized = entries["external-assets.json"].data.decode("utf-8")
    assert str(tmp_path.resolve()) not in serialized
    assert "path" not in serialized
    for name, metadata in embedded["assets"].items():
        assert metadata["argument"] == "--" + name.replace("_", "-")
        assert metadata["expected_kind"] in {"file", "directory"}
        assert metadata["expected_identity"] == f"fixture-{name}"
        assert metadata["expected_size_bytes"] >= 100
        assert "sha256" in metadata or "tree_sha256" in metadata


def test_external_assets_reject_paths_traversal_and_secret_bearing_metadata(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import BundleError, _sanitized_external_assets

    _, _, assets = _create_bundle_fixture(tmp_path)
    original = json.loads(assets.read_text(encoding="utf-8"))
    mutations = {
        "windows-absolute": lambda value: value["assets"]["model"].__setitem__(
            "identity", "C:/Users/jackj/private-model"
        ),
        "embedded-windows-absolute": lambda value: value["assets"]["model"].__setitem__(
            "identity", "model metadata at C:/Users/jackj/private-model"
        ),
        "posix-absolute": lambda value: value["assets"]["model"].__setitem__(
            "identity", "/home/jack/private-model"
        ),
        "embedded-posix-absolute": lambda value: value["assets"]["model"].__setitem__(
            "identity", "model metadata at /home/jack/private-model"
        ),
        "unc-absolute": lambda value: value["assets"]["model"].__setitem__(
            "identity", r"\\server\share\private-model"
        ),
        "traversal": lambda value: value["assets"]["model"].__setitem__(
            "identity", "../private-model"
        ),
        "secret-field": lambda value: value["assets"]["model"].__setitem__(
            "metadata", {"api_token": "sk-do-not-serialize"}
        ),
        "nested-secret-field": lambda value: value["assets"]["model"].__setitem__(
            "metadata", {"secret_path": "redacted-but-forbidden"}
        ),
        "secret-value": lambda value: value["assets"]["model"].__setitem__(
            "identity", "-----BEGIN PRIVATE KEY-----"
        ),
    }
    for label, mutate in mutations.items():
        document = json.loads(json.dumps(original))
        mutate(document)
        candidate = tmp_path / f"assets-{label}.json"
        candidate.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(BundleError) as caught:
            _sanitized_external_assets(candidate)
        assert caught.value.code == "assets_manifest_invalid", label


def test_collection_is_independent_of_source_mtimes_and_returns_identical_archives(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import build_bundle, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    first_plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    first = build_bundle(first_plan, tmp_path / "first.zip")
    for index, path in enumerate(sorted(root.rglob("*"))):
        if path.is_file() and ".git" not in path.parts:
            timestamp = 946684800 + index
            os.utime(path, (timestamp, timestamp))
    second_plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    second = build_bundle(second_plan, tmp_path / "second.zip")
    assert first.sha256 == second.sha256
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    assert first.member_count == second.member_count == len(first_plan.entries)


def test_flat_task15_configs_are_classified_by_declared_backend(tmp_path) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, _, _ = _create_bundle_fixture(tmp_path)
    moves = (
        ("configs/tiny/a.json", "configs/smoke.json"),
        ("configs/tiny/b.json", "configs/screening.json"),
        ("configs/qwen/a.json", "configs/qwen_exact_cache.json"),
    )
    for source, destination in moves:
        _git(
            root,
            "mv",
            f"research/kmd2_ablation/{source}",
            f"research/kmd2_ablation/{destination}",
        )
    selected = root / "research/kmd2_ablation/configs/smoke.json"

    entries = _entry_map(
        plan_bundle(kind="tiny", repo_root=root, config_path=selected)
    )
    assert "research/kmd2_ablation/configs/smoke.json" in entries
    assert "research/kmd2_ablation/configs/screening.json" in entries
    assert "research/kmd2_ablation/configs/qwen_exact_cache.json" not in entries


def test_valid_untracked_configs_in_explicit_config_root_are_bundled(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, tracked, _ = _create_bundle_fixture(tmp_path)
    untracked = root / "research/kmd2_ablation/configs/tiny/worktree-screen.json"
    untracked.write_bytes(tracked.read_bytes())

    entries = _entry_map(
        plan_bundle(kind="tiny", repo_root=root, config_path=untracked)
    )

    assert "research/kmd2_ablation/configs/tiny/a.json" in entries
    assert "research/kmd2_ablation/configs/tiny/b.json" in entries
    assert "research/kmd2_ablation/configs/tiny/worktree-screen.json" in entries


def test_bundle_accepts_pep420_top_level_tests_namespace(tmp_path) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, tracked, _ = _create_bundle_fixture(tmp_path)
    _git(root, "rm", "tests/__init__.py")

    entries = _entry_map(
        plan_bundle(kind="tiny", repo_root=root, config_path=tracked)
    )

    assert "tests/__init__.py" not in entries
    assert "tests/ablation/__init__.py" in entries
    assert "tests/ablation/test_tiny_backend.py" in entries


def test_invalid_untracked_config_in_explicit_root_fails_closed(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, tracked, _ = _create_bundle_fixture(tmp_path)
    invalid = root / "research/kmd2_ablation/configs/tiny/untrusted.json"
    invalid.write_text('{"backend":"tiny"}\n', encoding="utf-8")

    with pytest.raises(BundleError) as caught:
        plan_bundle(kind="tiny", repo_root=root, config_path=tracked)

    assert caught.value.code == "config_invalid"


def test_manifest_records_dirty_git_state_and_normalized_diff_digest(tmp_path) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    clean = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    config_module = root / "research/kmd2_ablation/config.py"
    config_module.write_text(
        config_module.read_text(encoding="utf-8") + "# dirty\r\n",
        encoding="utf-8",
    )
    dirty = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    assert clean.manifest["git"]["dirty"] is False
    assert dirty.manifest["git"]["dirty"] is True
    assert dirty.manifest["git"]["revision"] == clean.manifest["git"]["revision"]
    assert dirty.manifest["git"]["diff_sha256"] != clean.manifest["git"]["diff_sha256"]


def test_collection_uses_explicit_roots_and_recursive_local_closure(tmp_path) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    source_root = root / "research/kmd2_ablation"
    _write_fixture_file(
        root,
        "research/kmd2_ablation/reachable_one.py",
        "from .reachable_two import value\n",
    )
    _write_fixture_file(
        root,
        "research/kmd2_ablation/reachable_two.py",
        "value = 7\n",
    )
    backend = source_root / "tiny_backend.py"
    backend.write_text(
        backend.read_text(encoding="utf-8")
        + "\nfrom .reachable_one import value as reachable_value\n",
        encoding="utf-8",
    )
    _write_fixture_file(
        root,
        "research/kmd2_ablation/arbitrary_untracked_suite.py",
        "raise RuntimeError('must never be bundled')\n",
    )
    _write_fixture_file(
        root,
        "tests/ablation/test_arbitrary_untracked_suite.py",
        "raise RuntimeError('must never be bundled')\n",
    )
    forbidden_segments = (
        "Secrets",
        "CACHE",
        ".WoRkTrEeS",
        "MODELS",
        "Data",
        "CHECKPOINTS",
        "Runs",
    )
    for segment in forbidden_segments:
        _write_fixture_file(
            root,
            f"research/kmd2_ablation/{segment}/leak.py",
            "SECRET = 'must never be bundled'\n",
        )
        _write_fixture_file(
            root,
            f"{segment}/root_leak.py",
            "SECRET = 'must never be bundled'\n",
        )

    plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    names = {entry.name for entry in plan.entries}
    assert "research/kmd2_ablation/reachable_one.py" in names
    assert "research/kmd2_ablation/reachable_two.py" in names
    assert "research/kmd2_ablation/arbitrary_untracked_suite.py" not in names
    assert "tests/ablation/test_arbitrary_untracked_suite.py" not in names
    forbidden = {segment.casefold() for segment in forbidden_segments}
    assert not any(
        part.casefold() in forbidden
        for name in names
        for part in PurePosixPath(name).parts
    )


def test_import_from_alias_must_resolve_to_module_or_exported_symbol(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    backend = root / "research/kmd2_ablation/tiny_backend.py"
    backend.write_text(
        backend.read_text(encoding="utf-8")
        + "\nfrom . import SUITE_VERSION\n"
        + "from . import definitely_missing\n",
        encoding="utf-8",
    )
    with pytest.raises(BundleError) as caught:
        plan_bundle(kind="tiny", repo_root=root, config_path=config)
    assert caught.value.code == "local_import_missing"
    assert "definitely_missing" in str(caught.value)


def test_local_closure_recurses_through_star_and_parent_relative_imports(tmp_path) -> None:
    from research.kmd2_ablation.bundle import plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    _write_fixture_file(
        root,
        "research/kmd2_ablation/helpers/__init__.py",
        "from .deep import *\n",
    )
    _write_fixture_file(
        root,
        "research/kmd2_ablation/helpers/deep.py",
        "from ..config import SUITE_VERSION\nVALUE = SUITE_VERSION\n",
    )
    backend = root / "research/kmd2_ablation/tiny_backend.py"
    backend.write_text(
        backend.read_text(encoding="utf-8") + "\nfrom .helpers import *\n",
        encoding="utf-8",
    )

    names = {
        entry.name
        for entry in plan_bundle(
            kind="tiny", repo_root=root, config_path=config
        ).entries
    }
    assert "research/kmd2_ablation/helpers/__init__.py" in names
    assert "research/kmd2_ablation/helpers/deep.py" in names


def test_star_import_requires_every_explicit_all_export_to_resolve(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    _write_fixture_file(
        root,
        "research/kmd2_ablation/helpers/__init__.py",
        '__all__ = ("definitely_missing",)\n',
    )
    backend = root / "research/kmd2_ablation/tiny_backend.py"
    backend.write_text(
        backend.read_text(encoding="utf-8") + "\nfrom .helpers import *\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleError) as caught:
        plan_bundle(kind="tiny", repo_root=root, config_path=config)
    assert caught.value.code == "local_import_missing"
    assert "definitely_missing" in str(caught.value)


def test_tiny_runtime_cannot_pull_qwen_modules_through_local_imports(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    backend = root / "research/kmd2_ablation/tiny_backend.py"
    backend.write_text(
        backend.read_text(encoding="utf-8")
        + "\nfrom .qwen_backend import ExternalAssetIdentity\n",
        encoding="utf-8",
    )
    with pytest.raises(BundleError) as caught:
        plan_bundle(kind="tiny", repo_root=root, config_path=config)
    assert caught.value.code == "forbidden_dependency"


@pytest.mark.parametrize(
    "line",
    [
        "torch==2.7.1 --index-url https://example.invalid/simple",
        "torch>=2.7.0 trailing-junk",
        "torch @ https://example.invalid/torch.whl",
        "torch==2.7.1 || echo injected",
    ],
)
def test_requirement_lines_are_fully_parsed_and_reject_suffix_junk(line: str) -> None:
    from research.kmd2_ablation.bundle import BundleError, _requirement_roots

    with pytest.raises(BundleError) as caught:
        _requirement_roots((line + "\n").encode(), kind="tiny")
    assert caught.value.code == "requirements_invalid"


def test_tiny_dependency_policy_allows_only_torch_even_when_declared() -> None:
    from research.kmd2_ablation.bundle import BundleError, _requirement_roots

    with pytest.raises(BundleError) as caught:
        _requirement_roots(b"torch==2.7.1\nunknown-vendor==1.0\n", kind="tiny")
    assert caught.value.code == "forbidden_dependency"


def test_qwen_requirement_parser_accepts_complete_simple_environment_marker() -> None:
    from research.kmd2_ablation.bundle import _requirement_roots

    roots = _requirement_roots(
        (
            "torch>=2.4,<3\n"
            "transformers>=5.12.1,<6\n"
            "safetensors>=0.5,<1\n"
            'triton>=3.0,<4; platform_system == "Linux"\n'
        ).encode(),
        kind="qwen",
    )
    assert roots == frozenset({"torch", "transformers", "safetensors", "triton"})


@pytest.mark.parametrize(
    ("kind", "target", "addition", "expected_code"),
    [
        (
            "tiny",
            "research/kmd2_ablation/tiny_backend.py",
            "\nimport transformers\n",
            "forbidden_dependency",
        ),
        (
            "tiny",
            "research/kmd2_ablation/requirements-tiny.txt",
            "triton==3.3.0\n",
            "forbidden_dependency",
        ),
        (
            "qwen",
            "research/kmd2_ablation/qwen_backend.py",
            "\nimport unknown_vendor_package\n",
            "undeclared_dependency",
        ),
    ],
)
def test_ast_requirements_audit_fails_closed(
    tmp_path, kind: str, target: str, addition: str, expected_code: str
) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, tiny_config, assets = _create_bundle_fixture(tmp_path)
    path = root / target
    path.write_text(path.read_text(encoding="utf-8") + addition, encoding="utf-8")
    config = (
        tiny_config
        if kind == "tiny"
        else root / "research/kmd2_ablation/configs/qwen/a.json"
    )
    with pytest.raises(BundleError) as caught:
        plan_bundle(
            kind=kind,
            repo_root=root,
            config_path=config,
            assets_manifest=assets if kind == "qwen" else None,
        )
    assert caught.value.code == expected_code


def test_missing_task15_artifacts_fail_actionably(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    requirements = root / "research/kmd2_ablation/requirements-tiny.txt"
    requirements.unlink()
    with pytest.raises(BundleError) as caught:
        plan_bundle(kind="tiny", repo_root=root, config_path=config)
    assert caught.value.code == "required_artifact_missing"
    assert "requirements-tiny.txt" in str(caught.value)


def _build_fixture_archive(
    tmp_path: Path, *, kind: str = "tiny"
) -> tuple[Path, Path, Path]:
    from research.kmd2_ablation.bundle import build_bundle, plan_bundle

    root, tiny_config, assets = _create_bundle_fixture(tmp_path)
    config = (
        tiny_config
        if kind == "tiny"
        else root / "research/kmd2_ablation/configs/qwen/a.json"
    )
    archive = tmp_path / f"fixture-{kind}.zip"
    plan = plan_bundle(
        kind=kind,
        repo_root=root,
        config_path=config,
        assets_manifest=assets if kind == "qwen" else None,
    )
    build_bundle(plan, archive)
    return archive, root, config


def _tamper_payload(source: Path, destination: Path, target: str) -> None:
    from research.kmd2_ablation.bundle import BundleEntry, write_deterministic_zip

    entries = []
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            data = archive.read(info)
            if info.filename == target:
                data += b"# tampered\n"
            entries.append(
                BundleEntry(
                    info.filename,
                    data,
                    stat.S_IMODE(info.external_attr >> 16),
                )
            )
    write_deterministic_zip(entries, destination)


def _rewrite_manifest_document(
    source: Path,
    destination: Path,
    transform,
) -> None:
    from research.kmd2_ablation.bundle import BundleEntry, write_deterministic_zip

    entries = []
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            data = archive.read(info)
            if info.filename == "MANIFEST.json":
                document = json.loads(data)
                transform(document)
                data = (
                    json.dumps(
                        document,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            entries.append(
                BundleEntry(
                    info.filename,
                    data,
                    stat.S_IMODE(info.external_attr >> 16),
                )
            )
    write_deterministic_zip(entries, destination)


def test_verifier_requires_exact_manifest_schema_and_nested_provenance(tmp_path) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    source, _, _ = _build_fixture_archive(tmp_path)
    mutations = {
        "missing-suite": lambda value: value.pop("suite_version"),
        "suite-config-mismatch": lambda value: value.__setitem__(
            "suite_version", "9.9.9"
        ),
        "unknown-top-level": lambda value: value.__setitem__("unexpected", True),
        "missing-git-digest": lambda value: value["git"].pop("diff_sha256"),
        "unknown-git": lambda value: value["git"].__setitem__("branch", "main"),
        "bad-revision": lambda value: value["git"].__setitem__("revision", "HEAD"),
        "bad-provenance": lambda value: value["provenance"].__setitem__(
            "build_command", ["python", "bundle"]
        ),
        "missing-smoke": lambda value: value.pop("smoke"),
    }
    for label, transform in mutations.items():
        damaged = tmp_path / f"manifest-{label}.zip"
        _rewrite_manifest_document(source, damaged, transform)
        result = verify_bundle(damaged)
        assert result.ok is False, label
        assert "manifest_schema_invalid" in result.codes, (label, result.codes)


def _rewrite_member_metadata(
    source: Path,
    destination: Path,
    target: str,
    *,
    compression: int | None = None,
    mode: int | None = None,
) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as rewritten:
        for old in original.infolist():
            info = zipfile.ZipInfo(old.filename, date_time=old.date_time)
            info.create_system = 3
            info.compress_type = (
                compression
                if old.filename == target and compression is not None
                else old.compress_type
            )
            selected_mode = (
                mode
                if old.filename == target and mode is not None
                else old.external_attr >> 16
            )
            info.external_attr = selected_mode << 16
            rewritten.writestr(info, original.read(old), compress_type=info.compress_type)


def _copy_zip_info(old: zipfile.ZipInfo) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(old.filename, date_time=old.date_time)
    info.create_system = old.create_system
    info.compress_type = old.compress_type
    info.external_attr = old.external_attr
    info.internal_attr = old.internal_attr
    info.extra = old.extra
    info.comment = old.comment
    return info


def _rewrite_archive_order(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as rewritten:
        for old in reversed(original.infolist()):
            info = _copy_zip_info(old)
            rewritten.writestr(
                info,
                original.read(old),
                compress_type=info.compress_type,
                compresslevel=9,
            )


def _append_canonical_member(archive_path: Path, name: str, data: bytes) -> None:
    with zipfile.ZipFile(
        archive_path,
        "a",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.internal_attr = 0
        info.extra = b""
        info.comment = b""
        archive.writestr(
            info,
            data,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )


def test_verifier_rejects_windows_ads_member_without_extraction_residue(tmp_path) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    source, _, _ = _build_fixture_archive(tmp_path)
    damaged = tmp_path / "windows-ads.zip"
    shutil.copyfile(source, damaged)
    _append_canonical_member(damaged, "README.md:private-stream", b"secret\n")

    extraction = tmp_path / "must-not-exist"
    result = verify_bundle(damaged, extract_to=extraction)
    assert result.ok is False
    assert "unsafe_member_name" in result.codes
    assert not extraction.exists()
    assert not tuple(tmp_path.glob(".must-not-exist.*.tmp"))

    with zipfile.ZipFile(source) as zipped:
        verifier = tmp_path / "standalone-verify.py"
        verifier.write_bytes(zipped.read("verify_bundle.py"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(verifier), str(damaged), "--extract-to", str(extraction)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode != 0
    assert "unsafe_member_name" in json.loads(completed.stdout)["codes"]
    assert not extraction.exists()
    assert not tuple(tmp_path.glob(".must-not-exist.*.tmp"))


def test_verifier_rejects_reversed_otherwise_canonical_member_order(tmp_path) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    source, _, _ = _build_fixture_archive(tmp_path)
    reversed_archive = tmp_path / "reversed.zip"
    _rewrite_archive_order(source, reversed_archive)

    with zipfile.ZipFile(reversed_archive) as archive:
        assert archive.namelist() == list(reversed(sorted(archive.namelist())))
    result = verify_bundle(reversed_archive)
    assert result.ok is False
    assert "noncanonical_member_order" in result.codes


def test_verify_bundle_reopens_exact_members_and_extracts_only_after_success(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    archive, _, _ = _build_fixture_archive(tmp_path)
    extraction = tmp_path / "fresh-extraction"
    result = verify_bundle(archive, extract_to=extraction)
    assert result.ok is True
    assert result.codes == ()
    assert result.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result.member_count > 0
    assert result.extracted_to == extraction
    with zipfile.ZipFile(archive) as zipped:
        assert sorted(
            path.relative_to(extraction).as_posix()
            for path in extraction.rglob("*")
            if path.is_file()
        ) == sorted(zipped.namelist())


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("tamper", "hash_mismatch"),
        ("duplicate", "member_name_collision"),
        ("traversal", "unsafe_member_name"),
        ("encrypted", "encrypted_member"),
        ("stored", "unsupported_compression"),
        ("symlink", "special_mode"),
    ],
)
def test_verify_bundle_rejects_tamper_duplicates_and_unsafe_metadata(
    tmp_path, mutation: str, expected_code: str
) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    source, _, _ = _build_fixture_archive(tmp_path)
    damaged = tmp_path / f"{mutation}.zip"
    if mutation == "tamper":
        _tamper_payload(source, damaged, "research/kmd2_ablation/tiny_backend.py")
    elif mutation in {"duplicate", "traversal"}:
        shutil.copyfile(source, damaged)
        name = "README.md" if mutation == "duplicate" else "../escape.txt"
        with zipfile.ZipFile(damaged, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, b"unsafe")
    elif mutation == "encrypted":
        data = bytearray(source.read_bytes())
        central = data.find(b"PK\x01\x02")
        assert central >= 0
        flags = int.from_bytes(data[central + 8 : central + 10], "little") | 1
        data[central + 8 : central + 10] = flags.to_bytes(2, "little")
        damaged.write_bytes(data)
    elif mutation == "stored":
        _rewrite_member_metadata(
            source,
            damaged,
            "README.md",
            compression=zipfile.ZIP_STORED,
        )
    else:
        _rewrite_member_metadata(
            source,
            damaged,
            "README.md",
            mode=stat.S_IFLNK | 0o777,
        )

    extraction = tmp_path / "must-not-exist"
    result = verify_bundle(damaged, extract_to=extraction)
    assert result.ok is False
    assert expected_code in result.codes
    assert not extraction.exists()


def test_embedded_standard_library_verifier_is_machine_readable_and_nonzero_on_failure(
    tmp_path,
) -> None:
    archive, _, _ = _build_fixture_archive(tmp_path)
    with zipfile.ZipFile(archive) as zipped:
        verifier = tmp_path / "verify_bundle.py"
        verifier.write_bytes(zipped.read("verify_bundle.py"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    valid = subprocess.run(
        [sys.executable, str(verifier), str(archive)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["ok"] is True

    damaged = tmp_path / "damaged.zip"
    _tamper_payload(archive, damaged, "README.md")
    invalid = subprocess.run(
        [sys.executable, str(verifier), str(damaged)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert invalid.returncode != 0
    report = json.loads(invalid.stdout)
    assert report["ok"] is False
    assert "hash_mismatch" in report["codes"]


def test_build_emits_documented_adjacent_verifier_sidecar(tmp_path) -> None:
    from research.kmd2_ablation.bundle import (
        VERIFY_BUNDLE_SOURCE,
        build_bundle,
        plan_bundle,
    )

    root, config, _ = _create_bundle_fixture(tmp_path)
    plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    archive = tmp_path / "delivery/kmd2-tiny.zip"
    result = build_bundle(plan, archive)
    sidecar = archive.with_name("verify_bundle.py")

    assert result.verifier_path == sidecar
    assert result.verifier_sha256 == hashlib.sha256(VERIFY_BUNDLE_SOURCE).hexdigest()
    assert sidecar.read_bytes() == VERIFY_BUNDLE_SOURCE
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read("verify_bundle.py") == sidecar.read_bytes()

    readme = (
        Path(__file__).resolve().parents[2]
        / "research/kmd2_ablation/README.md"
    ).read_text(encoding="utf-8")
    assert "python verify_bundle.py kmd2-tiny.zip" in readme
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(sidecar), archive.name],
        cwd=archive.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def test_build_bundle_is_atomic_and_preserves_existing_output_on_verification_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import research.kmd2_ablation.bundle as bundle_module

    root, config, _ = _create_bundle_fixture(tmp_path)
    plan = bundle_module.plan_bundle(kind="tiny", repo_root=root, config_path=config)
    destination = tmp_path / "bundle.zip"
    destination.write_bytes(b"existing-output")

    def corrupt_writer(entries, path):
        Path(path).write_bytes(b"not-a-zip")
        return hashlib.sha256(b"not-a-zip").hexdigest()

    monkeypatch.setattr(bundle_module, "write_deterministic_zip", corrupt_writer)
    with pytest.raises(bundle_module.BundleError, match="verification"):
        bundle_module.build_bundle(plan, destination)
    assert destination.read_bytes() == b"existing-output"
    assert not destination.with_name("verify_bundle.py").exists()
    assert not tuple(tmp_path.glob(".bundle.zip.*.tmp"))
    assert not tuple(tmp_path.glob(".verify_bundle.py.*.tmp"))


def test_archive_destination_cannot_collide_with_verifier_sidecar(tmp_path) -> None:
    from research.kmd2_ablation.bundle import BundleError, build_bundle, plan_bundle

    root, config, _ = _create_bundle_fixture(tmp_path)
    plan = plan_bundle(kind="tiny", repo_root=root, config_path=config)
    destination = tmp_path / "verify_bundle.py"
    with pytest.raises(BundleError) as caught:
        build_bundle(plan, destination)
    assert caught.value.code == "verifier_sidecar_collision"
    assert not destination.exists()


def _run_extracted_command(
    extraction: Path,
    arguments: list[str],
    *,
    forbidden_imports: tuple[str, ...] = (),
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    command = [sys.executable, "-m", "research.kmd2_ablation.run_ablation"]
    if forbidden_imports:
        bootstrap = (
            "import runpy,sys\n"
            f"blocked={forbidden_imports!r}\n"
            "for name in blocked: sys.modules[name]=None\n"
            "sys.argv=['research.kmd2_ablation.run_ablation',*sys.argv[1:]]\n"
            "code=0\n"
            "try:\n"
            " runpy.run_module('research.kmd2_ablation.run_ablation',run_name='__main__')\n"
            "except SystemExit as signal:\n"
            " code=int(signal.code or 0)\n"
            "assert not any(name.partition('.')[0] in blocked and module is not None for name,module in sys.modules.items())\n"
            "raise SystemExit(code)\n"
        )
        command = [sys.executable, "-c", bootstrap]
    completed = subprocess.run(
        [*command, *arguments],
        cwd=extraction,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _smoke_extracted_launcher_syntax(extraction: Path, kind: str) -> None:
    relative = Path(
        f"research/kmd2_ablation/scripts/run_remote_{kind}.sh"
    )
    script = extraction / relative
    source = script.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "python -m research.kmd2_ablation.run_ablation preflight" in source
    assert "python -m research.kmd2_ablation.run_ablation run" in source
    bash = shutil.which("bash")
    if bash is None:
        return
    syntax = subprocess.run(
        [bash, "-n", relative.as_posix()],
        cwd=extraction,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [bash, relative.as_posix(), "--help"],
        cwd=extraction,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "usage:" in help_result.stderr


def test_verified_fresh_tiny_extraction_runs_when_optional_qwen_packages_are_unavailable(
    tmp_path,
) -> None:
    from research.kmd2_ablation.bundle import verify_bundle
    from research.kmd2_ablation.results import validate_completed_run

    archive, _, config = _build_fixture_archive(tmp_path)
    extraction = tmp_path / "tiny-extracted"
    assert verify_bundle(archive, extract_to=extraction).ok
    assert not (extraction / "research/__init__.py").exists()
    _smoke_extracted_launcher_syntax(extraction, "tiny")
    relative_config = config.relative_to(config.parents[4]).as_posix()
    # PyTorch may probe optional Triton when constructing an optimizer.  Seeding
    # these roots to None models a torch-only remote and proves the Tiny suite does
    # not require or successfully load either optional Qwen dependency.
    preflight = _run_extracted_command(
        extraction,
        [
            "preflight",
            "--backend",
            "tiny",
            "--config",
            relative_config,
            "--out",
            "results",
        ],
        forbidden_imports=("transformers", "triton"),
    )
    assert preflight["ok"] is True
    assert preflight["codes"] == []
    assert preflight["schema_version"] == "1.0.0"
    assert len(preflight["jobs"]) == 1
    assert preflight["jobs"][0]["backend"] == "tiny"
    run = _run_extracted_command(
        extraction,
        [
            "run",
            "--backend",
            "tiny",
            "--config",
            relative_config,
            "--out",
            "results",
            "--job-index",
            "0",
            "--num-jobs",
            "1",
        ],
        forbidden_imports=("transformers", "triton"),
    )
    assert run["ok"] is True
    assert run["codes"] == []
    assert run["outcomes"] == [
        {
            "job_id": preflight["jobs"][0]["job_id"],
            "status": "completed",
        }
    ]

    manifest = json.loads((extraction / "results/manifest.json").read_text("utf-8"))
    jobs = json.loads((extraction / "results/jobs.json").read_text("utf-8"))["jobs"]
    records = tuple((extraction / "results/runs").rglob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text("utf-8"))
    provenance = {
        field: manifest[field]
        for field in (
            "schema_version",
            "suite_version",
            "source_hashes",
            "config_hash",
            "asset_hashes",
            "git",
            "environment",
        )
    }
    validate_completed_run(record, jobs[0], provenance)
    assert record["status"] == "completed"
    assert record["performance"]["wall_time_seconds"] >= 0.0
    assert record["metrics"]


def test_verified_fresh_qwen_extraction_preflights_with_metadata_only_assets(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.bundle import verify_bundle

    archive, root, _ = _build_fixture_archive(tmp_path, kind="qwen")
    extraction = tmp_path / "qwen-extracted"
    assert verify_bundle(archive, extract_to=extraction).ok
    _smoke_extracted_launcher_syntax(extraction, "qwen")
    relative_config = "research/kmd2_ablation/configs/qwen/a.json"
    external = root.parent / "external"
    monkeypatch.setenv("GDN3_FAST_SCAN", "1")
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    report = _run_extracted_command(
        extraction,
        [
            "preflight",
            "--backend",
            "qwen",
            "--config",
            relative_config,
            "--out",
            "results",
            "--dry-run",
            "--model",
            str(external / "model"),
            "--tokenizer",
            str(external / "tokenizer"),
            "--checkpoint",
            str(external / "checkpoint"),
            "--data",
            str(external / "data"),
            "--student-device",
            "cpu",
            "--dtype",
            "float32",
        ],
    )
    assert report["ok"] is True
    assert report["codes"] == []
    assert report["commands"]["preflight"][-1] == "--dry-run"
    assert {"model", "tokenizer", "checkpoint", "data"} <= set(report["assets"])
    assert not any(path.name.endswith((".bin", ".pt")) for path in extraction.rglob("*"))
    assert str(root.resolve()) not in (extraction / "external-assets.json").read_text(
        encoding="utf-8"
    )


def test_bundle_cli_handler_and_run_ablation_report_json_without_absolute_paths(
    tmp_path,
) -> None:
    from io import StringIO

    from research.kmd2_ablation.bundle import cli_handler
    from research.kmd2_ablation.run_ablation import main

    root, config, _ = _create_bundle_fixture(tmp_path)
    destination = tmp_path / "portable.zip"
    report = cli_handler(
        SimpleNamespace(
            backend="tiny",
            config=config,
            out=destination,
            assets_manifest=None,
            repo_root=root,
        )
    )
    assert report["ok"] is True
    assert report["codes"] == []
    assert report["archive"] == "portable.zip"
    assert report["verifier"] == "verify_bundle.py"
    assert report["verifier_sha256"] == hashlib.sha256(
        destination.with_name("verify_bundle.py").read_bytes()
    ).hexdigest()
    assert report["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert str(tmp_path.resolve()) not in json.dumps(report, sort_keys=True)

    stdout = StringIO()
    exit_code = main(
        [
            "bundle",
            "--backend",
            "tiny",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "via-cli.zip"),
            "--repo-root",
            str(root),
        ],
        stdout=stdout,
    )
    assert exit_code == 0
    emitted = json.loads(stdout.getvalue())
    assert emitted["ok"] is True
    assert emitted["archive"] == "via-cli.zip"


def test_bundle_cli_failure_is_machine_readable_exit_six(tmp_path) -> None:
    from io import StringIO

    from research.kmd2_ablation.run_ablation import main

    root, config, _ = _create_bundle_fixture(tmp_path)
    (root / "research/kmd2_ablation/requirements-tiny.txt").unlink()
    stdout = StringIO()
    exit_code = main(
        [
            "bundle",
            "--backend",
            "tiny",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "failed.zip"),
            "--repo-root",
            str(root),
        ],
        stdout=stdout,
    )
    assert exit_code == 6
    report = json.loads(stdout.getvalue())
    assert report["ok"] is False
    assert report["codes"] == ["required_artifact_missing"]
    assert "requirements-tiny.txt" in report["error"]
