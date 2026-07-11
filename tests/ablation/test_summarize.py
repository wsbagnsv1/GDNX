from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import research.kmd2_ablation.summarize as summarize_module
from research.kmd2_ablation.metrics import (
    BootstrapInterval,
    MetricSample,
    NamedInterval,
    Option3Evidence,
    Option3Thresholds,
    ProtectedEffect,
)
from research.kmd2_ablation.summarize import (
    SummaryValidationError,
    build_summary_artifacts,
    classify_factorial_addition,
    classify_paired_reliance,
    decide_option3_summary,
    write_summary_artifacts,
)
from research.kmd2_ablation.tasks.ruler import (
    RULER_ARMS,
    RULER_CONTEXT_LENGTHS,
    RULER_DEPTH_STRATA,
    RULER_LONG_CELLS,
    RULER_MIN_EPISODES_PER_CELL,
    RULER_QUERY_COUNTS,
    RulerCell,
    build_ruler_episode,
    ruler_evidence_scope,
    score_free_generation,
    score_teacher_forced,
    select_free_generation_subset,
)


def test_summarize_exposes_production_cli_handler() -> None:
    assert callable(summarize_module.cli_handler)


def test_summary_accepts_backend_native_non_ruler_evaluation_rows() -> None:
    artifacts = build_summary_artifacts(
        [
            {
                "schema_version": "1.0.0",
                "job_id": "tiny-job",
                "experiment_id": "tiny-experiment",
                "arm_id": "native",
                "seed": 11,
                "status": "completed",
                "metrics": {"exact_match": 0.5},
                "evaluations": [
                    {
                        "split": "id",
                        "seed": 12,
                        "examples": 1,
                        "tokens": 14,
                        "loss": 1.25,
                    }
                ],
            }
        ]
    )

    assert b'"execution_status":"completed"' in artifacts.ledger_jsonl


class _ByteTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return SimpleNamespace(
            input_ids=[
                int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "big")
                for word in text.split()
            ]
        )


def test_ruler_pins_cells_identity_source_spans_and_scoring() -> None:
    assert RULER_CONTEXT_LENGTHS == (512, 2048, 4096, 8192, 16384, 32768)
    assert RULER_QUERY_COUNTS == (1, 4, 8)
    assert RULER_DEPTH_STRATA == ("early", "middle", "late")
    assert RULER_LONG_CELLS == (
        "16k_4q",
        "16k_8q",
        "32k_4q",
        "32k_8q",
    )
    assert RULER_ARMS == ("native", "recency", "surprise")
    assert RULER_MIN_EPISODES_PER_CELL == 64

    cell = RulerCell(context_length=16384, needles=16, queries=8)
    tokenizer = _ByteTokenizer()
    first = build_ruler_episode(tokenizer, cell=cell, seed=71, example_index=3)
    second = build_ruler_episode(tokenizer, cell=cell, seed=71, example_index=3)
    assert first == second
    assert first.cell.cell_id == "16k_8q"
    assert first.prompt_end > cell.context_length
    assert len(first.input_ids[: cell.context_length]) == cell.context_length
    assert len(first.source_spans) == cell.queries
    assert len(first.answer_spans) == cell.queries
    assert set(first.depth_strata) == set(RULER_DEPTH_STRATA)
    assert all(0 <= start < end <= cell.context_length for start, end in first.source_spans)
    assert all(first.prompt_end <= start < end <= len(first.input_ids) for start, end in first.answer_spans)

    perfect = score_teacher_forced(first, first.input_ids)
    assert perfect.evaluation_mode == "teacher_forced"
    assert perfect.numerator == perfect.denominator == 8
    assert perfect.episode_exact is True

    predictions = list(first.input_ids)
    start, _ = first.answer_spans[0]
    predictions[start] += 1
    imperfect = score_teacher_forced(first, predictions)
    assert imperfect.numerator == 7
    assert imperfect.episode_exact is False

    generated = score_free_generation(first, first.answers)
    assert generated.evaluation_mode == "free_generation"
    assert generated.episode_exact is True


def test_ruler_free_generation_subset_and_feasibility_are_deterministic() -> None:
    tokenizer = _ByteTokenizer()
    episodes = tuple(
        build_ruler_episode(
            tokenizer,
            cell=RulerCell(512, 16, 4),
            seed=seed,
            example_index=index,
        )
        for seed in (11, 22, 33)
        for index in range(4)
    )
    shuffled = list(episodes)
    random.Random(991).shuffle(shuffled)
    selected_a = select_free_generation_subset(episodes, count=5, seed=17)
    selected_b = select_free_generation_subset(shuffled, count=5, seed=17)
    assert tuple(item.episode_id for item in selected_a) == tuple(
        item.episode_id for item in selected_b
    )


def _ruler_identities(
    *,
    arms: tuple[str, ...] = RULER_ARMS,
    seeds_by_arm: dict[str, tuple[int, ...]] | None = None,
    cells: tuple[str, ...] = RULER_LONG_CELLS,
    episodes_per_cell: int = RULER_MIN_EPISODES_PER_CELL,
) -> dict[str, tuple[tuple[int, str, str, str], ...]]:
    seeds_by_arm = seeds_by_arm or {arm: (11, 22, 33) for arm in arms}
    return {
        arm: tuple(
            (seed, f"episode-{seed}-{cell}-{index:03d}", cell, "teacher_forced")
            for seed in seeds_by_arm[arm]
            for cell in cells
            for index in range(episodes_per_cell)
        )
        for arm in arms
    }


def test_ruler_scope_requires_complete_matched_promotion_evidence() -> None:
    assert ruler_evidence_scope(identities=_ruler_identities()) == "promotion"
    assert (
        ruler_evidence_scope(identities=_ruler_identities(arms=("native",)))
        == "feasibility"
    )
    assert (
        ruler_evidence_scope(
            identities=_ruler_identities(
                seeds_by_arm={
                    "native": (11, 22, 33),
                    "recency": (44, 55, 66),
                    "surprise": (11, 22, 33),
                }
            )
        )
        == "feasibility"
    )
    assert (
        ruler_evidence_scope(
            identities=_ruler_identities(cells=RULER_LONG_CELLS[:-1])
        )
        == "feasibility"
    )
    assert (
        ruler_evidence_scope(
            identities=_ruler_identities(
                episodes_per_cell=RULER_MIN_EPISODES_PER_CELL - 1
            )
        )
        == "feasibility"
    )


_LONG_CELL_METADATA = {
    "16k_4q": (16384, 4),
    "16k_8q": (16384, 8),
    "32k_4q": (32768, 4),
    "32k_8q": (32768, 8),
}


def _evaluation(seed: int, cell_id: str, index: int, *, mode: str = "teacher_forced") -> dict:
    context_length, queries = _LONG_CELL_METADATA[cell_id]
    episode_id = f"episode-{seed}-{cell_id}-{index:03d}"
    return {
        "task": "ruler",
        "cell_id": cell_id,
        "context_length": context_length,
        "needles": 16,
        "queries": queries,
        "depth_stratum": RULER_DEPTH_STRATA[index % len(RULER_DEPTH_STRATA)],
        "example_id": episode_id,
        "episode_id": episode_id,
        "evaluation_mode": mode,
        "evidence_scope": "promotion",
        "numerator": queries,
        "denominator": queries,
        "episode_exact": True,
        "source_spans": [[10 + offset, 11 + offset] for offset in range(queries)],
        "target_digest": f"target-{episode_id}",
        "cache_diagnostics": {
            "persistent_hit": 1.0,
            "conditional_read": 1.0,
        },
        "paired_interval": {
            "direction": 1,
            "point": 0.2,
            "lower": 0.1,
            "upper": 0.3,
            "seed_count": 3,
            "example_count": 192,
            "resamples": 100,
        },
    }


def _promotion_records() -> list[dict]:
    records: list[dict] = []
    for arm in RULER_ARMS:
        for seed in (11, 22, 33):
            evaluations = [
                _evaluation(seed, cell, index)
                for cell in RULER_LONG_CELLS
                for index in range(RULER_MIN_EPISODES_PER_CELL)
            ]
            evaluations.extend(
                _evaluation(seed, cell, index, mode="free_generation")
                for cell in RULER_LONG_CELLS
                for index in range(4)
            )
            records.append(
                {
                    "schema_version": "1.0.0",
                    "job_id": f"job-{arm}-{seed}",
                    "experiment_id": "experiment-ruler",
                    "arm_id": arm,
                    "seed": seed,
                    "status": "completed",
                    "scientific_label": (
                        "inconclusive" if arm == "surprise" and seed == 11 else "candidate"
                    ),
                    "metrics": {"token_accuracy": 1.0},
                    "evaluations": evaluations,
                }
            )
    return records


def _shuffle_records(records: list[dict], seed: int) -> list[dict]:
    copied = copy.deepcopy(records)
    rng = random.Random(seed)
    for record in copied:
        rng.shuffle(record["evaluations"])
    rng.shuffle(copied)
    return copied


def test_summary_outputs_are_byte_deterministic_and_preserve_rows(tmp_path: Path) -> None:
    records = _promotion_records()
    first = build_summary_artifacts(_shuffle_records(records, 1), promotion=True)
    second = build_summary_artifacts(_shuffle_records(records, 2), promotion=True)
    assert first.ledger_jsonl == second.ledger_jsonl
    assert first.results_json == second.results_json
    assert first.results_csv == second.results_csv

    rows = [json.loads(line) for line in first.ledger_jsonl.splitlines()]
    assert any(
        row["execution_status"] == "completed"
        and row["scientific_label"] == "inconclusive"
        for row in rows
    )
    assert {row["evaluation"]["evaluation_mode"] for row in rows} == {
        "teacher_forced",
        "free_generation",
    }
    assert all("cache_diagnostics" in row["evaluation"] for row in rows)
    assert all("paired_interval" in row["evaluation"] for row in rows)

    output = tmp_path / "summary"
    written = write_summary_artifacts(_shuffle_records(records, 3), output, promotion=True)
    assert (output / "ledger.jsonl").read_bytes() == written.ledger_jsonl
    assert (output / "results.json").read_bytes() == written.results_json
    assert (output / "results.csv").read_bytes() == written.results_csv


def test_summary_rejects_unmatched_duplicate_missing_and_feasibility_evidence() -> None:
    records = _promotion_records()

    duplicate = list(records) + [records[0]]
    with pytest.raises(SummaryValidationError, match="duplicate.*job"):
        build_summary_artifacts(duplicate, promotion=True)

    unmatched = _shuffle_records(records, 10)
    unmatched[0]["evaluations"].pop()
    with pytest.raises(SummaryValidationError, match="matched.*seed/example/cell"):
        build_summary_artifacts(unmatched, promotion=True)

    missing_cell = _shuffle_records(records, 11)
    for record in missing_cell:
        record["evaluations"] = [
            row for row in record["evaluations"] if row["cell_id"] != "32k_8q"
        ]
    with pytest.raises(SummaryValidationError, match="required RULER cells"):
        build_summary_artifacts(missing_cell, promotion=True)

    feasibility = _shuffle_records(records, 12)
    feasibility[0]["evaluations"][0]["evidence_scope"] = "feasibility"
    with pytest.raises(SummaryValidationError, match="feasibility.*promotion"):
        build_summary_artifacts(feasibility, promotion=True)

    identity_mismatch = _shuffle_records(records, 13)
    recency = next(record for record in identity_mismatch if record["arm_id"] == "recency")
    spans = list(recency["evaluations"][0]["source_spans"])
    spans[0] = [1, 2]
    recency["evaluations"][0]["source_spans"] = spans
    with pytest.raises(SummaryValidationError, match="identity.*arms"):
        build_summary_artifacts(identity_mismatch, promotion=True)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cell_id": "1k_4q", "context_length": 1024}, "pinned.*context"),
        ({"cell_id": "16k_2q", "queries": 2, "denominator": 2, "numerator": 2,
          "source_spans": [[10, 11], [11, 12]]}, "queries.*1, 4, or 8"),
        ({"cell_id": "bogus"}, "cell identity"),
    ],
)
def test_summary_rejects_unpinned_or_inconsistent_ruler_cells(
    updates: dict, message: str
) -> None:
    record = _promotion_records()[0]
    record = copy.deepcopy(record)
    record["evaluations"] = [record["evaluations"][0]]
    record["evaluations"][0].update(updates)
    with pytest.raises(SummaryValidationError, match=message):
        build_summary_artifacts([record])


def test_summary_rejects_duplicate_ruler_rows_without_promotion() -> None:
    record = copy.deepcopy(_promotion_records()[0])
    record["evaluations"] = [record["evaluations"][0], record["evaluations"][0]]
    with pytest.raises(SummaryValidationError, match="duplicate RULER seed/example/cell"):
        build_summary_artifacts([record], promotion=False)


def test_summary_distinguishes_failed_missing_completed_and_inconclusive() -> None:
    completed = {
        "job_id": "job-completed",
        "experiment_id": "experiment",
        "arm_id": "native",
        "seed": 1,
        "status": "completed",
        "scientific_label": "inconclusive",
        "metrics": {"token_accuracy": 0.5},
        "evaluations": [],
    }
    failed = {
        "job_id": "job-failed",
        "experiment_id": "experiment",
        "arm_id": "recency",
        "seed": 1,
        "status": "failed",
        "error": {"code": "oom"},
    }
    expected = [
        {"job_id": "job-completed", "experiment_id": "experiment", "arm_id": "native", "seed": 1},
        {"job_id": "job-failed", "experiment_id": "experiment", "arm_id": "recency", "seed": 1},
        {"job_id": "job-missing", "experiment_id": "experiment", "arm_id": "surprise", "seed": 1},
    ]
    artifacts = build_summary_artifacts([failed, completed], expected_jobs=expected)
    summary = json.loads(artifacts.results_json)
    assert summary["execution_counts"] == {"completed": 1, "failed": 1, "missing": 1}
    rows = [json.loads(line) for line in artifacts.ledger_jsonl.splitlines()]
    statuses = {row["job_id"]: row["execution_status"] for row in rows}
    assert statuses == {
        "job-completed": "completed",
        "job-failed": "failed",
        "job-missing": "missing",
    }
    completed_row = next(row for row in rows if row["job_id"] == "job-completed")
    assert completed_row["scientific_label"] == "inconclusive"


def _simple_completed_record(
    *,
    job_id: str = "job-simple",
    metric: float = 0.5,
    evaluations: list[dict] | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "experiment_id": "experiment-simple",
        "arm_id": "native",
        "seed": 17,
        "backend": "qwen",
        "stage": "qwen_heal",
        "pairing_id": "pair-simple",
        "status": "completed",
        "metrics": {"token_accuracy": metric},
        "evaluations": evaluations or [],
    }


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("experiment_id", "other-experiment"),
        ("arm_id", "surprise"),
        ("seed", 99),
        ("backend", "tiny"),
        ("stage", "tiny_screen"),
        ("pairing_id", "other-pair"),
    ],
)
def test_expected_jobs_reject_every_mismatched_identity_field(
    field: str, wrong_value: object
) -> None:
    record = _simple_completed_record()
    expected = {
        key: record[key]
        for key in (
            "job_id",
            "experiment_id",
            "arm_id",
            "seed",
            "backend",
            "stage",
            "pairing_id",
        )
    }
    record[field] = wrong_value
    with pytest.raises(SummaryValidationError, match=rf"expected job identity.*{field}"):
        build_summary_artifacts([record], expected_jobs=[expected])


@pytest.mark.parametrize("fail_after", (1, 2, 3, 4))
def test_summary_publication_rolls_back_every_partial_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after: int
) -> None:
    output = tmp_path / "summary"
    write_summary_artifacts([_simple_completed_record(metric=0.25)], output)
    names = ("ledger.jsonl", "results.json", "results.csv")
    previous = {name: (output / name).read_bytes() for name in names}
    original = summarize_module._atomic_write
    calls = 0

    def fail_after_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        original(path, payload)
        if calls == fail_after:
            raise OSError(f"injected publication failure after stage {fail_after}")

    monkeypatch.setattr(summarize_module, "_atomic_write", fail_after_write)
    with pytest.raises(OSError, match="injected publication failure"):
        write_summary_artifacts([_simple_completed_record(metric=0.75)], output)
    assert {name: (output / name).read_bytes() for name in names} == previous


def test_non_ruler_rows_use_canonical_final_sort_key() -> None:
    evaluations = [
        {"task": "other", "metric": "beta", "value": 2},
        {"task": "other", "metric": "alpha", "value": 1},
    ]
    forward = build_summary_artifacts(
        [_simple_completed_record(evaluations=copy.deepcopy(evaluations))]
    )
    reverse = build_summary_artifacts(
        [_simple_completed_record(evaluations=list(reversed(copy.deepcopy(evaluations))))]
    )
    assert forward == reverse


def test_generic_evaluation_sort_accepts_mixed_json_field_types() -> None:
    evaluations = [
        {
            "task": "other",
            "cell_id": 1,
            "example_id": "same-example",
            "evaluation_mode": "generic",
            "value": "numeric-cell",
        },
        {
            "task": "other",
            "cell_id": "1",
            "example_id": "same-example",
            "evaluation_mode": "generic",
            "value": "string-cell",
        },
    ]
    forward = build_summary_artifacts(
        [_simple_completed_record(evaluations=copy.deepcopy(evaluations))]
    )
    reverse = build_summary_artifacts(
        [_simple_completed_record(evaluations=list(reversed(copy.deepcopy(evaluations))))]
    )
    assert forward == reverse


def test_ruler_aggregate_ignores_non_ruler_arms_and_seeds() -> None:
    ruler = copy.deepcopy(_promotion_records()[0])
    ruler["evaluations"] = [ruler["evaluations"][0]]
    unrelated = _simple_completed_record(
        job_id="job-unrelated",
        evaluations=[{"task": "other", "metric": "accuracy", "value": 1.0}],
    )
    unrelated["arm_id"] = "unrelated-arm"
    unrelated["seed"] = 999
    summary = json.loads(build_summary_artifacts([unrelated, ruler]).results_json)
    assert summary["ruler"]["arms"] == ["native"]
    assert summary["ruler"]["seeds"] == [11]


def _interval(value: float, *, direction: int = 1) -> BootstrapInterval:
    return BootstrapInterval(value, value, value, direction, 3, 192, 100)


def _option3(branch: str) -> Option3Evidence:
    return Option3Evidence(
        branch=branch,
        valid=True,
        mechanism_unchanged=True,
        long_macro=_interval(0.15),
        long_cells=tuple(NamedInterval(cell, _interval(0.15)) for cell in RULER_LONG_CELLS),
        surprise_vs_recency=_interval(0.08) if branch == "surprise" else None,
        short_macro=_interval(0.0),
        short_cells=tuple(
            NamedInterval(cell, _interval(0.0)) for cell in ("512", "1k", "2k", "4k")
        ),
        eight_k=_interval(0.0),
        episode_exact=_interval(0.0),
        freshness_latest_native=_interval(0.0),
        freshness_latest_recency=_interval(0.0),
        freshness_stale_native=_interval(0.0, direction=-1),
        freshness_stale_recency=_interval(0.0, direction=-1),
        ce_delta=_interval(0.0),
        kl_delta=_interval(0.0),
        mean_kl_native=0.1,
        nonfinite_count=0,
        skipped_steps=0,
        capacity_w64=_interval(0.15),
        capacity_w32=_interval(0.06),
        capacity_w128=_interval(0.06),
        requires_width_above_128=False,
        amplitudes=(0.005, 0.02),
        persistent_hit=_interval(0.30),
        conditional_read=_interval(0.60),
        shuffle_drop=_interval(0.06),
        persistent_bytes=10 * 1024 * 1024,
        persistent_bytes_limit=10 * 1024 * 1024,
        decode_throughput_ratio=0.80,
        prefill_throughput_ratio=0.75,
        dynamic_memory_flat=True,
    )


def test_option3_summary_uses_metrics_gate_order_and_rejects_feasibility() -> None:
    surprise = _option3("surprise")
    recency = _option3("recency")
    selected = decide_option3_summary(surprise, recency, Option3Thresholds())
    assert selected.selected_branch == "surprise"

    failed_surprise = replace(surprise, long_macro=_interval(0.09))
    selected = decide_option3_summary(failed_surprise, recency, Option3Thresholds())
    assert selected.selected_branch == "recency"
    assert selected.surprise.rejection_codes == ("long_macro_lcb",)

    selected = decide_option3_summary(
        failed_surprise,
        replace(recency, long_macro=_interval(0.09)),
        Option3Thresholds(),
    )
    assert selected.selected_branch == "no_promote"
    with pytest.raises(SummaryValidationError, match="feasibility.*promotion"):
        decide_option3_summary(
            surprise,
            recency,
            Option3Thresholds(),
            evidence_scope="feasibility",
        )


def _samples(value: float, *, missing_last: bool = False) -> tuple[MetricSample, ...]:
    result = tuple(
        MetricSample(seed, f"example-{index}", "budget", value, 1.0)
        for seed in (1, 2, 3)
        for index in range(2)
    )
    return result[:-1] if missing_last else result


def test_factorial_and_reliance_summary_delegate_matched_classifiers_and_vetoes() -> None:
    cells = {
        "M00": _samples(0.2),
        "M10": _samples(0.4),
        "M01": _samples(0.2),
        "M11": _samples(0.6),
    }
    safe = ProtectedEffect("latency_ms", _interval(0.0, direction=-1), 0.1)
    classified = classify_factorial_addition(
        cells,
        metric="token_accuracy",
        protected=(safe,),
        valid=True,
        min_useful=0.1,
        harm_threshold=0.1,
        min_synergy=0.05,
        random_seed=19,
        resamples=100,
    )
    assert classified.label == "synergistic"
    assert classified.statistics.interaction.point == pytest.approx(0.2)

    harmful = ProtectedEffect("latency_ms", _interval(-0.2, direction=-1), 0.1)
    vetoed = classify_factorial_addition(
        cells,
        metric="token_accuracy",
        protected=(harmful,),
        valid=True,
        min_useful=0.1,
        harm_threshold=0.1,
        min_synergy=0.05,
        random_seed=19,
        resamples=100,
    )
    assert vetoed.label == "harmful"

    with pytest.raises(ValueError, match="all four factorial cells"):
        classify_factorial_addition(
            {key: value for key, value in cells.items() if key != "M00"},
            metric="token_accuracy",
            protected=(),
            valid=True,
            min_useful=0.1,
            harm_threshold=0.1,
            min_synergy=0.05,
            random_seed=19,
            resamples=100,
        )
    with pytest.raises(ValueError, match="matched seed/example"):
        classify_factorial_addition(
            dict(cells, M11=_samples(0.6, missing_last=True)),
            metric="token_accuracy",
            protected=(),
            valid=True,
            min_useful=0.1,
            harm_threshold=0.1,
            min_synergy=0.05,
            random_seed=19,
            resamples=100,
        )

    reliance = classify_paired_reliance(
        current=_samples(0.6),
        ablated=_samples(0.3),
        metric="token_accuracy",
        valid=True,
        min_reliance=0.1,
        equivalence=0.02,
        harm_threshold=0.1,
        random_seed=23,
        resamples=100,
    )
    assert reliance.label == "relied-on"
    assert reliance.interval.point == pytest.approx(0.3)
