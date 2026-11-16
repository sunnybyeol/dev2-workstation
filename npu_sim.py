"""Pure-Python 2D MAC operations and dataset validation for the Mini NPU lab."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Sequence, Tuple

EPSILON = 1e-9
BENCHMARK_REPEATS = 10
LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"


def standardize_label(value: Any) -> Optional[str]:
    """Map accepted aliases to Cross, X, or UNDECIDED."""
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"cross", "+", "십자", "plus"}:
        return LABEL_CROSS
    if text in {"x", "×", "엑스"}:
        return LABEL_X
    if text in {"undecided", "tie", "동점", "판정 불가"}:
        return LABEL_UNDECIDED
    return None


def validate_binary_matrix(value: Any, size: int, field: str) -> List[List[int]]:
    """Validate an exact N by N matrix whose entries are 0 or 1."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("{} must have {} rows.".format(field, size))
    matrix: List[List[int]] = []
    for row_number, row in enumerate(value, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(
                "{} row {} must contain exactly {} values.".format(field, row_number, size)
            )
        clean_row: List[int] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or item not in (0, 1):
                raise ValueError("{} values must be 0 or 1.".format(field))
            clean_row.append(int(item))
        matrix.append(clean_row)
    return matrix


def validate_weight_matrix(value: Any, size: int, field: str) -> List[List[float]]:
    """Validate a square finite-number filter matrix; weights may be fractional."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("{} must have {} rows.".format(field, size))
    matrix: List[List[float]] = []
    for row_number, row in enumerate(value, start=1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(
                "{} row {} must contain exactly {} values.".format(field, row_number, size)
            )
        clean_row: List[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("{} weights must be finite numbers.".format(field))
            try:
                weight = float(item)
            except (OverflowError, ValueError):
                raise ValueError("{} weights must be finite numbers.".format(field))
            if not math.isfinite(weight):
                raise ValueError("{} weights must be finite numbers.".format(field))
            clean_row.append(weight)
        matrix.append(clean_row)
    return matrix


def generate_pattern(size: int, label: str) -> List[List[int]]:
    """Generate a Cross (+) or X pattern. Even sizes use both center rows/columns."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("N must be a positive integer.")
    normalized = standardize_label(label)
    if normalized not in (LABEL_CROSS, LABEL_X):
        raise ValueError("Pattern label must be Cross or X.")

    middle = {(size - 1) // 2, size // 2}
    pattern: List[List[int]] = []
    for row in range(size):
        current: List[int] = []
        for column in range(size):
            if normalized == LABEL_CROSS:
                value = int(row in middle or column in middle)
            else:
                value = int(row == column or row + column == size - 1)
            current.append(value)
        pattern.append(current)
    return pattern


def generate_standard_filters(size: int) -> Dict[str, List[List[int]]]:
    return {
        LABEL_CROSS: generate_pattern(size, LABEL_CROSS),
        LABEL_X: generate_pattern(size, LABEL_X),
    }


def flatten(matrix: Sequence[Sequence[int]]) -> List[int]:
    return [value for row in matrix for value in row]


def mac_2d(pattern: Sequence[Sequence[float]], weights: Sequence[Sequence[float]]) -> float:
    """Reference nested-row multiply-accumulate implementation."""
    size = len(pattern)
    if size == 0 or len(weights) != size:
        raise ValueError("Pattern and filter must have the same non-zero size.")
    total = 0.0
    for row in range(size):
        if len(pattern[row]) != size or len(weights[row]) != size:
            raise ValueError("Pattern and filter must both be square N by N matrices.")
        for column in range(size):
            total += float(pattern[row][column]) * float(weights[row][column])
    return total


def mac_flat(pattern: Sequence[float], weights: Sequence[float]) -> float:
    """Flat-array MAC implementation; callers flatten once, outside timed loops."""
    if not pattern or len(pattern) != len(weights):
        raise ValueError("Flat pattern and filter must have the same non-zero length.")
    total = 0.0
    for index in range(len(pattern)):
        total += float(pattern[index]) * float(weights[index])
    return total


def classify_scores(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    """Use a strict epsilon tie rule; the order of scores never breaks a tie."""
    if abs(score_cross - score_x) < epsilon:
        return LABEL_UNDECIDED
    return LABEL_CROSS if score_cross > score_x else LABEL_X


def compare_manual(
    pattern: Sequence[Sequence[float]],
    filter_a: Sequence[Sequence[float]],
    filter_b: Sequence[Sequence[float]],
    epsilon: float = EPSILON,
) -> Tuple[float, float, str]:
    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    if abs(score_a - score_b) < epsilon:
        judgment = LABEL_UNDECIDED
    else:
        judgment = "A" if score_a > score_b else "B"
    return score_a, score_b, judgment


@dataclass
class CaseResult:
    case_id: str
    size: Optional[int]
    score_cross: Optional[float]
    score_x: Optional[float]
    predicted: str
    expected: str
    passed: bool
    reason: str = ""


def _size_from_key(key: str) -> int:
    match = re.fullmatch(r"size_(\d+)_(\d+)", key)
    if not match:
        raise ValueError("case key must use size_{N}_{idx} format.")
    size = int(match.group(1))
    if size < 1:
        raise ValueError("size in case key must be positive.")
    return size


def _filter_pair(filters: Any, size: int) -> Tuple[List[List[float]], List[List[float]]]:
    bank_key = "size_{}".format(size)
    if not isinstance(filters, dict) or not isinstance(filters.get(bank_key), dict):
        raise ValueError("filter bank {} is missing or is not an object.".format(bank_key))
    bank = filters[bank_key]
    canonical: Dict[str, Any] = {}
    for name, matrix in bank.items():
        label = standardize_label(name)
        if label in (LABEL_CROSS, LABEL_X):
            if label in canonical:
                raise ValueError("filter bank {} repeats label {}.".format(bank_key, label))
            canonical[label] = matrix
    if LABEL_CROSS not in canonical or LABEL_X not in canonical:
        raise ValueError("filter bank {} must contain Cross and X filters.".format(bank_key))
    cross = validate_weight_matrix(canonical[LABEL_CROSS], size, bank_key + ".Cross")
    x_filter = validate_weight_matrix(canonical[LABEL_X], size, bank_key + ".X")
    return cross, x_filter


def evaluate_dataset(dataset: Any) -> Tuple[List[CaseResult], List[str]]:
    """Evaluate every case independently, preserving failures instead of aborting."""
    global_errors: List[str] = []
    if not isinstance(dataset, dict):
        return [], ["JSON root must be an object."]
    filters = dataset.get("filters")
    patterns = dataset.get("patterns")
    if not isinstance(patterns, dict):
        return [], ["patterns must be an object keyed as size_{N}_{idx}."]
    if not patterns:
        global_errors.append("patterns contains no test cases.")

    results: List[CaseResult] = []
    for case_id, case_data in patterns.items():
        size: Optional[int] = None
        expected = "<invalid>"
        try:
            if not isinstance(case_id, str):
                raise ValueError("case key must be text.")
            size = _size_from_key(case_id)
            if not isinstance(case_data, dict):
                raise ValueError("case value must be an object.")
            expected_label = standardize_label(case_data.get("expected"))
            if expected_label is None:
                raise ValueError("expected label must be Cross, X, or UNDECIDED.")
            expected = expected_label
            pattern = validate_binary_matrix(case_data.get("input"), size, case_id + ".input")
            cross, x_filter = _filter_pair(filters, size)
            score_cross = mac_2d(pattern, cross)
            score_x = mac_2d(pattern, x_filter)
            predicted = classify_scores(score_cross, score_x)
            passed = predicted == expected
            reason = "" if passed else "expected {}, calculated {}.".format(expected, predicted)
            results.append(
                CaseResult(case_id, size, score_cross, score_x, predicted, expected, passed, reason)
            )
        except (TypeError, ValueError, KeyError) as error:
            results.append(CaseResult(case_id, size, None, None, "FAIL", expected, False, str(error)))
    return results, global_errors


def benchmark_by_size(dataset: Any, repeats: int = BENCHMARK_REPEATS) -> List[Dict[str, Any]]:
    """Compare nested and flat MAC times; matrix validation and flattening are excluded."""
    if repeats < 10:
        raise ValueError("Benchmark requires at least 10 repeats.")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("patterns"), dict):
        return []

    groups: Dict[int, List[Tuple[List[List[int]], List[List[float]], List[List[float]]]]] = {}
    filters = dataset.get("filters")
    for case_id, case_data in dataset["patterns"].items():
        try:
            size = _size_from_key(case_id)
            if not isinstance(case_data, dict):
                continue
            pattern = validate_binary_matrix(case_data.get("input"), size, case_id + ".input")
            cross, x_filter = _filter_pair(filters, size)
            groups.setdefault(size, []).append((pattern, cross, x_filter))
        except (TypeError, ValueError, KeyError):
            continue

    rows: List[Dict[str, Any]] = []
    for size in sorted(groups):
        cases = groups[size]
        flat_cases = [(flatten(p), flatten(c), flatten(x)) for p, c, x in cases]
        for pattern, cross, x_filter in cases:
            reference = (mac_2d(pattern, cross), mac_2d(pattern, x_filter))
            optimized = (mac_flat(flatten(pattern), flatten(cross)), mac_flat(flatten(pattern), flatten(x_filter)))
            if reference != optimized:
                raise AssertionError("flat MAC result differs for size_{}".format(size))

        start = perf_counter_ns()
        for _ in range(repeats):
            for pattern, cross, x_filter in cases:
                mac_2d(pattern, cross)
                mac_2d(pattern, x_filter)
        two_d_ns = perf_counter_ns() - start

        start = perf_counter_ns()
        for _ in range(repeats):
            for pattern, cross, x_filter in flat_cases:
                mac_flat(pattern, cross)
                mac_flat(pattern, x_filter)
        flat_ns = perf_counter_ns() - start

        comparisons = repeats * len(cases)
        rows.append(
            {
                "size": size,
                "cases_per_repeat": len(cases),
                "repeats": repeats,
                "macs_per_filter": size * size,
                "macs_per_case_pair": 2 * size * size,
                "avg_2d_ms": two_d_ns / 1_000_000.0 / comparisons,
                "avg_flat_ms": flat_ns / 1_000_000.0 / comparisons,
            }
        )
    return rows


def benchmark_generated_size(size: int, repeats: int = BENCHMARK_REPEATS) -> Dict[str, Any]:
    filters = generate_standard_filters(size)
    patterns = [generate_pattern(size, LABEL_CROSS), generate_pattern(size, LABEL_X)]
    dataset = {
        "filters": {"size_{}".format(size): filters},
        "patterns": {
            "size_{}_1".format(size): {"input": patterns[0], "expected": LABEL_CROSS},
            "size_{}_2".format(size): {"input": patterns[1], "expected": LABEL_X},
        },
    }
    return benchmark_by_size(dataset, repeats)[0]
