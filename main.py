"""Console entry point for the pure-Python Mini NPU simulator."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns
from typing import List

from npu_sim import (
    BENCHMARK_REPEATS,
    LABEL_CROSS,
    LABEL_UNDECIDED,
    LABEL_X,
    benchmark_by_size,
    benchmark_generated_size,
    compare_manual,
    evaluate_dataset,
    generate_pattern,
    generate_standard_filters,
    mac_flat,
    mac_2d,
    flatten,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.json"


class InputCancelled(Exception):
    """Raised for Ctrl+C or EOF so the program can leave cleanly."""


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError) as error:
        print("\n입력이 끝나 안전하게 종료합니다.")
        raise InputCancelled from error


def ask_int(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        raw = ask(prompt)
        try:
            value = int(raw)
        except ValueError:
            print("정수를 입력해 주세요.")
            continue
        if minimum <= value <= maximum:
            return value
        print("{}부터 {} 사이의 정수를 입력해 주세요.".format(minimum, maximum))


def read_matrix(size: int, title: str) -> List[List[int]]:
    print("{} (각 줄에 0/1 {}개, 공백으로 구분)".format(title, size))
    matrix: List[List[int]] = []
    for row_number in range(1, size + 1):
        while True:
            raw = ask("{}행: ".format(row_number))
            parts = raw.split()
            try:
                row = [int(part) for part in parts]
            except ValueError:
                print("입력 형식 오류: 각 줄에 {}개의 0 또는 1을 공백으로 구분해 입력하세요.".format(size))
                continue
            if len(row) != size or any(value not in (0, 1) for value in row):
                print("입력 형식 오류: 각 줄에 {}개의 0 또는 1을 공백으로 구분해 입력하세요.".format(size))
                continue
            matrix.append(row)
            break
    return matrix


def print_matrix(matrix: List[List[int]]) -> None:
    for row in matrix:
        print(" ".join(str(value) for value in row))


def show_benchmark_rows(rows: List[dict]) -> None:
    print("\n성능 분석 (입출력 제외, {}회 반복 평균)".format(BENCHMARK_REPEATS))
    print("크기 | 사례/반복 | 2D 평균(ms) | 1D 평균(ms) | MAC/필터 | MAC/비교")
    for row in rows:
        print(
            "{size}x{size} | {cases_per_repeat} | {avg_2d_ms:.6f} | {avg_flat_ms:.6f} | "
            "{macs_per_filter} | {macs_per_case_pair}".format(**row)
        )
    if not rows:
        print("유효한 패턴 데이터가 없어 성능을 측정하지 못했습니다.")


def mode_manual() -> None:
    print("\n=== 모드 1: 사용자 입력 (3x3) ===")
    filter_a = read_matrix(3, "필터 A")
    filter_b = read_matrix(3, "필터 B")
    pattern = read_matrix(3, "패턴")
    mac_started = perf_counter_ns()
    score_a, score_b, judgment = compare_manual(pattern, filter_a, filter_b)
    mac_elapsed_ms = (perf_counter_ns() - mac_started) / 1_000_000.0

    flat_a = mac_flat(flatten(pattern), flatten(filter_a))
    flat_b = mac_flat(flatten(pattern), flatten(filter_b))
    if (score_a, score_b) != (flat_a, flat_b):
        raise AssertionError("2D와 1D MAC 점수가 다릅니다.")

    print("\nMAC 결과")
    print("A 점수: {:.1f}".format(score_a))
    print("B 점수: {:.1f}".format(score_b))
    print("판정: {}".format(judgment))
    print("MAC 횟수: 필터당 9회, A/B 합계 18회")
    print("2개 필터 MAC 연산 시간: {:.6f} ms".format(mac_elapsed_ms))
    print("2D/1D 결과 일치: PASS")


def load_dataset(path: Path = DATA_PATH):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mode_json() -> None:
    print("\n=== 모드 2: data.json 분석 ===")
    try:
        dataset = load_dataset()
    except (OSError, json.JSONDecodeError) as error:
        print("data.json을 읽지 못했습니다: {}".format(error))
        return

    results, global_errors = evaluate_dataset(dataset)
    for message in global_errors:
        print("데이터 오류: {}".format(message))
    for result in results:
        if result.score_cross is None or result.score_x is None:
            print("{} | FAIL | 원인: {}".format(result.case_id, result.reason))
            continue
        status = "PASS" if result.passed else "FAIL"
        print(
            "{} | Cross={:.6f} | X={:.6f} | 판정={} | expected={} | {}".format(
                result.case_id,
                result.score_cross,
                result.score_x,
                result.predicted,
                result.expected,
                status,
            )
        )
        if result.reason:
            print("  실패 원인: {}".format(result.reason))

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed + len(global_errors)
    print("\n전체 테스트: {}개".format(len(results)))
    print("통과: {}개".format(passed))
    print("실패: {}개".format(failed))
    if not failed:
        print("실패 케이스: 없음")
    else:
        print("실패 케이스 요약:")
        for result in results:
            if not result.passed:
                print("- {}: {}".format(result.case_id, result.reason or "판정 불일치"))
        for message in global_errors:
            print("- data.json: {}".format(message))

    try:
        show_benchmark_rows(benchmark_by_size(dataset, BENCHMARK_REPEATS))
    except (TypeError, ValueError, AssertionError) as error:
        print("성능 분석을 마치지 못했습니다: {}".format(error))


def mode_generate() -> None:
    print("\n=== 보너스: 패턴 자동 생성 + 1D 최적화 비교 ===")
    size = ask_int("패턴 크기 N (1-25): ", 1, 25)
    label_choice = ask_int("생성할 패턴 (1=Cross, 2=X): ", 1, 2)
    label = LABEL_CROSS if label_choice == 1 else LABEL_X
    pattern = generate_pattern(size, label)
    filters = generate_standard_filters(size)

    print("\n생성된 {} 패턴 ({}x{})".format(label, size, size))
    print_matrix(pattern)
    score_cross = mac_2d(pattern, filters[LABEL_CROSS])
    score_x = mac_2d(pattern, filters[LABEL_X])
    print("Cross 점수: {:.1f}".format(score_cross))
    print("X 점수: {:.1f}".format(score_x))
    print("판정: {}".format("Cross" if score_cross > score_x else "X" if score_x > score_cross else LABEL_UNDECIDED))

    row = benchmark_generated_size(size, BENCHMARK_REPEATS)
    show_benchmark_rows([row])


def show_menu() -> None:
    print("\n===== Mini NPU Simulator =====")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석 (5x5, 13x13, 25x25)")
    print("3. 보너스: 패턴 자동 생성 + 성능 비교")
    print("4. 종료")


def main() -> None:
    actions = {1: mode_manual, 2: mode_json, 3: mode_generate}
    try:
        while True:
            show_menu()
            selection = ask_int("모드 선택 (1-4): ", 1, 4)
            if selection == 4:
                print("시뮬레이터를 종료합니다.")
                break
            actions[selection]()
    except InputCancelled:
        print("작업을 안전하게 마쳤습니다.")


if __name__ == "__main__":
    main()
