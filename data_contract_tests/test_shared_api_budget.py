from concurrent.futures import ThreadPoolExecutor
import io
import json

import pytest

from scripts.run_paired_generator_diagnostic import ExternalJudge
from scripts.shared_api_budget import BudgetConfig, BudgetExceeded, SharedApiBudget


def test_shared_request_cap_is_atomic(tmp_path):
    config = BudgetConfig(3, 10.0, 2.0, 8.0)
    ledger = tmp_path / "budget.sqlite3"

    def attempt(index):
        guard = SharedApiBudget(ledger, config)
        try:
            return guard.reserve(
                max_input_tokens=1000,
                max_output_tokens=1000,
                worker_id=f"worker-{index % 4}",
                purpose="mock",
            )
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(attempt, range(20)))
    accepted = [item for item in reservations if item]
    assert len(accepted) == 3
    guard = SharedApiBudget(ledger, config)
    guard.settle(accepted[0], actual_input_tokens=100, actual_output_tokens=50)
    guard.fail(accepted[1], "mock transport failure")
    status = guard.status()
    assert status["calls"] == 3
    assert status["by_status"]["settled"]["calls"] == 1
    assert status["by_status"]["failed_reserved"]["calls"] == 1


def test_usd_cap_reserves_worst_case_before_dispatch(tmp_path):
    config = BudgetConfig(10, 0.01, 2.0, 8.0)
    guard = SharedApiBudget(tmp_path / "budget.sqlite3", config)
    guard.reserve(max_input_tokens=1000, max_output_tokens=1000, worker_id="a", purpose="mock")
    with pytest.raises(BudgetExceeded, match="USD cap"):
        guard.reserve(max_input_tokens=1000, max_output_tokens=1000, worker_id="b", purpose="mock")


def test_ledger_settings_are_immutable(tmp_path):
    ledger = tmp_path / "budget.sqlite3"
    SharedApiBudget(ledger, BudgetConfig(3, 1.0, 2.0, 8.0))
    with pytest.raises(ValueError, match="different immutable settings"):
        SharedApiBudget(ledger, BudgetConfig(4, 1.0, 2.0, 8.0))


def test_external_judge_mock_settles_and_fails_closed_before_second_call(tmp_path, monkeypatch):
    config = BudgetConfig(1, 1.0, 2.0, 8.0)
    guard = SharedApiBudget(tmp_path / "budget.sqlite3", config)
    body = {
        "model": "gpt-4.1-2025-04-14",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
    }
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    judge = ExternalJudge(
        "https://example.invalid",
        max_tokens=100,
        budget_guard=guard,
        budget_max_input_tokens=1000,
        budget_worker_id="mock-worker",
    )
    assert judge.call("short mock prompt")[0] == "{}"
    assert len(calls) == 1
    assert guard.status()["by_status"]["settled"]["calls"] == 1
    with pytest.raises(BudgetExceeded, match="request cap"):
        judge.call("second prompt must not dispatch")
    assert len(calls) == 1


def test_guard_does_not_change_judge_payload_or_output_allowance(tmp_path, monkeypatch):
    body = {
        "model": "gpt-4.1-2025-04-14",
        "usage": {"prompt_tokens": 40, "completion_tokens": 7},
        "choices": [{"finish_reason": "stop", "message": {"content": '{"same": true}'}}],
    }
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    unguarded = ExternalJudge("https://example.invalid", max_tokens=321)
    guarded = ExternalJudge(
        "https://example.invalid",
        max_tokens=321,
        budget_guard=SharedApiBudget(tmp_path / "budget.sqlite3", BudgetConfig(2, 1.0, 2.0, 8.0)),
        budget_max_input_tokens=2000,
        budget_worker_id="mock",
    )
    assert unguarded.call("identical prompt")[0] == guarded.call("identical prompt")[0]
    assert payloads[0] == payloads[1]
    assert payloads[1]["max_tokens"] == 321


def test_input_ceiling_blocks_before_transport_and_failed_attempt_is_reserved_on_restart(tmp_path, monkeypatch):
    ledger = tmp_path / "budget.sqlite3"
    config = BudgetConfig(2, 1.0, 2.0, 8.0)
    guard = SharedApiBudget(ledger, config)
    dispatched = []

    def fail_urlopen(request, timeout):
        dispatched.append(request)
        raise OSError("mock transport failure")

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    too_small = ExternalJudge(
        "https://example.invalid", max_tokens=10, budget_guard=guard,
        budget_max_input_tokens=10, budget_worker_id="worker-a",
    )
    with pytest.raises(ValueError, match="pre-reservation"):
        too_small.call("request too large")
    assert not dispatched and guard.status()["calls"] == 0

    judge = ExternalJudge(
        "https://example.invalid", max_tokens=10, budget_guard=guard,
        budget_max_input_tokens=2000, budget_worker_id="worker-a",
    )
    with pytest.raises(OSError):
        judge.call("first physical request")
    restarted = SharedApiBudget(ledger, config)
    assert restarted.status()["by_status"]["failed_reserved"]["calls"] == 1
    with pytest.raises(OSError):
        ExternalJudge(
            "https://example.invalid", max_tokens=10, budget_guard=restarted,
            budget_max_input_tokens=2000, budget_worker_id="worker-b",
        ).call("retry is a second physical request")
    assert len(dispatched) == 2
    assert restarted.status()["calls"] == 2
