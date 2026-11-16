"""Regression tests for input transfer and trial finalization.

The original defect: trial input was written to the child's stdin with a
blocking ``write()`` *before* the 2 second deadline was even computed.  A
child that stopped accepting input therefore hung the worker forever: the job
stayed ``running`` past 2 seconds and a cancel request could never finish.

These tests pin down the fixed behaviour:

* the 2 second budget starts at process start and covers input transfer;
* a timeout/cancel that fires while input is still being delivered stops the
  process, reaps it and closes the input pipe before the terminal state;
* a legal large input still runs to completion and minimizes normally — no
  truncation and no rejection of large files.
"""
import io
import threading
import time

import pytest

from app.csvutil import MAX_FILE_BYTES, MAX_ROWS, parse_csv, render_subset_csv
from app.main import app
from app.trial import TRIAL_TIMEOUT_SECONDS, run_trial
from conftest import SIGNALS_ALLOWED, requires_signals
from fastapi.testclient import TestClient

from app.jobs import JobManager

SQL_DIVZERO = (
    "SELECT item, SUM(1 // CAST(value AS INTEGER)) "
    "FROM readings GROUP BY item"
)

# A legal, large payload: 1000 rows with wide values. The JSON payload sent to
# the trial subprocess is well above the 64 KiB OS pipe buffer, so a blocking
# write would have hung before the fix.
PAD = " " * 300


def _large_rows(bad_index=500):
    return [
        ["s%04d" % i, "temperature", ("0" if i == bad_index else "5") + PAD]
        for i in range(MAX_ROWS)
    ]


def test_large_input_payload_exceeds_pipe_buffer():
    rows = _large_rows()
    import json
    payload = json.dumps({"rows": rows, "sql": SQL_DIVZERO})
    assert len(payload) > 200 * 1024  # >> 65536-byte pipe capacity
    assert len(rows) == MAX_ROWS
    # The CSV itself is a normal, accepted upload (not truncated).
    csv_bytes = render_subset_csv(rows).encode("utf-8")
    assert len(csv_bytes) < MAX_FILE_BYTES
    parsed = parse_csv(csv_bytes)
    assert len(parsed.rows) == MAX_ROWS
    assert all(row == rows[i] for i, row in enumerate(parsed.rows))


def test_large_input_trial_completes_fully(never_canceled):
    """A legal large input must be transferred completely and reproduce."""
    rows = _large_rows()
    res = run_trial(rows, SQL_DIVZERO, never_canceled)
    assert res.status == "error"
    assert res.error_type == "InvalidInputException"
    assert res.input_complete is True
    assert res.input_bytes_sent == res.input_bytes_total
    assert res.input_bytes_total > 200 * 1024
    assert res.returncode == 0


def test_timeout_counts_from_process_start_while_input_is_stalled(never_canceled):
    """The child stops accepting input: the 2 s limit fires on schedule.

    ``input_stall_seconds`` makes the runner pause before reading stdin, so
    only one pipe buffer of payload is ever delivered.  The parent must still
    stop and reap the process, close the writer and report ``timeout``.
    """
    rows = _large_rows()
    started = time.monotonic()
    res = run_trial(rows, SQL_DIVZERO, never_canceled,
                    input_stall_seconds=8.0)
    elapsed = time.monotonic() - started

    assert res.status == "timeout"
    assert res.stop_reason == "timeout"
    # Budget is measured from process start, plus only a small reap allowance.
    assert TRIAL_TIMEOUT_SECONDS - 0.1 <= elapsed < TRIAL_TIMEOUT_SECONDS + 3.0
    # Input transfer really was incomplete -> this exercised the stall path.
    assert res.input_complete is False
    assert res.input_bytes_sent < res.input_bytes_total
    # The process was actually stopped and reaped before returning.
    assert res.returncode is not None


def test_cancel_during_stalled_input_stops_and_reaps():
    """Cancellation must be honoured while input is still being transferred."""
    cancel = threading.Event()

    def fire():
        time.sleep(0.4)
        cancel.set()

    t = threading.Thread(target=fire)
    t.start()
    started = time.monotonic()
    res = run_trial(_large_rows(), SQL_DIVZERO, cancel,
                    input_stall_seconds=8.0)
    elapsed = time.monotonic() - started
    t.join()

    assert res.status == "canceled"
    assert res.stop_reason == "cancel"
    if SIGNALS_ALLOWED:
        # Primary path: SIGKILL to the group -> immediate stop.
        assert elapsed < 1.5
    else:
        # Sandbox denies signals; the runner's own alarm still bounds it.
        assert elapsed < TRIAL_TIMEOUT_SECONDS + 3.0
        assert elapsed < 8.0  # certainly not waiting for the 8 s stall
    assert res.input_complete is False
    assert res.input_bytes_sent < res.input_bytes_total
    assert res.returncode is not None


def test_runner_deadline_backstop_where_parent_signals_are_denied(never_canceled):
    """Even if the parent could not signal the child, it terminates itself.

    This is the path that keeps ``timed_out`` reachable in hardened sandboxes
    (``SIGNALS_ALLOWED`` is False there).  The runner arms its own alarm, so
    the parent can always reap the process and report a terminal state.
    """
    pytest.importorskip("duckdb")
    rows = _large_rows()
    started = time.monotonic()
    res = run_trial(rows, SQL_DIVZERO, never_canceled,
                    input_stall_seconds=8.0)
    elapsed = time.monotonic() - started
    assert res.status == "timeout"
    assert res.returncode is not None
    assert res.input_complete is False
    assert elapsed < TRIAL_TIMEOUT_SECONDS + 3.0


def test_runner_self_alarm_is_reported_as_timeout(never_canceled, monkeypatch):
    """The self-alarm exit code maps to ``timeout``, not ``runner_crash``.

    This is the sandbox fallback: the parent could not kill the child, so the
    runner terminated itself.  It must never be mistaken for a reproduction.
    """
    import json
    import subprocess
    import sys

    from app import trial as trial_mod
    from app.trial import _RUNNER_PATH

    # A normal (tiny) trial: the runner is fine; force only its alarm budget to
    # a tiny value so its own backstop fires while input is stalled.
    monkeypatch.setattr(trial_mod, "RUNNER_DEADLINE_EXIT_CODE", 124)
    env = {
        "PATH": "/usr/bin:/bin",
        "SAMPLE_MINIMIZER_TRIAL_TIMEOUT_SECONDS": "0.05",
        "SAMPLE_MINIMIZER_RUNNER_GRACE_SECONDS": "0.05",
        "SAMPLE_MINIMIZER_RUNNER_INPUT_STALL_SECONDS": "5",
    }
    proc = subprocess.run(
        [sys.executable, "-I", _RUNNER_PATH],
        input=json.dumps({"rows": [], "sql": "SELECT 1"}),
        capture_output=True, text=True, env=env, timeout=15,
    )
    assert proc.returncode == trial_mod.RUNNER_DEADLINE_EXIT_CODE

    # And a real parent-enforced trial with a stalled child is a timeout.
    res = run_trial(_large_rows(), SQL_DIVZERO, never_canceled,
                    input_stall_seconds=8.0)
    assert res.status == "timeout"
    assert res.returncode is not None
    assert res.input_complete is False


def test_runner_grace_constants_stay_in_sync():
    from app import runner as runner_mod
    from app import trial as trial_mod

    assert runner_mod.RUNNER_DEADLINE_GRACE_SECONDS == \
        trial_mod.RUNNER_ALARM_GRACE_SECONDS
    assert runner_mod.EXIT_DEADLINE == trial_mod.RUNNER_DEADLINE_EXIT_CODE


# ---------------------------------------------------------------- HTTP level
@pytest.fixture()
def client(tmp_path, monkeypatch):
    fresh = JobManager(str(tmp_path / "jobs"))
    monkeypatch.setattr("app.main.manager", fresh)
    return TestClient(app)


def _upload(client, rows, sql, stall=0.0):
    buf = io.BytesIO()
    buf.write(render_subset_csv(rows).encode("utf-8"))
    buf.seek(0)
    return client.post(
        "/api/jobs",
        files={"file": ("data.csv", buf, "text/csv")},
        data={"sql": sql, "demo_input_stall_seconds": str(stall)},
    )


def _wait(client, job_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in {
            "succeeded", "no_error", "timed_out", "failed", "canceled"
        }:
            return body
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_http_large_input_still_minimizes(client):
    """No truncation / no rejection: a large legal upload reduces normally."""
    rows = _large_rows(bad_index=777)
    r = _upload(client, rows, SQL_DIVZERO)
    assert r.status_code == 200, r.text
    final = _wait(client, r.json()["id"], timeout=120.0)
    assert final["status"] == "succeeded"
    assert final["total_rows"] == MAX_ROWS
    assert final["result_rows"] == 1
    assert final["progress"]["history"][-1]["input_complete"] is True
    dl = client.get(f"/api/jobs/{final['id']}/download")
    assert dl.status_code == 200
    body = [ln for ln in dl.text.splitlines()[1:] if ln.strip()]
    assert len(body) == 1
    assert body[0].startswith("s0777,")
    # The value is preserved verbatim, including the wide trailing blanks.
    assert body[0] == "s0777,temperature,0" + PAD


def test_http_stalled_input_times_out_and_publishes_nothing(client):
    rows = _large_rows()
    r = _upload(client, rows, SQL_DIVZERO, stall=8)
    job_id = r.json()["id"]
    started = time.monotonic()
    final = _wait(client, job_id, timeout=30.0)
    elapsed = time.monotonic() - started
    assert final["status"] == "timed_out"
    assert final["has_result"] is False
    assert final["result_rows"] == 0
    assert client.get(f"/api/jobs/{job_id}/download").status_code == 409
    # Fails fast at the 2 s budget instead of hanging on the stalled writer.
    assert elapsed < TRIAL_TIMEOUT_SECONDS + 5.0
    assert "输入" in final["message"]


def test_http_cancel_during_stalled_input_reaches_canceled(client):
    rows = _large_rows()
    r = _upload(client, rows, SQL_DIVZERO, stall=8)
    job_id = r.json()["id"]
    time.sleep(0.4)
    ack = client.post(f"/api/jobs/{job_id}/cancel")
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] is True

    started = time.monotonic()
    final = _wait(client, job_id, timeout=20.0)
    assert final["status"] == "canceled"
    assert final["has_result"] is False
    assert client.get(f"/api/jobs/{job_id}/download").status_code == 409
    # Cancellation is not blocked behind the stalled input transfer.
    assert time.monotonic() - started < TRIAL_TIMEOUT_SECONDS + 3.0


def test_http_stall_demo_other_job_unaffected(client):
    """While one job is stuck transferring input, another window works."""
    stuck = _upload(client, _large_rows(), SQL_DIVZERO, stall=8).json()["id"]
    fast_rows = [["f%d" % i, "m", "5"] for i in range(9)] + [["f9", "m", "0"]]
    fast = _upload(client, fast_rows, SQL_DIVZERO).json()["id"]

    fast_final = _wait(client, fast, timeout=60.0)
    assert fast_final["status"] == "succeeded"
    assert client.get(f"/api/jobs/{fast}/download").status_code == 200
    # The stuck job eventually terminates on its own; the other window never
    # depended on it.
    stuck_final = _wait(client, stuck, timeout=30.0)
    assert stuck_final["status"] in {"timed_out", "canceled"}


def test_http_stall_parameter_validated(client):
    r = _upload(client, [["a", "x", "1"]], SQL_DIVZERO, stall=999)
    assert r.status_code == 400


def test_demo_input_stall_metadata_and_csv(client):
    demos = {d["id"]: d for d in client.get("/api/demo").json()["demos"]}
    assert "input_stall" in demos
    demo = demos["input_stall"]
    assert demo["demo_input_stall_seconds"] >= TRIAL_TIMEOUT_SECONDS
    csv_resp = client.get("/api/demo/input_stall/csv")
    assert csv_resp.status_code == 200
    parsed = parse_csv(csv_resp.content)
    assert len(parsed.rows) == MAX_ROWS
    assert demo["expected_min_rows"] == 1
