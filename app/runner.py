"""Isolated trial runner.

Executed in a *fresh* Python subprocess (``python -I``) for every single
trial so that:

* each trial gets its own in-memory DuckDB database (no state leakage);
* the 2 second wall-clock limit (measured from process start) can be enforced
  by the parent killing the whole process group;
* a crash inside DuckDB only takes the trial down, never the web server.

This script deliberately imports nothing from the application package: under
``python -I`` the project directory is not on ``sys.path``.  Only the standard
library and the installed ``duckdb`` package are required.

Self-imposed deadline
---------------------
The parent sets ``SAMPLE_MINIMIZER_TRIAL_TIMEOUT_SECONDS``.  The runner arms a
``SIGALRM`` (``ITIMER_REAL``) slightly after that budget and exits with a
dedicated code if it fires.  Normally the parent enforces the limit first;
the alarm is a backstop so the process still terminates and can be reaped in
sandboxes that deny the parent permission to signal a live child.

Demo hook
---------
``SAMPLE_MINIMIZER_RUNNER_INPUT_STALL_SECONDS`` makes the runner pause *before
reading stdin*, simulating a trial process that has stopped accepting input.
It is unset (0) in normal operation.

Protocol
--------
stdin  : one JSON message describing the trial (rows + deterministic SELECT).
stdout : exactly one JSON line, one of
         {"status": "success", "elapsed_ms": ...}
         {"status": "error", "error_type": "...", "first_line": "...",
          "elapsed_ms": ...}
stderr : diagnostics only (never parsed by the parent).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback

# Exit code used when the self-imposed trial deadline fires.
EXIT_DEADLINE = 124
# Extra time after the parent's budget before the self-alarm fires.  Keep in
# sync with RUNNER_ALARM_GRACE_SECONDS in trial.py.
RUNNER_DEADLINE_GRACE_SECONDS = 0.25


def _trial_timeout() -> float:
    try:
        return float(os.environ.get("SAMPLE_MINIMIZER_TRIAL_TIMEOUT_SECONDS", "2"))
    except ValueError:
        return 2.0


def _deadline_grace() -> float:
    try:
        return float(
            os.environ.get(
                "SAMPLE_MINIMIZER_RUNNER_GRACE_SECONDS",
                str(RUNNER_DEADLINE_GRACE_SECONDS),
            )
        )
    except ValueError:
        return RUNNER_DEADLINE_GRACE_SECONDS


def _arm_deadline_alarm() -> None:
    """Backstop: exit non-zero if the parent could not stop us in time."""
    try:
        # Small grace so the parent's own enforcement wins in normal setups.
        delay = _trial_timeout() + _deadline_grace()
        signal.signal(
            signal.SIGALRM,
            lambda signum, frame: os._exit(EXIT_DEADLINE),
        )
        signal.setitimer(signal.ITIMER_REAL, max(0.1, delay))
    except (AttributeError, OSError, ValueError):
        pass  # e.g. platform without SIGALRM; parent still enforces the limit


def main() -> int:
    _arm_deadline_alarm()

    stall = 0.0
    try:
        stall = float(
            os.environ.get("SAMPLE_MINIMIZER_RUNNER_INPUT_STALL_SECONDS", "0")
        )
    except ValueError:
        stall = 0.0
    if stall > 0:
        # Simulated "process stopped accepting input": do not read stdin yet.
        time.sleep(stall)

    payload = json.load(sys.stdin)
    rows = payload["rows"]
    sql = payload["sql"]

    # Imported inside main so an import failure still produces a JSON line.
    import duckdb

    started = time.monotonic()
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            "CREATE TABLE readings ("
            "sample_id VARCHAR, item VARCHAR, value VARCHAR)"
        )
        if rows:
            con.executemany(
                "INSERT INTO readings VALUES (?, ?, ?)",
                rows,
            )
        con.execute(sql).fetchall()
    except Exception as exc:  # noqa: BLE001 - every DuckDB error is a trial result
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "first_line": str(exc).splitlines()[0] if str(exc) else "",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }
    else:
        result = {
            "status": "success",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }
    finally:
        con.close()

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        # Normal interpreter exit (including exit code 0 from main()).
        raise
    except BaseException:  # pragma: no cover - defensive last resort
        traceback.print_exc()
        sys.stdout.write(
            json.dumps(
                {
                    "status": "runner_crash",
                    "error_type": "RunnerCrash",
                    "first_line": "trial runner crashed before producing a result",
                    "elapsed_ms": 0.0,
                }
            )
            + "\n"
        )
        sys.stdout.flush()
        raise SystemExit(2)
