"""Subprocess management for individual trials.

Each trial is an independent process (and process group) with its own
in-memory DuckDB instance.  The parent enforces the hard 2 second limit and
cancels promptly.

Input transfer
--------------
The trial payload (all rows + SQL) can be much larger than the OS pipe
buffer.  It is therefore written **incrementally on a non-blocking stdin
descriptor**, interleaved with stdout/stderr draining and cancellation /
deadline checks.  A child that stops accepting input can never block the
parent: the 2 second budget is measured from the moment the process is
started, so a stalled transfer still times out or cancels on schedule.

Termination and finalization
----------------------------
On timeout/cancellation the parent (1) stops writing and closes the input
pipe, (2) signals the process group (primary path), (3) waits until the
process has actually exited and been reaped while draining its pipes, and
only then reports ``timeout``/``canceled``.  As a backstop for environments
that deny the parent permission to signal a live child, the runner arms its
own wall-clock alarm for the same budget, so the trial process always exits
and is reaped.

Two-tier termination
--------------------
1. **Signal-based (primary).**  The trial runs in a fresh process group; on
   cancellation/timeout the parent sends SIGKILL to the whole group.  This is
   immediate and is how the application behaves inside the Docker container.
2. **Pre-launch cancel check + runner self-alarm (restricted sandboxes).**
   Some hardened environments deny the parent permission to signal a live
   child.  The parent re-checks the cancel event before launching every trial
   and the runner terminates itself when the trial budget elapses.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Hard per-trial wall-clock budget required by the specification.  It is
# measured from process start, including input transfer and row loading.
TRIAL_TIMEOUT_SECONDS = 2.0

# Extra time the runner allows itself before its self-alarm fires.  The parent
# normally enforces the limit first; the alarm is only a backstop that keeps
# termination possible where parent signals are denied.  Keep in sync with
# ``RUNNER_DEADLINE_GRACE_SECONDS`` in runner.py (asserted by the tests).
RUNNER_ALARM_GRACE_SECONDS = 0.25

# Exit code the runner uses when its own deadline alarm fires.  Must match
# ``EXIT_DEADLINE`` in runner.py.
RUNNER_DEADLINE_EXIT_CODE = 124

_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
_CHUNK = 64 * 1024
_POLL_SECONDS = 0.02


@dataclass
class TrialResult:
    """Outcome of one trial.

    status is one of:
      ``success``       - the query ran without error
      ``error``         - the query raised (see error_type / first_line)
      ``timeout``       - the trial exceeded the 2 second limit
      ``canceled``      - the job was canceled while the trial was running
      ``runner_crash``  - the runner process itself died unexpectedly

    The ``input_*`` fields describe how much of the payload reached the child.
    They make it possible to verify that a timeout/cancel was honoured even
    while input transfer was still in progress.  ``returncode`` is set once
    the process has been reaped; ``None`` means reaping could not be
    confirmed.
    """

    status: str
    elapsed_ms: float = 0.0
    error_type: Optional[str] = None
    first_line: Optional[str] = None
    input_bytes_total: int = 0
    input_bytes_sent: int = 0
    input_complete: bool = False
    returncode: Optional[int] = None
    stop_reason: Optional[str] = None

    @property
    def reproduces(self) -> bool:
        """Only an identical exception class + first-line message counts."""
        return self.status == "error" and self.error_type is not None


def _kill_group(proc: subprocess.Popen) -> bool:
    """Kill the trial process group. Returns True if a signal was delivered."""
    if proc.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
            return True
        except (ProcessLookupError, PermissionError):
            return False


def _set_nonblocking(fd: int) -> None:
    try:
        os.set_blocking(fd, False)
    except (OSError, ValueError):
        pass


def _read_available(fd: int, sink: bytearray) -> bool:
    """Drain a non-blocking descriptor. Returns True at EOF/error."""
    while True:
        try:
            chunk = os.read(fd, _CHUNK)
        except (BlockingIOError, InterruptedError):
            return False
        except (OSError, ValueError):
            return True
        if not chunk:
            return True
        sink.extend(chunk)


def run_trial(
    rows: list[list[str]],
    sql: str,
    canceled_event,
    timeout_seconds: float = TRIAL_TIMEOUT_SECONDS,
    input_stall_seconds: float = 0.0,
) -> TrialResult:
    """Run one trial in an isolated subprocess.

    ``canceled_event`` is a :class:`threading.Event`-like object. If it is
    already set, no subprocess is launched; if it becomes set while a trial is
    running (including while its input is still being transferred) the trial
    is stopped and the parent waits for the process to be reaped before
    returning.

    ``input_stall_seconds`` is a demo/test hook: the *runner* pauses for that
    long before reading stdin, simulating a trial process that has stopped
    accepting input.  It has no effect in normal operation (default 0).
    """
    # Do not even launch a new trial once cancellation was requested: this is
    # the boundary that makes cancellation immediate between trials even when
    # the sandbox forbids signaling an already-running process.
    if canceled_event is not None and canceled_event.is_set():
        return TrialResult(status="canceled", stop_reason="cancel",
                           input_complete=False)

    # Serialization happens *before* the process starts and is deliberately
    # not part of the 2 second trial budget.
    payload_text = json.dumps({"rows": rows, "sql": sql})
    payload = payload_text.encode("utf-8")

    env = os.environ.copy()
    env["SAMPLE_MINIMIZER_TRIAL_TIMEOUT_SECONDS"] = repr(float(timeout_seconds))
    if input_stall_seconds and input_stall_seconds > 0:
        env["SAMPLE_MINIMIZER_RUNNER_INPUT_STALL_SECONDS"] = repr(
            float(input_stall_seconds)
        )

    # The budget starts when the process is started, NOT after the payload has
    # been handed to the child.
    started = time.monotonic()
    deadline = started + timeout_seconds
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", _RUNNER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # new process group -> clean group kill
            bufsize=0,
            env=env,
        )
    except OSError as exc:
        return TrialResult(status="runner_crash", first_line=str(exc),
                           input_bytes_total=len(payload),
                           input_complete=False)

    assert proc.stdin is not None and proc.stdout is not None
    assert proc.stderr is not None
    stdin_fd = proc.stdin.fileno()
    stdout_fd = proc.stdout.fileno()
    stderr_fd = proc.stderr.fileno()
    for fd in (stdin_fd, stdout_fd, stderr_fd):
        _set_nonblocking(fd)

    stdout_bytes = bytearray()
    stderr_bytes = bytearray()
    stdin_open = True
    stdout_open = True
    stderr_open = True
    write_pos = 0
    stop_reason: Optional[str] = None

    def close_stdin() -> None:
        nonlocal stdin_open
        if not stdin_open:
            return
        stdin_open = False
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    def drain_output() -> None:
        nonlocal stdout_open, stderr_open
        if stdout_open:
            stdout_open = not _read_available(stdout_fd, stdout_bytes)
        if stderr_open:
            stderr_open = not _read_available(stderr_fd, stderr_bytes)

    # ------------------------------------------------------------------
    # Event loop: input is delivered incrementally; deadline/cancel are
    # checked on every iteration, so a child that stops reading cannot stall
    # the job.
    # ------------------------------------------------------------------
    while True:
        now = time.monotonic()
        if canceled_event is not None and canceled_event.is_set():
            stop_reason = "cancel"
            break
        if now >= deadline:
            stop_reason = "timeout"
            break

        # --- deliver a chunk of input, if any is still pending ---
        if stdin_open:
            if write_pos >= len(payload):
                close_stdin()
            else:
                try:
                    written = os.write(
                        stdin_fd, payload[write_pos:write_pos + _CHUNK]
                    )
                    write_pos += written
                    if write_pos >= len(payload):
                        # EOF lets the runner finish json.load() and execute.
                        close_stdin()
                except (BlockingIOError, InterruptedError):
                    pass  # pipe full: child is not accepting input right now
                except (BrokenPipeError, OSError):
                    # Child died or closed its read end; nothing left to send.
                    close_stdin()

        # --- wait for input capacity or child output ---
        readable = []
        if stdout_open:
            readable.append(stdout_fd)
        if stderr_open:
            readable.append(stderr_fd)
        writable = [stdin_fd] if stdin_open else []
        wait = min(_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
        try:
            ready_r, _, _ = select.select(readable, writable, [], wait)
        except (OSError, ValueError):
            ready_r = []
            time.sleep(wait)
        if stdout_fd in ready_r:
            stdout_open = not _read_available(stdout_fd, stdout_bytes)
        if stderr_fd in ready_r:
            stderr_open = not _read_available(stderr_fd, stderr_bytes)

        if proc.poll() is not None:
            # Child finished on its own (normal fast path).
            break

    drain_output()
    close_stdin()

    input_complete = write_pos >= len(payload)
    input_bytes_sent = write_pos

    # ------------------------------------------------------------------
    # Stop and reap.  A terminal state is only reported after this block:
    # no writer remains and the process has exited (returncode is set).
    # ------------------------------------------------------------------
    if stop_reason is not None:
        # Primary path: kill the whole process group (immediate in Docker).
        _kill_group(proc)

    # Backstop for signal-denied sandboxes: the runner arms its own alarm.
    reap_grace = max(5.0, timeout_seconds + 5.0)
    reap_deadline = time.monotonic() + reap_grace
    drain_output()
    while proc.poll() is None:
        if time.monotonic() >= reap_deadline:
            # Last resort: try again, then give the OS a brief moment.
            _kill_group(proc)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            break
        time.sleep(0.01)
        drain_output()

    final_rc = proc.poll()
    drain_output()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    stdout = stdout_bytes.decode("utf-8", "replace")
    stderr = stderr_bytes.decode("utf-8", "replace")
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)

    common = dict(
        elapsed_ms=elapsed_ms,
        input_bytes_total=len(payload),
        input_bytes_sent=input_bytes_sent,
        input_complete=input_complete,
        returncode=final_rc,
    )

    if canceled_event is not None and canceled_event.is_set():
        return TrialResult(status="canceled", stop_reason="cancel", **common)
    if stop_reason == "timeout":
        return TrialResult(status="timeout", stop_reason="timeout", **common)
    if stop_reason == "cancel":
        return TrialResult(status="canceled", stop_reason="cancel", **common)
    if final_rc == RUNNER_DEADLINE_EXIT_CODE:
        # The runner's own backstop fired (parent signals were unavailable):
        # this is still a per-trial timeout, never a reproduction.
        return TrialResult(status="timeout", stop_reason="runner_alarm", **common)
    if final_rc != 0:
        detail = (stdout.strip().splitlines() or []) + (
            stderr.strip().splitlines() or []
        )
        return TrialResult(
            status="runner_crash",
            error_type="RunnerCrash",
            first_line=(detail[-1] if detail else "runner exited with error")[:500],
            stop_reason="runner_exit",
            **common,
        )

    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return TrialResult(status="runner_crash",
                           first_line="unparseable runner output",
                           stop_reason="runner_exit", **common)

    status = data.get("status", "runner_crash")
    if status == "success":
        return TrialResult(status="success",
                           elapsed_ms=float(data.get("elapsed_ms", elapsed_ms)),
                           **{k: v for k, v in common.items()
                              if k != "elapsed_ms"})
    if status == "error":
        return TrialResult(
            status="error",
            error_type=data.get("error_type", "Unknown"),
            first_line=data.get("first_line", ""),
            elapsed_ms=float(data.get("elapsed_ms", elapsed_ms)),
            **{k: v for k, v in common.items() if k != "elapsed_ms"},
        )
    return TrialResult(
        status="runner_crash",
        error_type=data.get("error_type", "RunnerCrash"),
        first_line=data.get("first_line", "runner crash"),
        stop_reason="runner_exit",
        **common,
    )
