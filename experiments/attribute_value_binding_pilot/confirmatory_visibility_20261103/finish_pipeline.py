"""Durable, single-worker continuation and downstream artifacts; no design changes.

Run with the GTE environment. --wait-pid attaches to an already running Harrier
worker; no second GPU process is launched while that process remains alive.
Windows automatic sleep is inhibited only for this process's lifetime. Manual
sleep/shutdown is not overridden. The report still requires visual review.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GTE = ROOT / 'phase2/.venv/Scripts/python.exe'
HARRIER = ROOT / 'revision_recent_models_20261016/.venv/Scripts/python.exe'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def awake():
    ticks = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
        raise ctypes.WinError()
    return ticks.value / 10_000_000


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-pid', type=int)
    args = ap.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logpath = HERE / f'qa/continuation_{stamp}.log'
    statuspath = HERE / 'qa/continuation_status.json'
    start_m, start_a, start_w = time.monotonic(), awake(), time.time()
    state = dict(started_utc=utc(), status='running', stage='starting', log=logpath.name,
                 pid=os.getpid(), steps=[], visual_review='pending',
                 note='Awake includes CPU/I/O/GPU wait, not GPU kernel time. Manual sleep or shutdown can still interrupt.')

    def status(stage, **extra):
        state.update(stage=stage, updated_utc=utc(), **extra)
        state.update(monotonic_elapsed_seconds=time.monotonic()-start_m,
                     awake_elapsed_seconds=awake()-start_a,
                     wall_elapsed_seconds=time.time()-start_w)
        write(statuspath, state)

    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    kernel = ctypes.windll.kernel32
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: temporary, no display wake request.
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError()
    try:
        with logpath.open('a', encoding='utf8', buffering=1) as log:
            def step(label, command):
                status(label)
                log.write(f'\n{utc()} START {label}\n')
                sm, sa = time.monotonic(), awake()
                result = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                state['steps'].append(dict(stage=label, exit_code=result.returncode,
                                          monotonic_seconds=time.monotonic()-sm,
                                          awake_seconds=awake()-sa))
                status(label)
                if result.returncode:
                    raise RuntimeError(f'{label} exited {result.returncode}; see {logpath.name}')

            if args.wait_pid:
                # A process HANDLE avoids accidentally waiting on a reused PID.
                kernel.OpenProcess.restype = ctypes.c_void_p
                handle = kernel.OpenProcess(0x00100000, False, args.wait_pid)
                if handle:
                    try:
                        status('waiting_for_existing_harrier', worker_pid=args.wait_pid)
                        while True:
                            code = kernel.WaitForSingleObject(ctypes.c_void_p(handle), 30000)
                            if code == 0:
                                break
                            if code != 258:
                                raise ctypes.WinError()
                            status('waiting_for_existing_harrier')
                    finally:
                        kernel.CloseHandle(ctypes.c_void_p(handle))
            if not (HERE/'qa/formal/harrier_complete.json').exists():
                step('harrier_formal', [HARRIER, HERE/'run.py', '--stage', 'formal', '--model', 'harrier'])
            step('analysis', [GTE, HERE/'analyze.py'])
            marker = HERE/'qa/pdf_authoring_marker.json'
            if not marker.exists():
                marker_script = Path('C:/Users/ycw/.codex/plugins/cache/openai-primary-runtime/pdf/26.1103.10131/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
                step('pdf_authoring_marker', ['node', marker_script, '--operation-kind', 'create', '--expected-output-count', '1', '--output-format', 'pdf'])
                write(marker, dict(success=True, completed_utc=utc(), script=str(marker_script)))
            step('paper_artifacts', [GTE, HERE/'paper_artifacts.py'])
            step('report', [GTE, HERE/'report.py'])
            step('layout_render', [GTE, HERE/'layout_preview.py'])
            step('verification', [GTE, HERE/'verify.py'])
            status('awaiting_visual_review', status='numerical_pipeline_complete')
    except BaseException as exc:
        status(state['stage'], status='failed', error=repr(exc))
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
