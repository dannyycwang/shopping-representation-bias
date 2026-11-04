"""Account for saved work in an interrupted run without inventing its end time."""
from core import *


def main():
    prior_path = HERE/'qa/attempts/harrier_formal_1793711056702914900.json'
    prior = read(prior_path)
    next_attempts = sorted(
        [read(p) for p in (HERE/'qa/attempts').glob('harrier_formal_*.json')
         if read(p)['started_utc'] > prior['started_utc']],
        key=lambda x: x['started_utc'])
    assert next_attempts
    end = next_attempts[0]['started_utc']
    rows = []
    for p in (HERE/'cache/harrier').glob('*.json'):
        m = read(p)
        t = m['timing']
        if prior['started_utc'] <= t['started_utc'] < end:
            assert sha(p.with_suffix('.npy')) == m['sha256']
            rows.append(dict(checkpoint=p.name, role=m['descriptor']['role'],
                             context=canonical(m['descriptor']['context']), **t))
    assert rows
    csv('qa/interrupted_20261103_checkpoint_timings.csv', rows)
    dump('qa/interrupted_20261103_final_attempt.json', dict(
        audited_utc=now(), prior_attempt=prior_path.name, prior_attempt_sha256=sha(prior_path),
        next_attempt_started_utc=end, prior_status=prior['status'],
        termination_time_and_cause='not observed; process was absent on 2026-11-04 status check',
        forward_checkpoints=len(rows),
        awake_forward_seconds=sum(r['awake_elapsed_seconds'] for r in rows),
        monotonic_forward_seconds=sum(r['monotonic_elapsed_seconds'] for r in rows),
        possible_suspended_forward_seconds=sum(r['possible_suspended_seconds'] for r in rows),
        last_checkpoint_finished_utc=max(r['finished_utc'] for r in rows),
        complete_end_to_end_timing_available=False,
        timing_interpretation='Completed-checkpoint lower bound only. Excludes uncaptured setup, I/O, replay verification and unfinished work; not pure GPU time. Do not count the gap before restart as active execution.'))
    print('INTERRUPTED RUN AUDITED', len(rows), 'saved forwards', flush=True)


if __name__ == '__main__':
    main()
