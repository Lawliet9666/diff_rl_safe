"""One-time W&B report refresh after the November 1 Matilda reruns end.

--check reads current states. --refresh-now captures current results, including
active runs. --wait polls every 15 minutes and refreshes once after training.
The local Mac must stay awake and online for timely checks. Completion is based
on the tracked W&B runs, not a Slurm query. A short finished run must remain
terminal across two checks before its missing successors are reported.
The original HTML/CSV are backed up before replacement. Authentication or
network failures are logged and stop monitoring after three consecutive errors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/wandb_summary_20261101'
PROJECT = 'xinywa_umich/diff_rl_safe'
ROBOTS = ('single_integrator', 'unicycle', 'double_integrator', 'humanoid', 'quadrotor')
PLAN = {
    404092: ('cbf_rl', ROBOTS),
    404093: ('barriernet', ROBOTS[2:]),
    404094: ('barriernet_sim', ROBOTS[2:]),
    404095: ('ppo_filter', ROBOTS[1:]),
}
COHORT = {404073, 404074, 404075, 404076, *PLAN}
TERMINAL = {'finished', 'failed', 'crashed', 'killed'}
NAME = re.compile(r'^matilda_(?:left_\d{8}_)?(\d+)_(ppo_filter|barriernet_sim|cbf_rl|barriernet|ppo)_'
                  r'(single_integrator|double_integrator|unicycle|humanoid|quadrotor)_')
DATA = re.compile(r'(<script id="run-data" type="application/json">)(.*?)(</script>)', re.S)
SUMMARY_FIELDS = {'global_step', '_runtime', '_timestamp', 'charts/sps', 'train/success_rate',
                  'train/qp_failure_rate', 'charts/eval_success_rate', 'charts/eval_return_mean',
                  'charts/eval_collision_rate', 'charts/eval_timeout_rate', 'charts/eval_qp_failure_rate'}
LOG = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    return value


def read_report(folder):
    html = (folder / 'index.html').read_text()
    matches = list(DATA.finditer(html))
    if len(matches) != 1:
        raise ValueError('Expected exactly one embedded run-data script; report left unchanged')
    return html, json.loads(matches[0][2])


def atomic_write(path, content):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def metadata(run, previous):
    config = run.config
    match = NAME.search(run.name)
    old = previous.get(run.id, {})
    summary = dict(run.summary)
    return finite(dict(
        id=run.id, name=run.name, state=run.state, created_at=run.created_at,
        url='https://forge.coreweave.com/wandb/' + PROJECT + '/runs/' + run.id,
        job_id=int(match[1]) if match else old.get('job_id'),
        method=match[2] if match else old.get('method', 'other'),
        robot=match[3] if match else old.get('robot'),
        config={key: config.get(key) for key in ('seed', 'robot', 'trainer', 'model', 'env')},
        summary={key: value for key, value in summary.items() if key in SUMMARY_FIELDS},
    ))


def select_latest(records):
    chosen = {}
    for record in records:
        record['latest_batch'] = False
        if record['job_id'] in COHORT:
            key = (record['robot'], record['method'])
            if key not in chosen or (record['created_at'], record['id']) > (
                    chosen[key]['created_at'], chosen[key]['id']):
                chosen[key] = record
    for record in chosen.values():
        record['latest_batch'] = True
    return chosen


def batch_status(records, previous_signatures):
    """Use W&B terminal states; require two observations for an early finish."""
    states, signatures = {}, {}
    for job, (method, robots) in PLAN.items():
        candidates = [r for r in records if r['job_id'] == job]
        if any(r['method'] != method or r['robot'] not in robots for r in candidates):
            raise ValueError('Unexpected method/robot for tracked job %s' % job)
        latest = {}
        for record in sorted(candidates, key=lambda r: (r['created_at'], r['id'])):
            latest[record['robot']] = record
        missing = [robot for robot in robots if robot not in latest]
        active = [robot for robot, r in latest.items() if r['state'] not in TERMINAL]
        early = []
        for robot, r in latest.items():
            target = (r['config'].get('trainer') or {}).get('total_timesteps')
            step = r['summary'].get('global_step')
            if r['state'] in TERMINAL and not (number(step) and number(target) and step >= target):
                early.append(robot)
        status = 'waiting_for_runs' if not latest else 'running'
        if latest and not active:
            if not missing:
                status = 'ended'
            elif any(r['state'] in {'failed', 'crashed', 'killed'} for r in latest.values()):
                status = 'ended_early'
            elif early:
                signature = json.dumps(sorted((r['id'], r['state'], r['summary']) for r in latest.values()), sort_keys=True)
                signatures[str(job)] = signature
                status = 'ended_early' if previous_signatures.get(str(job)) == signature else 'confirming_early_finish'
        states[str(job)] = dict(status=status, missing=missing, active=active, incomplete=early)
    return states, signatures


def series(rows, key):
    values = {int(row['global_step']): float(row[key]) for row in rows
              if number(row.get('global_step')) and number(row.get(key))}
    return [[step, value] for step, value in sorted(values.items())]


def collect_curves(run, *, allow_running=False):
    if run.state not in TERMINAL and not allow_running:
        raise RuntimeError('Run %s became active again; refusing a final refresh' % run.id)
    # The SDK pages over _step ranges. This trainer sets _step=global_step,
    # advancing by a rollout batch, so span about 1000 updates per request.
    batch = run.config['trainer']['timesteps_per_batch']
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ValueError('Run %s has an invalid rollout batch size' % run.id)
    window = 1000 * batch
    # Fetch a fresh history once, then reuse that cache for the independent
    # sparse training/evaluation queries in this same collection.
    train = list(run.scan_history(keys=['global_step', 'train/success_rate'], page_size=window, use_cache=False))
    speed = list(run.scan_history(keys=['global_step', 'charts/sps'], page_size=window, use_cache=True))
    evaluation = list(run.scan_history(keys=['global_step', 'charts/eval_success_rate',
                                              'charts/eval_return_mean'], page_size=window, use_cache=True))
    curves = dict(train=series(train, 'train/success_rate'), sps=series(speed, 'charts/sps'),
                  evaluation=series(evaluation, 'charts/eval_success_rate'),
                  returns=series(evaluation, 'charts/eval_return_mean'))
    LOG.info('Collected %s history points: %s', run.id, {key: len(value) for key, value in curves.items()})
    return curves


def dashboard_record(record, curves, old, labels):
    s, cfg = record['summary'], record['config']
    trainer, robot, model, env = (cfg.get(k) or {} for k in ('trainer', 'robot', 'model', 'env'))
    step, target = s.get('global_step'), trainer.get('total_timesteps')
    complete = number(step) and number(target) and target > 0 and step >= target
    state = record['state']
    status = ('Running' if state == 'running' else 'Completed' if state == 'finished' and complete
              else 'Finished early' if state == 'finished' else state.capitalize())
    notes = []
    if state in TERMINAL and not complete:
        notes.append('No training metrics logged' if step is None else 'Stopped before the configured training budget')
    if record['method'] == 'cbf_rl':
        notes.append('Training uses a safety filter; evaluation removes it')
    tail = [value for _, value in curves.get('train', [])[-100:]]
    best_return = max(curves.get('returns', []), key=lambda p: p[1], default=None)
    return dict(
        **{k: record[k] for k in ('id', 'name', 'url', 'created_at', 'job_id', 'method', 'robot', 'latest_batch')},
        method_label=labels['methods'][record['method']], robot_label=labels['robots'].get(record['robot'], 'Unknown'),
        history_sampled=False if record['latest_batch'] else old.get('history_sampled', False),
        wandb_state=state, status=status, steps=step, target=target,
        progress=min(step / target, 1) if number(step) and number(target) and target > 0 else None,
        train_tail100=sum(tail) / len(tail) if tail else None, train_tail_count=len(tail),
        last_train=s.get('train/success_rate'), last_eval=s.get('charts/eval_success_rate'),
        best_eval=max((y for _, y in curves.get('evaluation', [])), default=None),
        last_return=s.get('charts/eval_return_mean'), best_return=best_return[1] if best_return else None,
        best_return_step=best_return[0] if best_return else None, sps=s.get('charts/sps'), runtime=s.get('_runtime'),
        eval_collision=s.get('charts/eval_collision_rate'), eval_timeout=s.get('charts/eval_timeout_rate'),
        eval_qp_failure=s.get('charts/eval_qp_failure_rate'), train_qp_failure=s.get('train/qp_failure_rate'), notes=notes,
        settings=dict(num_envs=trainer.get('num_envs'), batch=trainer.get('timesteps_per_batch'),
                      minibatches=trainer.get('num_minibatches'), epochs=trainer.get('n_updates_per_iteration'),
                      eval_episodes=trainer.get('eval_episodes'), v_max=robot.get('v_max'), seed=cfg.get('seed'),
                      model=model.get('type'), eval_interval=trainer.get('eval_interval'), dt=env.get('dt'),
                      simulation_loss=(trainer.get('reward_loss') or {}).get('weight'),
                      simulation_horizon=(trainer.get('reward_loss') or {}).get('horizon'),
                      cbf_rl=trainer.get('cbf_rl'), cbf_reward=robot.get('reward', {}).get('cbf_rl')),
        curves=curves,
    )


def update_narrative(html):
    """Keep the original dashboard's prose tied to its embedded run data."""
    html = html.replace('latest training batch, grouped', 'latest available training results, grouped')
    html = html.replace('Latest batch only', 'Selected latest runs').replace(
        'Latest batch · inspect before reuse', 'Selected latest runs · inspect before reuse')
    html = html.replace(' (sampled for the active run)', '')
    cbf_highlight = "`CBF-RL latest evaluation success: ${['single_integrator','unicycle','double_integrator','humanoid','quadrotor'].map(b=>{const r=get(b,'cbf_rl');return esc(robots[b])+': '+(r?'<strong>'+pct(r.last_eval)+'</strong>'+(r.status==='Running'?' (running)':r.job_id===404074?' (earlier run)':''):'not recorded');}).join('; ')}.`,"
    html = re.sub(r'^`CBF-RL single and double integrator finish with .*?`,$',
                  lambda m: cbf_highlight, html, count=1, flags=re.M)
    di_highlight = "`Double integrator: latest evaluation success is <strong>${pct(get('double_integrator','barriernet')?.last_eval)}</strong> for BarrierNet and <strong>${pct(get('double_integrator','barriernet_sim')?.last_eval)}</strong> for BarrierNet + simulation loss.`,"
    if 'Double integrator: latest evaluation success' not in html:
        html = html.replace('`PPO double integrator ends at', di_highlight + '\n`PPO double integrator ends at', 1)
    hide_empty = "$('findings').hidden = incomplete.length === 0;"
    if hide_empty not in html:
        html = html.replace("$('findings').innerHTML=", hide_empty + "\n$('findings').innerHTML=", 1)
    return html


def refresh(folder, api_runs, records, job_states, *, allow_running=False):
    html, old_data = read_report(folder)
    old = {r['id']: r for r in old_data['runs']}
    chosen = select_latest(records)
    live = {r.id: r for r in api_runs}
    selected = list(chosen.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        histories = dict(zip((r['id'] for r in selected), pool.map(
            lambda r: collect_curves(live[r['id']], allow_running=allow_running), selected)))
    # A live run can log more data during collection. Keep its curves consistent
    # with the step captured in this snapshot's summary and progress table.
    for record in selected:
        step = record['summary'].get('global_step')
        if number(step):
            histories[record['id']] = {key: [point for point in points if point[0] <= step]
                                       for key, points in histories[record['id']].items()}
    runs = [dashboard_record(r, histories.get(r['id'], old.get(r['id'], {}).get('curves', {})),
                             old.get(r['id'], {}), old_data) for r in records]
    # Retain historical snapshots even if an older run is no longer returned by W&B.
    present = {r['id'] for r in runs}
    runs.extend(dict(r, latest_batch=False) for r in old_data['runs'] if r['id'] not in present)
    runs.sort(key=lambda r: (not r['latest_batch'], r['robot_label'], r['method_label'], r['created_at'] or ''))
    captured = datetime.now(timezone.utc)
    data = dict(old_data, captured=captured.astimezone(ZoneInfo('America/Detroit')).strftime('%B %d, %Y · %I:%M %p %Z'),
                runs=runs, refresh=dict(jobs=list(PLAN), job_states=job_states, source='W&B Public API',
                                       kind='live_snapshot' if any(r['state'] not in TERMINAL for r in selected)
                                       else 'final_snapshot'))
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    job_ids = ', '.join(str(j) for j in sorted({r['job_id'] for r in selected}))
    issues = ['job %s: missing %s; incomplete %s' % (
        job, ', '.join(state['missing']) or 'none', ', '.join(state['incomplete']) or 'none')
        for job, state in job_states.items()
        if state['status'] in {'ended', 'ended_early'} and (state['missing'] or state['incomplete'])]
    overview = ('<section id="overview"><h2>Latest results at a glance</h2><p class="muted">Jobs ' + job_ids +
                ' · each cell selects the latest run for that robot/method. Earlier runs remain in the table. '
                'Running runs show progress at capture time. An empty cell means no run was recorded in the tracked batches. ' +
                ('Incomplete reruns — ' + '; '.join(issues) + '. ' if issues else '') + '</p>')
    html, count = re.subn(r'<section id="overview"><h2>.*?</h2><p class="muted">.*?</p>', overview, html, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Report overview changed; refusing to overwrite it')
    html = re.sub(r'<option value="latest">[^<]*</option>', '<option value="latest">Latest runs</option>', html, count=1)
    html = html.replace('Latest-batch curves', 'Latest-run curves').replace('in the latest batch', 'for the selected latest runs')
    html = update_narrative(html)
    html = re.sub(r'(<section id="curves">.*?<p class="muted">).*?(</p>)',
                  lambda m: m[1] + 'Charts select the latest run for each robot/method, with histories collected through the captured step. '
                  'Robot and method filters also apply. Missing results are not treated as zero.' + m[2], html, count=1, flags=re.S)
    html = DATA.sub(lambda m: m[1] + encoded + m[3], html)
    # Build both outputs before replacing the existing snapshot; preserve a dated backup.
    columns = next(csv.reader(io.StringIO((folder / 'runs.csv').read_text())))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(runs)
    backup = folder / ('before_refresh_' + captured.strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir()
    for name in ('index.html', 'runs.csv'):
        shutil.copy2(folder / name, backup / name)
    atomic_write(folder / 'dashboard-data.json', json.dumps(data, allow_nan=False))
    atomic_write(folder / 'source.json', json.dumps(dict(captured_at=captured.isoformat(), runs=records), allow_nan=False))
    atomic_write(folder / 'runs.csv', output.getvalue())
    atomic_write(folder / 'index.html', html)
    return dict(runs=len(runs), latest=len(selected), backup=str(backup))


def snapshot(folder):
    import wandb
    # A fresh client avoids reusing cached run states across polling intervals.
    api = wandb.Api(timeout=30, overrides={'base_url': 'https://api.forge.coreweave.com'})
    api_runs = list(api.runs(PROJECT, per_page=100))
    _, data = read_report(folder)
    previous = {r['id']: r for r in data['runs']}
    return api_runs, [metadata(r, previous) for r in api_runs]


def watch(folder, poll_seconds, timeout_hours):
    with (folder / 'refresh.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An automatic refresh watcher is already running') from None
        started, deadline = now(), time.time() + timeout_hours * 3600
        signatures, failures = {}, 0
        state = dict(pid=os.getpid(), started_at=started, jobs=list(PLAN), poll_seconds=poll_seconds,
                     report=str(folder / 'index.html'), status='starting')
        state_path = folder / 'refresh_state.json'
        try:
            while True:
                try:
                    api_runs, records = snapshot(folder)
                    jobs, signatures = batch_status(records, signatures)
                    failures = 0
                except Exception as error:
                    failures += 1
                    LOG.exception('W&B check failed (%s/3); existing report is unchanged', failures)
                    state.update(status='retrying', checked_at=now(), error=str(error), consecutive_failures=failures)
                    atomic_write(state_path, json.dumps(state, indent=2))
                    if failures >= 3 or time.time() >= deadline:
                        raise
                    time.sleep(poll_seconds)
                    continue
                state.update(status='waiting', checked_at=now(), job_states=jobs, error=None)
                atomic_write(state_path, json.dumps(state, indent=2))
                LOG.info('Job states: %s', json.dumps(jobs, sort_keys=True))
                if all(j['status'] in {'ended', 'ended_early'} for j in jobs.values()):
                    details = refresh(folder, api_runs, records, jobs)
                    partial = any(j['missing'] or j['incomplete'] for j in jobs.values())
                    state.update(status='refreshed_partial' if partial else 'refreshed', finished_at=now(), **details)
                    atomic_write(state_path, json.dumps(state, indent=2))
                    LOG.info('Report refreshed: %s', json.dumps(details))
                    return
                if time.time() >= deadline:
                    raise TimeoutError('Tracked runs did not reach terminal states before the deadline; report left unchanged')
                time.sleep(poll_seconds)
        except BaseException as error:
            state.update(status='error', finished_at=now(), error=str(error))
            atomic_write(state_path, json.dumps(state, indent=2))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--wait', action='store_true', help='Refresh once after the tracked runs end, then exit')
    mode.add_argument('--check', action='store_true', help='Read current run states without modifying the report')
    mode.add_argument('--refresh-now', action='store_true', help='Refresh current results and curves, including active runs')
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    parser.add_argument('--poll-seconds', type=int, default=900)
    parser.add_argument('--timeout-hours', type=float, default=168)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        parser.error('Polling interval and timeout must be positive and finite')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.check:
        _, records = snapshot(args.report_dir)
        print(json.dumps(batch_status(records, {})[0], indent=2))
    elif args.refresh_now:
        with (args.report_dir / 'refresh.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('The automatic watcher is active; coordinate it before a manual refresh') from None
            api_runs, records = snapshot(args.report_dir)
            states, _ = batch_status(records, {})
            print(json.dumps(refresh(args.report_dir, api_runs, records, states, allow_running=True)))
    else:
        watch(args.report_dir, args.poll_seconds, args.timeout_hours)


if __name__ == '__main__':
    main()
