import json
import subprocess

import numpy as np
import pytest
import wandb

from conftest import METHODS, make_config
from scripts.run_ppo_base import build_trainer
from scripts.eval import Evaluator

@pytest.mark.parametrize('method',METHODS)
def test_two_update_training_and_logging(method,tmp_path,monkeypatch):
    calls=[]
    cfg = make_config(method,extra=[f'save_dir={tmp_path}', 'run_name=smoke',
                                   'trainer.num_envs=8', 'trainer.timesteps_per_batch=64',
                                   'trainer.total_timesteps=128', 'trainer.n_updates_per_iteration=1',
                                   'trainer.num_minibatches=2', 'trainer.finite_check_interval=1',
                                   'trainer.eval_interval=1', 'trainer.eval_episodes=1', 'wandb_interval=1'])
    trainer = build_trainer(cfg)
    # wandb.init binds the module's log function; capture it after initialization.
    monkeypatch.setattr(wandb,'log',lambda payload,step: calls.append((payload,step)))
    trainer.train()
    records = [json.loads(line) for line in (trainer.save_dir/'metrics.jsonl').read_text().splitlines()]
    assert [record['global_step'] for record in records] == [64,128]
    assert len(calls) == 2
    for (payload,step),record in zip(calls,records):
        assert step == record['global_step']
        assert all(record[key] == value for key,value in payload.items())
        assert all(np.isfinite(value) for value in record.values())
    assert 'charts/eval_collision_rate' in calls[-1][0]
    if method.startswith('barriernet'):
        assert 'diagnostics/alpha_active_min' in calls[-1][0]
    if method.startswith('diff_cvar'):
        assert 'diagnostics/beta_active_min' in calls[-1][0]
    checkpoint = sorted(trainer.save_dir.glob('ckpt_*.pkl'))[-1]
    if method == 'barriernet_gnn':
        plain = Evaluator(trainer.save_dir,seeds=[100],episodes=2).eval_all_ckpts(checkpoint)
        rendered = Evaluator(trainer.save_dir,visualize=True,seeds=[100],episodes=2).eval_all_ckpts(checkpoint)
        for key,value in plain[checkpoint.name].items():
            if value is not None:
                np.testing.assert_allclose(rendered[checkpoint.name][key],value,rtol=1e-5,atol=1e-6)
        videos = list(trainer.save_dir.glob('visualize_*/*.mp4'))
        assert len(videos) == 2
        import imageio_ffmpeg
        decoded = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-i', str(videos[0]),
             '-vf', 'scale=32:32', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, check=True)
        assert len(decoded.stdout) % (32*32*3) == 0
        assert len(decoded.stdout) >= 2*32*32*3
        saved = json.loads((trainer.save_dir/'eval_results.json').read_text())
        assert saved[checkpoint.name] == rendered[checkpoint.name]
