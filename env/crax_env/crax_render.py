"""Render recorded MJX states with MuJoCo."""
import ctypes
import os
from pathlib import Path
import sys

import numpy as np


class RecordedPointRenderer:
    def __init__(self, env):
        if os.environ.get("MUJOCO_GL") == "osmesa":
            ctypes.CDLL(str(Path(sys.prefix) / "lib/libOSMesa.so"), mode=ctypes.RTLD_GLOBAL)
        import mujoco
        self.mj = mujoco
        self.env = env
        self.model = env.model
        size = env.render_size
        self.model.vis.global_.offwidth = size
        self.model.vis.global_.offheight = size
        floor = self.model.geom("floor").id
        self.model.geom_matid[floor] = -1
        self.model.geom_rgba[floor] = [.94, .95, .97, 1.]
        self.model.geom_size[floor, :2] = 20.
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, size, size)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0., 0., 0.]
        self.camera.distance = 29.
        self.camera.azimuth, self.camera.elevation = 90., -90.
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 1

    def frame(self, qpos, qvel, mocap_pos, mask, qp_ids, trajectory, reward, step, status):
        mj, model = self.mj, self.model
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.mocap_pos[:] = mocap_pos
        mj.mj_forward(model, self.data)
        ids = self.env.crowd_geom_ids
        model.geom_rgba[ids] = [.5, .5, .5, 1.]
        model.geom_rgba[ids, 3] = mask
        for i in qp_ids:
            if 0 <= int(i) < len(ids) and mask[int(i)] > .5:
                model.geom_rgba[ids[int(i)], :3] = [.9, .05, .05]
        model.geom_rgba[self.env.agent_geom_ids] = [.15, .4, .95, 1.]
        model.geom_rgba[model.geom("goal1").id] = [.95, .8, .1, 1.]
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        scene = self.renderer.scene

        def segment(a, b, color, width=2.):
            if scene.ngeom == scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_LINE, width,
                             np.r_[a, .04], np.r_[b, .04])
            scene.ngeom += 1

        blue = [.15, .4, .95, 1.]
        for a, b in zip(trajectory[:-1], trajectory[1:]):
            segment(a, b, blue)
        low, high = np.asarray(self.env.nav_cfg.workspace_low), np.asarray(self.env.nav_cfg.workspace_high)
        corners = [low, [high[0], low[1]], high, [low[0], high[1]], low]
        for a, b in zip(corners[:-1], corners[1:]):
            segment(a, b, [.1, .1, .1, 1.], 1.)
        angles = np.linspace(0., 2*np.pi, 33)
        circle = qpos[:2] + self.env.nav_cfg.robot_radius * np.c_[np.cos(angles), np.sin(angles)]
        for a, b in zip(circle[:-1], circle[1:]):
            segment(a, b, blue)
        from PIL import Image, ImageDraw
        image = Image.fromarray(self.renderer.render().copy())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.env.render_size, 42), fill="white")
        draw.text((10, 6), f"CRAX MJX Point | {step*self.env.nav_cfg.dt:.1f}s | {status} | return {reward:.2f}\nRed: selected QP obstacles | Blue: planning envelope and trajectory", fill="black")
        return np.asarray(image)

    def close(self):
        self.renderer.close()


def save_videos(env, video_dir, traces, records, env_cfg, hparams, module):
    import json
    from scripts.eval import save_video

    renderer = RecordedPointRenderer(env)
    try:
        for index, (trace, record) in enumerate(zip(traces, records)):
            length = record["episode_length"]
            arrays = {name: np.concatenate([frame["snapshot"][name] for frame in trace])
                      for name in ("qpos", "qvel", "mocap_pos")}
            qpos, qvel, mocap = (arrays[name] for name in ("qpos", "qvel", "mocap_pos"))
            mask = trace[0]["snapshot"]["mask"][0]
            ids = np.asarray([frame["qp_ids"] for frame in trace[1:]])
            rewards = np.asarray([frame["reward"] for frame in trace[1:]], dtype=np.float32)
            totals = np.r_[0., np.cumsum(rewards)]
            status = ("collision" if record["collision"] else "success" if record["success"]
                      else "timeout")
            frames = [renderer.frame(
                qpos[t], qvel[t], mocap[t], mask, trace[t]["qp_ids"], qpos[:t+1, :2],
                totals[t], t, status if t == length else "running")
                for t in range(length+1)]
            filename = f"episode_{index:03d}_seed_{record['seed']}.mp4"
            save_video(video_dir/filename, frames, fps=1./env_cfg.dt)
            record["video"] = filename
            np.savez_compressed(video_dir/filename.replace(".mp4", ".npz"),
                qpos=qpos, qvel=qvel, mocap_pos=mocap, reward=rewards,
                qp_ids=ids, mask=mask, seed=record["seed"])
            print(f"Video: {video_dir/filename}", flush=True)
    finally:
        renderer.close()
    (video_dir/"episodes.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")
