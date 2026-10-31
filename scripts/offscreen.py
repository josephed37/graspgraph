"""Offscreen video recording shared by the demo scripts.

Call `setup_gl()` before anything imports mujoco: MuJoCo binds its OpenGL backend once, at
import time. Then `VideoRecorder` renders frames from a robot's on_step callback.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_gl() -> str:
    """Use MUJOCO_GL if set, otherwise the first offscreen backend that works: EGL, then OSMesa.

    Backends are probed in a subprocess, because a failed import cannot be undone in-process.
    """
    if "MUJOCO_GL" not in os.environ:
        for backend in ("egl", "osmesa"):
            probe = subprocess.run(
                [sys.executable, "-c", "import mujoco; mujoco.GLContext(16, 16).make_current()"],
                env={**os.environ, "MUJOCO_GL": backend},
                capture_output=True,
            )
            if probe.returncode == 0:
                os.environ["MUJOCO_GL"] = backend
                break
        else:
            raise RuntimeError("No offscreen OpenGL backend works; install libEGL or libOSMesa")
    return os.environ["MUJOCO_GL"]


class VideoRecorder:
    """Renders a frame every 1/fps seconds of simulation time; pass `record` as a robot's on_step."""

    def __init__(self, robot, fps: int = 30, width: int = 640, height: int = 480) -> None:
        import mujoco

        self.robot, self.fps = robot, fps
        self.renderer = mujoco.Renderer(robot.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.42, 0.0, 0.22)
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.65, 150.0, -28.0
        self.every = max(1, round(1.0 / (fps * robot.model.opt.timestep)))
        self.frames = []

    def record(self) -> None:
        if self.robot.total_steps % self.every == 0:
            self.renderer.update_scene(self.robot.data, self.camera)
            self.frames.append(self.renderer.render())

    def save(self, out: Path) -> None:
        """Write an .mp4 (or a .gif with a third of the frames, to keep it small)."""
        import imageio.v2 as imageio

        self.renderer.close()  # free the GL context now, not during interpreter shutdown
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".gif":
            imageio.mimsave(out, self.frames[::3], duration=3000 / self.fps, loop=0)
        else:
            imageio.mimsave(out, self.frames, fps=self.fps, macro_block_size=8)
