# Hopper-v3 env
# three objectives
# running speed, jumping height, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 3
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/hopper.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        a = np.clip(a, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward_energy = 4.0 - 1.0 * np.square(a).sum() + alive_bonus
        reward_run = 1.5 * (posafter - posbefore) / self.dt + alive_bonus
        reward_jump = 12. * (height - self.init_qpos[1]) + alive_bonus
        s = self.state_vector()
        done = not((s[1] > 0.4) and abs(s[2]) < np.deg2rad(90) and abs(s[3]) < np.deg2rad(90) and abs(s[4]) < np.deg2rad(90) and abs(s[5]) < np.deg2rad(90))

        ob = self._get_obs()
        return ob, reward, done, {'obj': np.array([reward_run, reward_jump, reward_energy])}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
