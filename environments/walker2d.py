# Walker2d-v2 env
# two objectives
# running speed, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/walker2d.xml"), frame_skip = 4)

    def step(self, a):
        qpos0_sum = np.sum(self.sim.data.qpos)
        qvel0_sum = np.sum(self.sim.data.qvel)
        posbefore = self.sim.data.qpos[0]
        a = np.clip(a, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward_speed = (posafter - posbefore) / self.dt + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(a).sum() + alive_bonus
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        return ob, 0., done, {'obj': np.array([reward_speed, reward_energy])}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20