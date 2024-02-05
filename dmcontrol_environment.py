import copy
import numpy as np
from gymnasium import core, spaces
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from typing import Dict, Optional, OrderedDict
from dm_control import suite
import dm_env

TimeStep = Tuple[np.ndarray, float, bool, bool, dict]

def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(low=spec.minimum,
                          high=spec.maximum,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        return spaces.Box(low=-float('inf'),
                          high=float('inf'),
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


class DMCEnv(core.Env):

    def __init__(self,
                 domain_name: Optional[str] = None,
                 task_name: Optional[str] = None,
                 env: Optional[dm_env.Environment] = None,
                 task_kwargs: Optional[Dict] = {},
                 environment_kwargs=None):
        assert 'random' in task_kwargs, 'Please specify a seed, for deterministic behaviour.'
        assert (
            env is not None
            or (domain_name is not None and task_name is not None)
        ), 'You must provide either an environment or domain and task names.'

        if env is None:
            env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=True
            )

        self._env = env
        self.domain_name = domain_name
        self.task_name = task_name
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info  = {}
        trunc = done and (time_step.discount == 1.0)
        term = done and (time_step.discount != 1.0)
        if trunc:
            info['TimeLimit.truncated'] = True
        return obs, reward, term, trunc, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time_step = self._env.reset()
        info = {}
        return time_step.observation, info

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height,
                                        width=width,
                                        camera_id=camera_id)
