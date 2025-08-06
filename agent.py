import torch.nn as nn
import numpy as np
import torch,argparse
from experience_memory import ExperienceMemory
parser = argparse.ArgumentParser()
args = parser.parse_args()

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env
        args.buffer_size=1000000
        self._nx, self._nu = self.env.observation_space.shape, self.env.action_space.shape
        self._nx_flat, self._nu_flat = np.prod(self._nx), np.prod(self._nu)
        self._u_min = torch.from_numpy(self.env.action_space.low).float().to(self.device)
        self._u_max = torch.from_numpy(self.env.action_space.high).float().to(self.device)
        self._x_min = torch.from_numpy(self.env.observation_space.low).float().to(self.device)
        self._x_max = torch.from_numpy(self.env.observation_space.high).float().to(self.device)

        self._gamma = 0.99
        self._tau = 0.005

        args.dims = {
            "state": (args.buffer_size, self._nx_flat),
            "action": (args.buffer_size, self._nu_flat),
            "next_state": (args.buffer_size, self._nx_flat),
            "reward": (args.buffer_size),
            "terminated": (args.buffer_size),
            "step": (args.buffer_size)
        }

        self.experience_memory = ExperienceMemory(args)

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)

    def _hard_update(self, local_model, target_model):


        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def learn(self, max_iter=1):
        raise NotImplementedError(f"learn() not implemented for {self.name} agent")

    def select_action(self, warmup=False, exploit=False):
        raise NotImplementedError(f"select_action() not implemented for {self.name} agent")

    def store_transition(self, s, a, r, sp, terminated, step):
        self.experience_memory.add(s, a, r, sp, terminated, step)
