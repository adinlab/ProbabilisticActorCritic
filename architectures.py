import copy
import itertools
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, func as thf
from torch.nn import _reduction as _Reduction, functional as F

from experience_memory import ExperienceMemoryTorch

Module = nn.Module


def totorch(x, dtype=th.float32, device="cuda"):
    return th.as_tensor(x, dtype=dtype, device=device)


def tonumpy(x):
    return x.data.cpu().numpy()


def create_net(d_in, d_out, depth, width, act="crelu", has_norm=True, n_elements=1):
    assert depth > 0, "Need at least one layer"

    double_width = False
    if act == "crelu":
        act = CReLU
        double_width = True
    elif act == "relu":
        act = nn.ReLU
    else:
        raise NotImplementedError(f"{act} is not implemented")

    if depth == 1:
        arch = nn.Linear(d_in, d_out)
    elif depth == 2:
        arch = nn.Sequential(
            nn.Linear(d_in, width),
            (
                nn.LayerNorm(width, elementwise_affine=False)
                if has_norm
                else nn.Identity()
            ),
            act(),
            nn.Linear(2 * width if double_width else width, d_out),
        )
    else:
        in_layer = nn.Linear(d_in, width)
        if n_elements > 1:
            out_layer = nn.Linear(
                2 * width if double_width else width, d_out, n_elements
            )
        else:
            out_layer = nn.Linear(2 * width if double_width else width, d_out)

        hidden = list(
            itertools.chain.from_iterable(
                [
                    [
                        (
                            nn.LayerNorm(width, elementwise_affine=False)
                            if has_norm
                            else nn.Identity()
                        ),
                        act(),
                        nn.Linear(2 * width if double_width else width, width),
                    ]
                    for _ in range(depth - 1)
                ]
            )
        )[:-1]
        arch = nn.Sequential(in_layer, *hidden, out_layer)

    return arch


############################################################################
class CReLU(nn.Module):

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x, -x), -1)
        return F.relu(x)


############################################################################
class ActorNet(nn.Module):
    def __init__(
        self,
        dim_obs,
        dim_act,
        depth=3,
        width=256,
        act="crelu",
        has_norm=True,
        upper_clamp=None,
    ):
        super(ActorNet, self).__init__()

        self.arch = create_net(
            dim_obs[0], dim_act[0], depth, width, act, has_norm
        ).append(nn.Tanh())

    def forward(self, x, is_training=None):
        out = self.arch(x).clamp(-0.9999, 0.9999)
        return out, None


############################################################################
class ActorNetEnsemble(ActorNet):
    def __init__(
        self,
        dim_obs,
        dim_act,
        depth=3,
        width=256,
        act="crelu",
        has_norm=True,
        upper_clamp=None,
        n_elements=2,
    ):
        super(ActorNetEnsemble, self).__init__(
            dim_obs, dim_act, depth, width, act, has_norm, upper_clamp
        )

        self.dim_act = dim_act
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.arch = create_net(
            dim_obs[0], dim_act[0] * n_elements, depth, width, act, has_norm
        ).append(nn.Tanh())

        self.n_elements = n_elements

    def forward(self, x, is_training=None):
        x = x.to(self.device)
        out = self.arch(x).clamp(-0.9999, 0.9999)
        out = out.view(-1, self.n_elements, self.dim_act[0])
        return out, None


############################################################################
class ParallelCriticNet(nn.Module):
    def __init__(
        self, dim_obs, dim_act, depth=3, width=256, act="crelu", has_norm=True
    ):
        super(ParallelCriticNet, self).__init__()

        self.arch = create_net(
            dim_obs[0] + dim_act[0], 1, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, xu):
        return self.arch(xu)


############################################################################
class ParallelCritic(nn.Module):
    def __init__(self, arch, args, n_state, n_action):
        super(ParallelCritic, self).__init__()
        self.args = args
        self.arch = arch
        args.device = "cuda" if th.cuda.is_available() else "cpu"
        self.model = arch(
            n_state,
            n_action,
            depth=3,
            width=256,
            act="crelu",
            has_norm=not False,
        ).to(args.device)
        self.target = arch(
            n_state,
            n_action,
            depth=3,
            width=256,
            act="crelu",
            has_norm=not False,
        ).to(args.device)
        self.init_target()
        self.loss = nn.HuberLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), 3e-4)
        self.iter = 0
        self.args.tau = 0.005
        self.args.device = "cuda" if th.cuda.is_available() else "cpu"

    def set_writer(self, writer):
        self.writer = writer

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def Q(self, s, a):
        if a.shape == ():
            a = a.view(1, 1)
            s = s.to(self.args.device)
            a = a.to(self.args.device)
        return self.model(th.cat((s, a), -1))

    def Q_t(self, s, a):
        if a.shape == ():
            a = a.view(1, 1)
        return self.target(th.cat((s, a), -1))

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()
        self.iter += 1


############################################################################
class ParallelCritics(nn.Module):
    def __init__(self, arch, args, n_state, n_action, critictype=ParallelCritic):
        super(ParallelCritics, self).__init__()
        self.n_members = 2
        self.args = args
        self.arch = arch
        self.n_state = n_state
        self.n_action = n_action
        self.critictype = critictype
        self.iter = 0
        self.loss = self.critictype(
            self.arch, self.args, self.n_state, self.n_action
        ).loss
        self.optim = self.critictype(
            self.arch, self.args, self.n_state, self.n_action
        ).optim

        # Helperfunctions
        self.expand = lambda x: (
            x.expand(self.n_members, *x.shape) if len(x.shape) < 3 else x
        )

        self.args.learning_rate = 3e-4
        self.args.gamma = 0.99
        self.args.tau = 0.005

        self.reset()

    def reset(self):
        self.critics = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action)
            for _ in range(self.n_members)
        ]

        self.critics_model = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action).model
            for _ in range(self.n_members)
        ]
        self.critics_target = [
            self.critictype(self.arch, self.args, self.n_state, self.n_action).target
            for _ in range(self.n_members)
        ]

        self.params_model, self.buffers_model = thf.stack_module_state(
            self.critics_model
        )
        self.params_target, self.buffers_target = thf.stack_module_state(
            self.critics_target
        )

        self.base_model = copy.deepcopy(self.critics[0].model).to("meta")
        self.base_target = copy.deepcopy(self.critics[0].target).to("meta")

        def _fmodel(base_model, params, buffers, x):
            return thf.functional_call(base_model, (params, buffers), (x,))

        self.forward_model = thf.vmap(lambda p, b, x: _fmodel(self.base_model, p, b, x))
        self.forward_target = thf.vmap(
            lambda p, b, x: _fmodel(self.base_target, p, b, x)
        )
        self.optim = th.optim.Adam(
            self.params_model.values(), lr=self.args.learning_rate
        )

    def reduce(self, q_val):
        return q_val.min(0)[0]

    def __getitem__(self, item):
        return self.critics[item]

    def unstack(self, target=False, single=True, net_id=None):
        """
        Extract the single parameters back to the individual members
        target: whether the target ensemble should be extracted or not
        single: whether just the first member of the ensemble should be extracted
        """
        params = self.params_target if target else self.params_model
        if single and net_id is None:
            net_id = 0

        for key in params.keys():
            if single:
                tmp = (
                    self.critics[net_id].model
                    if not target
                    else self.critics[net_id].target
                )
                for name in key.split("."):
                    tmp = getattr(tmp, name)
                tmp.data.copy_(params[key][net_id])
            else:
                for net_id in range(self.n_members):
                    tmp = (
                        self.critics[net_id].model
                        if not target
                        else self.critics[net_id].target
                    )
                    for name in key.split("."):
                        tmp = getattr(tmp, name)
                    tmp.data.copy_(params[key][net_id])
                    if single:
                        break

    def set_writer(self, writer):
        assert (
            writer is None
        ), "For now nothing else is implemented for the parallel version"
        self.writer = writer
        [critic.set_writer(writer) for critic in self.critics]

    def Q(self, s, a):
        SA = self.expand(th.cat((s, a), -1))
        return self.forward_model(self.params_model, self.buffers_model, SA)

    @th.no_grad()
    def Q_t(self, s, a):
        SA = self.expand(th.cat((s, a), -1))
        return self.forward_target(self.params_target, self.buffers_target, SA)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), self.expand(y))
        loss.backward()
        self.optim.step()
        self.iter += 1

    @torch.no_grad()
    def update_target(self):
        for key in self.params_model.keys():
            self.params_target[key].data.mul_(1.0 - self.args.tau)
            self.params_target[key].data.add_(
                self.args.tau * self.params_model[key].data
            )

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        qp_t = self.reduce(qp) - alpha * (ep if ep is not None else 0)
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y


############################################################################
class Critic(nn.Module):
    def __init__(self, arch, args, n_state, n_action):
        super(Critic, self).__init__()
        self.args = args
        args.n_hidden = 256
        args.learning_rate = 3e-4
        self.args.tau = 0.005
        args.device = "cuda" if th.cuda.is_available() else "cpu"
        self.model = arch(n_state, n_action, args.n_hidden).to(args.device)
        self.target = arch(n_state, n_action, args.n_hidden).to(args.device)
        self.init_target()
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.iter = 0

    def set_writer(self, writer):
        self.writer = writer

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    @th.no_grad()
    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def Q(self, s, a):
        return self.model(s, a)

    def Q_t(self, s, a):
        return self.target(s, a)

    def update(self, s, a, y):  # y denotes bellman target
        self.optim.zero_grad()
        loss = self.loss(self.Q(s, a), y)
        loss.backward()
        self.optim.step()
        self.iter += 1


############################################################################
class CriticEnsemble(nn.Module):
    def __init__(self, arch, args, n_state, n_action, critictype=Critic):
        super(CriticEnsemble, self).__init__()
        args.n_critics = 2
        self.n_elements = args.n_critics
        self.args = args
        self.critics = [
            critictype(arch, args, n_state, n_action) for _ in range(self.n_elements)
        ]
        self.iter = 0
        self.args.gamma = 0.99
        self.args.tau = 0.005
        self.args.device = "cuda" if th.cuda.is_available() else "cpu"

    def __getitem__(self, item):
        return self.critics[item]

    def set_writer(self, writer):
        self.writer = writer
        [critic.set_writer(writer) for critic in self.critics]

    def Q(self, s, a):
        return [critic.Q(s, a) for critic in self.critics]

    def Q_t(self, s, a):
        return [critic.Q_t(s, a) for critic in self.critics]

    def update(self, s, a, y):
        [critic.update(s, a, y) for critic in self.critics]
        self.iter += 1

    def update_target(self):
        [critic.update_target() for critic in self.critics]

    def reduce(self, q_val_list):
        return torch.stack(q_val_list, dim=-1).min(dim=-1)[0]

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        ap, ep = actor.act(sp)
        qp = self.Q_t(sp, ap)
        if ep is None:
            ep = 0
        qp_t = self.reduce(qp) - alpha * ep
        y = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y


############################################################################
class Actor(nn.Module):
    def __init__(self, arch, args, n_state, n_action, has_target=False):
        super().__init__()
        self.model = arch(
            n_state,
            n_action,
            depth=3,
            width=256,
            act="crelu",
            has_norm=not False,
        ).to("cuda" if th.cuda.is_available() else "cpu")
        args.learning_rate = 3e-4

        self.optim = torch.optim.Adam(self.model.parameters(), args.learning_rate)
        self.args = args
        self.has_target = has_target
        self.iter = 0
        self.is_episode_end = False
        self.states = []
        self.print_freq = 500
        self.args.verbose = False
        self.args.tau = 0.005

        if has_target:
            self.target = arch(
                n_state,
                n_action,
                depth=3,
                width=256,
                act="crelu",
                has_norm=not False,
            ).to("cuda" if th.cuda.is_available() else "cpu")
            self.init_target()

    def init_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def set_writer(self, writer):
        self.writer = writer

    def act(self, s, is_training=True):
        a, e = self.model(s, is_training=is_training)
        if is_training:
            if self.args.verbose and self.iter % self.print_freq == 0:
                self.states.append(tonumpy(s))
        return a, e

    def act_target(self, s):
        a, e = self.target(s)
        return a, e

    def set_episode_status(self, is_end):
        self.is_episode_end = is_end

    @th.no_grad()
    def update_target(self):
        for target_param, local_param in zip(
            self.target.parameters(), self.model.parameters()
        ):

            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def loss(self, s, critics):
        a, _ = self.act(s)
        q_list = critics.Q(s, a)
        q = critics.reduce(q_list)
        return (-q).mean(), None

    def update(self, s, critics):
        self.optim.zero_grad()
        loss, _ = self.loss(s, critics)
        loss.backward()
        self.optim.step()

        if self.has_target:
            self.update_target()

        self.iter += 1


############################################################################
class Agent(nn.Module):
    def __init__(self, env, args):
        super(Agent, self).__init__()
        self.args = args
        args.device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = args.device
        self.env = env
        self.dim_obs, self.dim_act = (
            self.env.observation_space.shape,
            self.env.action_space.shape,
        )
        print(f"INFO: dim_obs = {self.dim_obs} dim_act = {self.dim_act}")
        self.dim_obs_flat, self.dim_act_flat = np.prod(self.dim_obs), np.prod(
            self.dim_act
        )
        self._u_min = totorch(self.env.action_space.low, device=self.device)
        self._u_max = totorch(self.env.action_space.high, device=self.device)
        self._x_min = totorch(self.env.observation_space.low, device=self.device)
        self._x_max = totorch(self.env.observation_space.high, device=self.device)
        self.args.tau = 0.005
        self._gamma = 0.99
        self._tau = 0.005
        args.buffer_size = 100000

        args.dims = {
            "state": (args.buffer_size, self.dim_obs_flat),
            "action": (args.buffer_size, self.dim_act_flat),
            "next_state": (args.buffer_size, self.dim_obs_flat),
            "reward": (args.buffer_size),
            "terminated": (args.buffer_size),
            "step": (args.buffer_size),
        }

        self.experience_memory = ExperienceMemoryTorch(args)

    def set_writer(self, writer):
        self.writer = writer

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.mul_(1.0 - self.args.tau)
            target_param.data.add_(self.args.tau * local_param.data)

    def _hard_update(self, local_model, target_model):

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def learn(self, max_iter=1):
        raise NotImplementedError(f"learn() not implemented for {self.name} agent")

    def select_action(self, warmup=False, exploit=False):
        raise NotImplementedError(
            f"select_action() not implemented for {self.name} agent"
        )

    def store_transition(self, s, a, r, sp, terminated, truncated, step):
        self.experience_memory.add(s, a, r, sp, terminated, step)
        self.actor.set_episode_status(terminated or truncated)


############################################################################
class ActorCritic(Agent):
    _agent_name = "AC"

    def __init__(
        self,
        env,
        args,
        actor_nn,
        critic_nn,
        CriticEnsembleType=CriticEnsemble,
        ActorType=Actor,
    ):
        super(ActorCritic, self).__init__(env, args)
        self.critics = CriticEnsembleType(critic_nn, args, self.dim_obs, self.dim_act)
        self.actor = ActorType(actor_nn, args, self.dim_obs, self.dim_act)
        self.n_iter = 0
        self.policy_delay = 1
        self.args.batch_size = 256

    def set_writer(self, writer):
        self.writer = writer
        self.actor.set_writer(writer)
        self.critics.set_writer(writer)

    def learn(self, max_iter=1):
        if self.args.batch_size > len(self.experience_memory):
            return None

        for ii in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(
                self.args.batch_size
            )
            y = self.critics.get_bellman_target(r, sp, done, self.actor)
            self.critics.update(s, a, y)

            if self.n_iter % self.policy_delay == 0:
                self.actor.update(s, self.critics)
            self.critics.update_target()
            self.n_iter += 1

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        a, _ = self.actor.act(s, is_training=is_training)
        return a

    def Q_value(self, s, a):

        if len(s.shape) == 1:
            s = s[None]
        if len(a.shape) == 1:
            a = a[None]
        if isinstance(self.critics, ParallelCritics):
            self.critics.unstack(target=False, single=True)

        q = self.critics[0].Q(s, a)
        return q.item()


############################################################################
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


############################################################################
class ProbabilisticLoss(_Loss):

    def forward(self, mu, logvar, y):
        pass
