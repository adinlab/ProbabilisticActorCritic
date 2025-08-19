import torch
from architectures import (
    ActorCritic,
    Actor,
    ProbabilisticLoss,
    ParallelCritics,
    ParallelCritic,
    ActorNetEnsemble,
    ParallelCriticNet,
)


def tonumpy(x):
    return x.data.cpu().numpy()


#####################################################################
class BootstrapEnsembleLoss(ProbabilisticLoss):

    def __init__(self, args):
        super(BootstrapEnsembleLoss, self).__init__()
        self.args = args
        self.args.bootstrap_rate = 0.05
        self.args.dqn_l2_reg = 1.0
        self.bootstrap_rate = self.args.bootstrap_rate

    def forward(self, q, y, weights):
        bootstrap_mask = (torch.rand_like(q) >= self.bootstrap_rate) * 1.0
        emp_loss = ((q - y) * bootstrap_mask) ** 2
        prior_loss = weights.pow(2).sum() * self.args.dqn_l2_reg
        return emp_loss.mean() + prior_loss


##################################################################################################
class BootDQNCritic(ParallelCritic):
    def __init__(self, arch, args, n_state, n_action):
        super(BootDQNCritic, self).__init__(arch, args, n_state, n_action)
        self.loss = BootstrapEnsembleLoss(args)


##################################################################################################
class BootDQNCritics(ParallelCritics):
    def __init__(self, arch, args, n_state, n_action, critictype=BootDQNCritic):
        super(BootDQNCritics, self).__init__(arch, args, n_state, n_action, critictype)
        self.args.gamma = 0.99

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        ap = actor.get_action(sp)
        sp = self.expand(sp)
        ap = ap.swapaxes(0, 1)
        SA = torch.cat((sp, ap), 2)
        qp_t = self.forward_target(self.params_target, self.buffers_target, SA)
        q_t = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return q_t

    @torch.no_grad()
    def Q_t(self, s, a):
        if len(a.shape) == 1:
            a = a.view(-1, 1)
        SA = self.expand(torch.cat((s, a), -1))
        return self.forward_target(self.params_target, self.buffers_target, SA)

    def update(self, s, a, y):
        self.optim.zero_grad()
        l_layer = len(self.base_model.arch) - 1
        weights = self.params_model[f"arch.{l_layer}.weight"]
        self.loss(self.Q(s, a), y, weights).backward()
        self.optim.step()
        self.iter += 1


#####################################################################
class ThompsonActor(Actor):
    def __init__(self, arch, args, n_state, n_action):
        super().__init__(arch, args, n_state, n_action)
        self.idx_active_critic = 0
        self.interaction_iter = 0
        self.args.posterior_sampling_rate = 5
        self.sampling_rate = self.args.posterior_sampling_rate
        self.iter_num = []
        self.q_vars_mean = []
        self.q_vars_std = []
        self.print_freq = 500
        self.args.verbose = False

    def act(self, s, is_training=True):
        a, e = self.model(s)

        if is_training:
            if self.is_episode_end or (self.interaction_iter % self.sampling_rate == 0):
                self.idx_active_critic = torch.randint(0, a.size(1), (1,)).item()
            self.interaction_iter += 1
            a = a[:, self.idx_active_critic, :].squeeze()

        else:
            a = a.mean(dim=1).squeeze()

        return a, e

    def get_action(self, s):
        a, _ = self.model(s)
        return a

    def loss(self, s, critics):
        a = self.get_action(s)
        s = critics.expand(s)
        a = a.swapaxes(0, 1)
        SA = torch.cat((s, a), 2)
        q = critics.forward_model(critics.params_model, critics.buffers_model, SA)

        return (-q).mean(), None


#####################################################################
class BootDQN(ActorCritic):
    _agent_name = "BootDQN"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetEnsemble,
        critic_nn=ParallelCriticNet,
        CriticEnsembleType=BootDQNCritics,
        ActorType=ThompsonActor,
    ):
        super().__init__(
            env=env,
            args=args,
            actor_nn=actor_nn,
            critic_nn=critic_nn,
            CriticEnsembleType=CriticEnsembleType,
            ActorType=ActorType,
        )
