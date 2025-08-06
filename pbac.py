import torch
from architectures import Actor, ActorNetEnsemble, ParallelCriticNet
from bootdqn import BootDQN, BootstrapEnsembleLoss, BootDQNCritic, BootDQNCritics


class PACBayesLoss(BootstrapEnsembleLoss):

    def __init__(self, args):
        super(PACBayesLoss, self).__init__(args)

    def forward(self, q, y, weights=None):
        mu = y.mean(dim=0)
        mu_0 = y.mean(dim=0)
        self.args.prior_variance = 1.0
        sig2_0 = self.args.prior_variance
        self.args.gamma = 0.99

        bootstrap_mask = (torch.rand_like(q) >= self.bootstrap_rate)*1.0
        sig2 = (q*bootstrap_mask).var(dim=0).clamp(1e-6, None)
        logsig2 = sig2.log()

        err_0 = (q - mu_0)*bootstrap_mask
        term1 = -0.5*logsig2 
        term2 = 0.5*(err_0**2).mean(dim=0)/sig2_0
        kl_term = (term1 + term2).mean()
        var_offset = (-self.args.gamma**2 * logsig2).mean()
        emp_loss= (((q - y)*bootstrap_mask)**2).mean()
        q_loss = emp_loss + kl_term + var_offset

        return q_loss

##################################################################################################
class PBACParallelCritic(BootDQNCritic):
    def __init__(self, arch, args, n_state, n_action):
        super(PBACParallelCritic, self).__init__(arch, args, n_state, n_action)
        self.loss = PACBayesLoss(args)

##################################################################################################
class PBACParallelCritics(BootDQNCritics):
    def __init__(self, arch, args, n_state, n_action, critictype=PBACParallelCritic):
        super(PBACParallelCritics, self).__init__(arch, args, n_state, n_action, critictype)

    @torch.no_grad()
    def get_bellman_target(self, r, sp, done, actor):
        ap = actor.get_action(sp)
        
        ap = ap[:,actor.idx_active_critic,:].squeeze()
        qp_t = self.Q_t(sp, ap)

        q_t = r.unsqueeze(-1) + (self.args.gamma * qp_t * (1 - done.unsqueeze(-1)))
        return q_t

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
    
######################################################################
class PBAC(BootDQN):
    _agent_name = "ParallelPBAC"

    def __init__(
        self,
        env,
        args,
        actor_nn=ActorNetEnsemble,
        critic_nn=ParallelCriticNet,
        CriticEnsembleType=PBACParallelCritics,
        actor_type=ThompsonActor,
    ):
        super(PBAC, self).__init__(
            env,
            args,
            CriticEnsembleType=PBACParallelCritics,
            actor_nn=actor_nn,
            critic_nn=critic_nn,
            ActorType=actor_type,
        )