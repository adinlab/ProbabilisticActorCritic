import torch, math
from torch import nn
import numpy as np
import torch.nn.init as init
from agent import Agent

class PAC(Agent):
    _agent_name = "PAC"

    def __init__(self, env, actor_nn, critic_nn):
        super(PAC, self).__init__(env)
        self._alpha = 0.2
        self._n_ensemble = 2
        self._delta = 0.05
        self._scheduled_LR = False
        self._critic_var_upper_clamp = 4.0
        self.n_hidden = 256
        self.learning_rate = 3e-4
        self.actor_var_upper_clamp=-7.0
        self._step = 0
        self._max_steps = 100000
        self._c = 0
        self.batch_size=256
        self._gamma=0.99
        self.critic = []
        self.critic_t = []
        self.critic_optim = []
        self.critic_scheduler = []

        for i in range(self._n_ensemble):
            self.critic.append(critic_nn(self._nx, self._nu, self.n_hidden).to(self.device))
            self.critic_t.append(critic_nn(self._nx, self._nu, self.n_hidden).to(self.device))
            optim_i = torch.optim.Adam(self.critic[i].parameters(), self.learning_rate)
            self.critic_optim.append(optim_i)
            self.critic_scheduler.append(torch.optim.lr_scheduler.LinearLR(optimizer=optim_i, start_factor=1, end_factor=0.1,total_iters=1000))
            self._hard_update(self.critic[i], self.critic_t[i]) # hard update at the beginning

        self._actor = actor_nn(self._nx, self._nu, self.n_hidden, self.actor_var_upper_clamp).to(self.device)
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), self.learning_rate)
        self._actor_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self._actor_optim, start_factor=1, end_factor=0.1,total_iters=1000)


    def KL(self, mu1, mu2, logvar1, logvar2):
        logvar1 = logvar1.clamp(-10,self._critic_var_upper_clamp)
        logvar2 = logvar2.clamp(-10,self._critic_var_upper_clamp)
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        return 0.5*(logvar2 - logvar1) + (var1 + (mu1-mu2)**2) / (2.0 * var2) - 0.5

    def compute_Q_target(self, s, a, r, sp, done):
        # generate q targets
        with torch.no_grad():
            ap_pred, _ = self._actor(sp, is_training=True)
            qp_t_mu_list = []
            qp_t_logvar_list = []
            for j in range(self._n_ensemble):
                qp_i_t = self.critic_t[j](sp, ap_pred)
                qp_i_t_mu = qp_i_t[:,0].view(-1,1)
                qp_i_t_logvar = qp_i_t[:,1].view(-1,1)
                qp_t_mu_list.append(qp_i_t_mu)
                qp_t_logvar_list.append(qp_i_t_logvar)

            # convert list to tensor
            qp_t_mu_list = torch.cat(qp_t_mu_list, dim=-1)
            qp_t_logvar_list = torch.cat(qp_t_logvar_list, dim=-1)

            idx = qp_t_mu_list.argmin(dim=-1, keepdim=True)
            qp_t_logvar = qp_t_logvar_list.gather(1, idx)

            q_t_logvar = 2.0*math.log(self._gamma) + qp_t_logvar
            qp_t_mu = qp_t_mu_list.gather(1, idx)

            q_t_mu = r.unsqueeze(-1) + (self._gamma * qp_t_mu * (1 - done.unsqueeze(-1)))

            return q_t_mu, q_t_logvar

    def Q_eval(self, s, a, critic_list):
        q_pi_mu_list = []
        q_pi_logvar_list = []
        for i in range(self._n_ensemble):
            q_pi_params = critic_list[i](s, a)
            q_pi_mu = q_pi_params[:,0].view(-1,1)
            q_pi_logvar = q_pi_params[:,1].view(-1,1)
            q_pi_mu_list.append(q_pi_mu)
            q_pi_logvar_list.append(q_pi_logvar)

        q_pi_mu_list = torch.cat(q_pi_mu_list, dim=-1)
        q_pi_logvar_list = torch.cat(q_pi_logvar_list, dim=-1)

        idx = q_pi_mu_list.argmin(dim=-1, keepdim=True)
        q_pi_logvar = q_pi_logvar_list.gather(1, idx)
        q_pi_mu = q_pi_mu_list.gather(1, idx)

        return q_pi_mu, q_pi_logvar.clamp(-10,self._critic_var_upper_clamp).exp()

    def learn(self, max_iter=1):
        self._step += 1

        if self.batch_size > len(self.experience_memory):
            return None

        for iteration in range(max_iter):
            s, a, r, sp, done, step = self.experience_memory.sample_random(self.batch_size)
            q_t_mu, q_t_logvar = self.compute_Q_target(s, a, r, sp, done)

            # update critic ensemble
            for i in range(self._n_ensemble):
                self.critic_optim[i].zero_grad()
                q = self.critic[i](s, a)
                q_mu = q[:,0].view(-1,1)
                q_logvar = q[:,1].view(-1,1)
                q_var = q_logvar.clamp(-10,self._critic_var_upper_clamp).exp()

                t1 = 1.0/torch.sqrt(1.0+2.0*q_var)
                pre_loss = (q_mu-q_t_mu)**2
                divisor = (2.0*q_var+1)
                t2 = torch.exp(-pre_loss/divisor)
                q_loss = (1.0 - t1*t2).mean()

                num_observations = self.batch_size
                mu_prior = q_t_mu.clone().detach()
                logvar_prior = torch.ones_like(mu_prior).to(self.device)*(-8)

                N = (torch.ones(1)*num_observations).to(self.device).squeeze()
                confidence_term = torch.log(2.0 * N.sqrt() / (self._delta))
                denominator = 2.0*N

                sqrt_term= ((self.KL(q_mu, mu_prior, q_logvar,logvar_prior).sum()+confidence_term)/denominator).sqrt()
                q_loss += sqrt_term
                q_loss.backward()
                self.critic_optim[i].step()

            #update actor
            self._actor_optim.zero_grad()
            a_pred, e_pred = self._actor(s, is_training=True)

            q_pi_mu, q_pi_var = self.Q_eval(s, a_pred, self.critic)
            pi_loss = (self._alpha*e_pred - (q_pi_mu + 0.5*q_pi_var/self._alpha)).mean()
            pi_loss.backward()
            self._actor_optim.step()

            for i in range(self._n_ensemble):
                self._soft_update(self.critic[i], self.critic_t[i])

    @torch.no_grad()
    def select_action(self, s, is_training=True):
        s = torch.from_numpy(s).view(1,-1).float().to(self.device)
        a, _ = self._actor(s, is_training=is_training)
        a = a.cpu().numpy().squeeze(0)
        return a

    @torch.no_grad()
    def Q_value(self,s,a):
        s = torch.from_numpy(s).view(1,-1).float().to(self.device)
        a = torch.from_numpy(a).view(1,-1).float().to(self.device)
        q =  self.critic[0](s,a)
        q_mu = q[:,0].view(-1,1)
        return q_mu.item()
