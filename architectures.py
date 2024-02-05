
class CriticNetEpistemic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256):
        super(CriticNetEpistemic, self).__init__()
        self.arch = nn.Sequential(nn.Linear(n_x[0] + n_u[0], n_hidden),
                                  nn.ReLU(),
                                  ###
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  ###
                                  )
        self.head = EpistemicGaussianHead(256, 1)

    def forward(self, x, u):
        f = torch.concat([x, u], dim=-1)
        f=self.arch(f)
        mu, var =  self.head.get_mean_var(f)
        return torch.concat([mu, var.log()], dim=-1)

class EpistemicGaussianHead(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior_log_sig2 =0) -> None:
        super(EpistemicGaussianHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_mu_prior = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=False)
        self.weight_log_sig2_prior = nn.Parameter(prior_log_sig2*torch.zeros((out_features, in_features)), requires_grad=False)
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sig2 = nn.Parameter(torch.Tensor(out_features))
            self.bias_mu_prior = nn.Parameter(torch.zeros(out_features), requires_grad=False)
            self.bias_log_sig2_prior = nn.Parameter(prior_log_sig2*torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sig2', None)
        self.reset_parameters(prior_log_sig2=prior_log_sig2)

    def reset_parameters(self, prior_log_sig2: float) -> None:
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(self.weight_mu.shape[1]))
        init.constant_(self.weight_log_sig2, -10)
        if self.has_bias:
            init.zeros_(self.bias_mu)
            init.constant_(self.bias_log_sig2, -10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp())
        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2) #,output_mu,output_sig2

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.linear(input, self.weight_mu, self.bias_mu)
        sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp())
        return mu, sig2


class ActorNetProbabilistic(nn.Module):
    def __init__(self, n_x, n_u, n_hidden=256, upper_clamp=-2.0):
        super(ActorNetProbabilistic, self).__init__()
        self.n_u = n_u
        self.arch = nn.Sequential(
                   nn.Linear(n_x[0], 256),
                   nn.ReLU(),
                   ##
                   nn.Linear(256, 256),
                   nn.ReLU(),
                   ##
                   nn.Linear(256, 2*n_u[0]))
        self.head = SquashedGaussianHead(self.n_u[0], upper_clamp)

    def forward(self, x, is_training=True):
        f = self.arch(x)
        return self.head(f, is_training)


class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super(SquashedGaussianHead, self).__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x, is_training=True):
        # bt means before tanh
        mean_bt = x[...,:self._n]
        log_var_bt = (x[...,self._n:]).clamp(-10,-self._upper_clamp)
        std_bt = log_var_bt.exp().sqrt()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)
        if is_training:
            y = dist.rsample()
            y_logprob = dist.log_prob(y).sum(dim=-1, keepdim=True)
        else:
            y_samples = dist.rsample((100,))
            y = y_samples.mean(dim=0)
            y_logprob = None

        return y, y_logprob # dist
