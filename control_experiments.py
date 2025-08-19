import time, argparse, torch
import numpy as np
from make_environment import Experiment


# Helpers for tensor conversion
def totorch(x, device="cpu"):
    return torch.from_numpy(x).float().to(device)


def tonumpy(x):
    return x.detach().cpu().numpy()


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--eval_frequency", type=int, default=1000)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--warmup_steps", type=int, default=10000)
parser.add_argument("--learn_frequency", type=int, default=1)
parser.add_argument("--max_iter", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()


class ControlExperiment(Experiment):
    def __init__(self, args):
        super(ControlExperiment, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.optimizer_args = {"lr": 4e-3}
        self.n_total_steps = 0

    def train(self):
        time_start = time.time()

        information_dict = {
            "episode_rewards": np.zeros(10000),
            "episode_steps": np.zeros(10000),
            "step_rewards": np.empty((2 * self.args.max_steps), dtype=object),
        }

        s, _ = self.env.reset()
        s = totorch(s, self.device)
        r_cum = 0
        episode = 0
        e_step = 0

        for step in range(self.args.max_steps):
            e_step += 1

            if step % self.args.eval_frequency == 0:
                self.eval(step)

            if step < self.args.warmup_steps:
                a = self.env.action_space.sample()
                a = totorch(np.clip(a, -1.0, 1.0), self.device)
            else:
                a = self.agent.select_action(s).clip(-1.0, 1.0)

            sp, r, done, truncated, info = self.env.step(tonumpy(a))
            sp = totorch(sp, self.device)

            self.agent.store_transition(s, a, r, sp, done, truncated, step + 1)
            information_dict["step_rewards"][self.n_total_steps + step] = (
                episode,
                step,
                r,
            )

            s = sp
            r_cum += r

            if (
                step >= self.args.warmup_steps
                and (step % self.args.learn_frequency) == 0
            ):
                self.agent.learn(max_iter=self.args.max_iter)

            if done or truncated:
                information_dict["episode_rewards"][episode] = r_cum
                information_dict["episode_steps"][episode] = step
                print("Episode:", episode, "Reward: %.3f" % r_cum, "N-steps:", step)
                s, _ = self.env.reset()
                s = totorch(s, self.device)
                r_cum = 0
                episode += 1
                e_step = 0

        self.eval(step)
        time_end = time.time()

    @torch.no_grad()
    def eval(self, n_step):
        self.agent.eval()
        results = np.zeros(self.args.eval_episodes)
        q_values = np.zeros(self.args.eval_episodes)
        avg_reward = np.zeros(self.args.eval_episodes)

        for episode in range(self.args.eval_episodes):
            s, _ = self.eval_env.reset()
            s = totorch(s, self.device)
            step = 0
            a = self.agent.select_action(s, is_training=False)
            q_values[episode] = self.agent.Q_value(s, a)
            done = False

            while not done:
                a = self.agent.select_action(s, is_training=False)
                sp, r, term, trunc, info = self.eval_env.step(tonumpy(a))
                done = term or trunc
                s = totorch(sp, self.device)
                results[episode] += r
                avg_reward[episode] += self.args.gamma**step * r
                step += 1

        self.agent.train()
