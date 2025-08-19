from control_experiments import ControlExperiment
import argparse

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

# ✅ Pass args when creating the experiment
exp = ControlExperiment(args)
exp.train()
