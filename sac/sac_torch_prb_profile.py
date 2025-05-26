#!/usr/bin/env python
"""
sac_torch.py

docs and experiment results can be found at 
https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy

This version now includes detailed profiling information for every RL step.
It logs latencies for:
  - Action selection
  - Environment step
  - Replay buffer addition
  - Replay buffer sampling (only for training updates)
  - Target and loss calculation (only for training updates)
  - Backpropagation and network update (only for training updates)

It also logs memory metrics (in bytes) for:
  • mem_nn: The neural network parameters (actor, qf1, qf2).
  • mem_env: The environment.
  • mem_rb: The replay buffer estimated as min(global_step, buffer_size) * 205.
  
For training update steps only, the following extra memory metrics are logged:
  • mem_opt: The combined memory footprint of the optimizers.
  • mem_act: A placeholder for activations memory.
  • mem_grad: Total memory (bytes) used by the stored gradients.

Data for all runs are appended to "profile_log_torch.csv".
"""

# ++++++++++++ Imports and Installs ++++++++++++ #
import os
import random
import time
import csv
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ListStorage
from torchrl.data import PrioritizedReplayBuffer

# ++++++++++++++ Global Variables ++++++++++++++ #
LOG_STD_MAX = 2
LOG_STD_MIN = -5
MAX_TD_ERROR = 10

# ++++++++++++++ Profiling Helper Functions ++++++++++++++ #
def get_model_memory(model):
    """Compute memory (in bytes) used by a model's parameters."""
    total = 0
    for param in model.parameters():
        total += param.element_size() * param.nelement()
    return total

def get_nn_memory():
    """Return the total memory for the actor, qf1, and qf2 networks."""
    global actor, qf1, qf2
    return get_model_memory(actor) + get_model_memory(qf1) + get_model_memory(qf2)

def get_env_memory(envs):
    """Return a rough estimate of the environment's memory footprint."""
    import sys
    return sys.getsizeof(envs)

def get_rb_memory(global_step, buffer_size):
    """Return the memory footprint (in bytes) of the replay buffer."""
    return min(global_step, buffer_size) * 205

def get_opt_memory(optimizer):
    """Return a rough memory usage (in bytes) for an optimizer's state."""
    import sys
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            try:
                total += v.element_size() * v.nelement() if hasattr(v, "element_size") else sys.getsizeof(v)
            except Exception:
                total += sys.getsizeof(v)
    return total

def get_grad_memory(model):
    """Return the memory usage (in bytes) of all gradients in a model."""
    total = 0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.element_size() * param.grad.nelement()
    return total

def get_activations_memory():
    """Placeholder for activations memory footprint (in bytes)."""
    return 0

# ++++++++++++++ Class Definitions +++++++++++++ #
@dataclass
class Args:
    # General Arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm-Specific Arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 20000
    """total timesteps of the experiments (set to 30,000)"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

# NOTE: initialize agent here
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() +
            np.prod(env.single_action_space.shape), 256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

# ++++++++++++++++ Helper Functions +++++++++++++++ #
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

# +++++++++++++++++ Main Function ++++++++++++++++ #
if __name__ == "__main__":

    # Check stable_baselines3 version
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )

    # Setup: parse arguments, initialize wandb and TensorBoard, etc.
    args = tyro.cli(Args)
    # Override total_timesteps to 30,000 (if not already set by CLI)
    args.total_timesteps = 20000

    # Define 5 seeds for the runs
    seeds = [1, 2, 3]

    # Set up the profiling CSV file (name is different for torch version)
    profile_filename = "profile_log_torch.csv"
    if os.path.exists(profile_filename):
        os.remove(profile_filename)
    csv_fields = [
        'seed', 'step', 'is_train_update',
        'latency_action', 'latency_env', 'latency_rb_add', 'latency_rb_sample',
        'latency_target_loss', 'latency_train_net', 'total_latency',
        'mem_nn', 'mem_env', 'mem_rb', 'mem_opt', 'mem_act', 'mem_grad',
        'reward'
    ]

    for current_seed in seeds:
        print(f"=== Starting run with seed {current_seed} ===")
        args.seed = current_seed

        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            ),
        )

        # Setup: data seeding and device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # Setup: environment
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
             for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        max_action = float(envs.single_action_space.high[0])

        # Setup: instantiate networks, target networks, and optimizers (and entropy tuning)
        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()),
                                 lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
        if args.autotune:
            target_entropy = -torch.prod(
                torch.Tensor(envs.single_action_space.shape).to(device)
            ).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        # Setup: create prioritized replay buffer
        envs.single_observation_space.dtype = np.float32
        rb = PrioritizedReplayBuffer(alpha=0.7, beta=0.9,
                                     storage=ListStorage(args.buffer_size))
        start_time = time.time()

        # Start the environment and get the initial observation
        obs, _ = envs.reset(seed=args.seed)
        # Main loop over timesteps
        for global_step in range(args.total_timesteps):
            profile = {'seed': current_seed, 'step': global_step}
            # Define training update flag: update only if past learning_starts and on every policy_frequency step
            is_train_update = (global_step > args.learning_starts) and (global_step % args.policy_frequency == 0)
            profile['is_train_update'] = int(is_train_update)

            # --- 1. Action Selection (latency measured) ---
            t0 = time.time()
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()
            profile['latency_action'] = time.time() - t0

            # --- 2. Environment Step (latency measured) ---
            t0 = time.time()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            profile['latency_env'] = time.time() - t0
            profile['reward'] = np.mean(rewards)

            # Log episodic info if available.
            if 'episode' in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            # --- 3. Replay Buffer Addition (latency measured) ---
            t0 = time.time()
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            # Prepare transition with a TD error placeholder (using MAX_TD_ERROR)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                next_obs_tensor = torch.FloatTensor(real_next_obs).to(device)
                rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
                terminations_tensor = torch.FloatTensor(terminations).reshape(-1, 1).to(device)
                td_errors = torch.ones(obs_tensor.shape[0]) * MAX_TD_ERROR
                transition = TensorDict({
                    "observations": obs_tensor,
                    "next_observations": next_obs_tensor,
                    "actions": torch.FloatTensor(actions).to(device),
                    "rewards": rewards_tensor,
                    "dones": terminations_tensor,
                    "td_error": td_errors,
                }, batch_size=[obs_tensor.shape[0]])
            rb.extend(transition)
            profile['latency_rb_add'] = time.time() - t0

            # Update state for next iteration.
            obs = next_obs

            # --- 4. Training Update (only if past learning_starts) ---
            if global_step > args.learning_starts:
                if is_train_update:
                    t0_sample = time.time()
                    data, info = rb.sample(batch_size=args.batch_size, return_info=True)
                    profile['latency_rb_sample'] = time.time() - t0_sample
                else:
                    data, info = rb.sample(batch_size=args.batch_size, return_info=True)
                    profile['latency_rb_sample'] = ""

                if is_train_update:
                    t0_target_loss = time.time()
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data['next_observations'])
                    qf1_next_target = qf1_target(data['next_observations'], next_state_actions)
                    qf2_next_target = qf2_target(data['next_observations'], next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data['rewards'].flatten() + (1 - data['dones'].flatten()) * args.gamma * (min_qf_next_target).view(-1)
                qf1_a_values = qf1(data['observations'], data['actions']).view(-1)
                qf2_a_values = qf2(data['observations'], data['actions']).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                if is_train_update:
                    profile['latency_target_loss'] = time.time() - t0_target_loss
                else:
                    profile['latency_target_loss'] = ""

                if is_train_update:
                    t0_train_net = time.time()
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data['observations'])
                        qf1_pi = qf1(data['observations'], pi)
                        qf2_pi = qf2(data['observations'], pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data['observations'])
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                if is_train_update:
                    profile['latency_train_net'] = time.time() - t0_train_net
                else:
                    profile['latency_train_net'] = ""
            else:
                profile['latency_rb_sample'] = ""
                profile['latency_target_loss'] = ""
                profile['latency_train_net'] = ""

            # Compute total latency for this step.
            total_latency = profile['latency_action'] + profile['latency_env'] + profile['latency_rb_add']
            for key in ['latency_rb_sample', 'latency_target_loss', 'latency_train_net']:
                if isinstance(profile[key], float):
                    total_latency += profile[key]
            profile['total_latency'] = total_latency

            # --- 5. Memory Logging ---
            profile['mem_nn'] = get_nn_memory()
            profile['mem_env'] = get_env_memory(envs)
            profile['mem_rb'] = get_rb_memory(global_step, args.buffer_size)
            if global_step > args.learning_starts and is_train_update:
                opt_mem = get_opt_memory(q_optimizer) + get_opt_memory(actor_optimizer)
                if args.autotune:
                    opt_mem += get_opt_memory(a_optimizer)
                profile['mem_opt'] = opt_mem
                profile['mem_act'] = get_activations_memory()  # placeholder
                grad_mem = get_grad_memory(actor) + get_grad_memory(qf1) + get_grad_memory(qf2)
                profile['mem_grad'] = grad_mem
            else:
                profile['mem_opt'] = ""
                profile['mem_act'] = ""
                profile['mem_grad'] = ""

            # Append the profiling data to the CSV file.
            with open(profile_filename, "a", newline="") as csvfile:
                writer_csv = csv.DictWriter(csvfile, fieldnames=csv_fields)
                if csvfile.tell() == 0:
                    writer_csv.writeheader()
                writer_csv.writerow(profile)

            # Optionally log other metrics to TensorBoard every 100 steps.
            if (global_step % 100 == 0) and (global_step > args.learning_starts):
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item() if 'actor_loss' in locals() else 0, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                print("SPS:", sps)

        envs.close()
        writer.close()
