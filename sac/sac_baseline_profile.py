#!/usr/bin/env python
"""
sac_baseline_profiled.py

This version of the SAC experiment logs detailed latency and memory breakdowns
for every RL step. When a training update is performed (when global_step % args.policy_frequency == 0
and global_step > args.learning_starts), the following portions are measured:

  - Action selection
  - Environment step
  - Replay buffer addition
  - Replay buffer sampling
  - Target and loss calculation
  - Backpropagation and network update

When no training update happens, only the first three steps are measured (and training update columns are left blank).

After each step, the following memory metrics (in bytes) are logged:

  • mem_nn: The memory footprint of the neural networks (actor, qf1, and qf2).
  • mem_env: The memory footprint of the environment.
  • mem_rb: The replay buffer memory, computed as
             min(global_step, buffer_size) * 205.
  
For training steps only, the following extra memory metrics are logged:
  • mem_opt: The combined memory footprint of the optimizers.
  • mem_act: The activations memory footprint (here a placeholder).
  • mem_grad: The total memory (bytes) used by stored gradients in all networks.

Also, the mean reward (averaged over the vectorized env) is logged as "reward."

Data from all runs (each with 30,000 steps) is appended to "profile_log.csv".
"""

import os
import time
import csv
import random
import psutil
import sys
import tyro
import wandb
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
import stable_baselines3 as sb3
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

# Global definitions
LOG_STD_MAX = 2
LOG_STD_MIN = -5
printed_once = False  # only to print parameter info on the first step

# --- Memory Helper Functions ---
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
    """Return the memory footprint (in bytes) of the replay buffer.
       It is computed as min(global_step, buffer_size) * 205."""
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

# --- Class Definitions ---
@dataclass
class Args:
    # General Arguments
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    # Algorithm-Specific Arguments
    env_id: str = "Hopper-v5"
    total_timesteps: int = 20000   # updated to 30,000 steps
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

# --- Neural Network Definitions ---
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() +
            np.prod(env.single_action_space.shape), 256)
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
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

# --- Helper Functions ---
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

def print_param_info(name, param):
    print(f"Parameter: {name}")
    print(f"  Type: {type(param)}")
    if isinstance(param, np.ndarray):
        print(f"  Numpy array with dtype: {param.dtype}, shape: {param.shape}")
        print(f"  Total bytes: {param.nbytes}, Total bits: {param.nbytes * 8}")
    elif isinstance(param, torch.Tensor):
        print(f"  Torch tensor with dtype: {param.dtype}, shape: {tuple(param.shape)}")
        total_bytes = param.element_size() * param.nelement()
        print(f"  Total bytes: {total_bytes}, Total bits: {total_bytes * 8}")
    else:
        size_bytes = sys.getsizeof(param)
        print(f"  sys.getsizeof: {size_bytes} bytes, {size_bytes * 8} bits")
    print()

# --- Main Execution ---
if __name__ == "__main__":
    if sb3.__version__ < "2.0":
        raise ValueError("Please install stable_baselines3 v2.0.0a1 or later")

    args = tyro.cli(Args)
    # Use the specified total_timesteps (set to 30,000 for this experiment)
    args.total_timesteps = 20000

    # List of seeds for runs (using 5 seeds)
    seeds = [1, 2, 3]

    # Remove existing CSV file if present
    profile_filename = "profile_log.csv"
    if os.path.exists(profile_filename):
        os.remove(profile_filename)

    # CSV header fields (reward added)
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
        writer = SummaryWriter(f"../eval/runs/{run_name}")
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))

        # Set random seeds and device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
             for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space supported"
        max_action = float(envs.single_action_space.high[0])

        # Instantiate networks and optimizers
        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
        if args.autotune:
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        obs, _ = envs.reset(seed=args.seed)
        # Main loop over steps
        for global_step in range(args.total_timesteps):
            profile = {'seed': current_seed, 'step': global_step}
            # Determine if this step includes a training update.
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

            # Log average reward for this step.
            profile['reward'] = np.mean(rewards)

            if 'episode' in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            # --- 3. Replay Buffer Addition (latency measured) ---
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            t0 = time.time()
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            profile['latency_rb_add'] = time.time() - t0

            # Update state for next iteration.
            obs = next_obs

            # For training updates, log additional latencies.
            if is_train_update:
                t0 = time.time()
                data = rb.sample(args.batch_size)
                profile['latency_rb_sample'] = time.time() - t0

                t0 = time.time()
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                profile['latency_target_loss'] = time.time() - t0

                t0 = time.time()
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                # Delayed policy (actor) update
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
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
                profile['latency_train_net'] = time.time() - t0
            else:
                # For steps without training update, leave these columns blank.
                profile['latency_rb_sample'] = ""
                profile['latency_target_loss'] = ""
                profile['latency_train_net'] = ""

            # Total latency for this step.
            total_latency = (profile['latency_action'] + profile['latency_env'] +
                             profile['latency_rb_add'] +
                             (profile['latency_rb_sample'] if isinstance(profile['latency_rb_sample'], float) else 0) +
                             (profile['latency_target_loss'] if isinstance(profile['latency_target_loss'], float) else 0) +
                             (profile['latency_train_net'] if isinstance(profile['latency_train_net'], float) else 0))
            profile['total_latency'] = total_latency

            # --- Memory Logging (new breakdown) ---
            profile['mem_nn'] = get_nn_memory()
            profile['mem_env'] = get_env_memory(envs)
            profile['mem_rb'] = get_rb_memory(global_step, args.buffer_size)
            if is_train_update:
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

            # Append the profile row to the CSV file.
            with open(profile_filename, "a", newline="") as csvfile:
                writer_csv = csv.DictWriter(csvfile, fieldnames=csv_fields)
                if csvfile.tell() == 0:
                    writer_csv.writeheader()
                writer_csv.writerow(profile)

            # Optionally log losses and SPS to TensorBoard every 100 steps.
            if (global_step % 100 == 0) and (global_step > args.learning_starts) and is_train_update:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                print("SPS:", sps)

        envs.close()
        writer.close()
