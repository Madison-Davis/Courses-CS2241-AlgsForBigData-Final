# sac_torch.py


# docs and experiment results can be found at 
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy


# ++++++++++++ Imports and Installs ++++++++++++ #
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import ogbench
import wandb
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyTensorStorage
from torchrl.data import TensorDictPrioritizedReplayBuffer

# ++++++++++++++ Global Variables ++++++++++++++ #
LOG_STD_MAX = 2
LOG_STD_MIN = -5
MAX_TD_ERROR = 10


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
    # NOTE: Change from Hopper
    env_id: str = "humanoidmaze-large-navigate-v0" # "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    # learning_starts: int = 5e3
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
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
                        (
                                env.single_action_space.high - env.single_action_space.low) / 2.0,
                        dtype=torch.float32,
                ),
        )
        self.register_buffer(
                "action_bias",
                torch.tensor(
                        (
                                env.single_action_space.high + env.single_action_space.low) / 2.0,
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                log_std + 1)
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
            env, _, _ = ogbench.make_env_and_datasets(env_id)
            # NOTE: Change from Hopper
            # We're not capturing a video in our case; if we were going to, then yes, add a RecordVideo func
            #env = gym.make(env_id, render_mode="rgb_array")
            #env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env, _, _ = ogbench.make_env_and_datasets(env_id)
            # NOTE: Change from Hopper
            #env = gym.make(env_id, render_mode="rgb_array")
        # OG Bench returns gymnasium environments so gym wrappers should work
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# +++++++++++++++++ Main Function ++++++++++++++++ #
if __name__ == "__main__":

    # Setup: dependencies
    if sb3.__version__ < "2.0":
        raise ValueError(
                """Ongoing migration: run the following command to install the new dependencies:
                poetry run pip install "stable_baselines3==2.0.0a1"
                """
        )

    # Setup: plotting-printint stuff
    args = tyro.cli(Args)
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
    writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join(
                    [f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup: data seeding and device to run it on (cuda/gpu, cpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Setup: env
    envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video,
                      run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space,
                      gym.spaces.Box), "only continuous action space is supported"
    max_action = float(envs.single_action_space.high[0])

    # Setup: instantiate actors, Q-networks, and optimizers, and entropy tuning
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
                torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # NOTE [CHANGE]
    rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=0.9,
                                           storage=LazyTensorStorage(
                                                   args.buffer_size),
                                           priority_key="td_error")

    start_time = time.time()

    # NOTE: Try not to change this
    # Start the game with a random seed of values
    obs, _ = envs.reset(seed=args.seed)
    # For the total # of timesteps...
    for global_step in range(args.total_timesteps):
        # Determine the actions we can do
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in
                                range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        # Take one step based on the actions we can do
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # Record rewards for plotting purposes
        if 'episode' in infos:
            print(
                    f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"],
                              global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"],
                              global_step)
        # Save data to reply buffer (rb); handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
                else:
                    real_next_obs[idx] = next_obs[idx]

        # NOTE: things are plural because the env is vectorized
        # vectorized: use batching to kind of help 'speed up' the process of learning where surprising things are
        # obs = curr state
        # real_next_obs = next state
        # actions = actions we had to get to real_next_obs
        # rewards = reward of each action in each vectorized env
        # terminations = did the envs terminate (vectorized)
        # infos = if things timeout, give info on how to handle it (vectorized)

        with torch.no_grad():
            # Convert numpy arrays to tensors for network computation
            obs_tensor = torch.FloatTensor(obs).to(device)
            actions_tensor = torch.FloatTensor(actions).to(device)
            next_obs_tensor = torch.FloatTensor(real_next_obs).to(device)
            rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(
                    device)
            terminations_tensor = torch.FloatTensor(terminations).reshape(-1,
                                                                          1).to(
                    device)

            # Get next state values
            next_actions, next_state_log_pi, _ = actor.get_action(
                    next_obs_tensor)
            qf1_next_target = qf1_target(next_obs_tensor, next_actions)
            qf2_next_target = qf2_target(next_obs_tensor, next_actions)
            min_qf_next_target = torch.min(qf1_next_target,
                                           qf2_next_target) - alpha * next_state_log_pi
            next_q_value = rewards_tensor + (
                    1 - terminations_tensor) * args.gamma * (
                                   min_qf_next_target).view(-1)
            qf1_current = qf1(obs_tensor, actions_tensor)
            qf2_current = qf2(obs_tensor, actions_tensor)

            # TD errors
            td_error1 = torch.abs(qf1_current - next_q_value)
            td_error2 = torch.abs(qf2_current - next_q_value)
            td_errors = torch.max(td_error1, td_error2)

        # Now create the transition with these TD errors
        transition = TensorDict({
                "observations": obs_tensor,
                "next_observations": next_obs_tensor,
                "actions": actions_tensor,
                "rewards": rewards_tensor,
                "dones": terminations_tensor,
                "td_error": td_errors
        }, [obs.shape[0]])

        rb.extend(transition)

        # Change 'state' to the next state we go to
        obs = next_obs

        # Train algorithm based on what we have in replay buffer (rb)
        # NOTE: double Q-learning, technique to improve stability
        # Take min value outputted by two Q-values, inhibits overlearning
        if global_step > args.learning_starts:
            data, info = rb.sample(batch_size=args.batch_size, return_info=True)

            # Move relevant tensors to device
            data = data.to(device)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                        data['next_observations'])
                qf1_next_target = qf1_target(data['next_observations'],
                                             next_state_actions)
                qf2_next_target = qf2_target(data['next_observations'],
                                             next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target,
                                               qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data['rewards'].flatten() + (
                        1 - data['dones'].flatten()) * args.gamma * (
                                       min_qf_next_target).view(-1)

            qf1_a_values = qf1(data['observations'], data['actions']).view(-1)
            qf2_a_values = qf2(data['observations'], data['actions']).view(-1)

            # ORIGINAL BASELINE
            # qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            # qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

            # NOTE [CHANGE]
            # Get importance sampling weights
            weights = info['_weight'].to(device)

            # Calculate losses with importance sampling (keeping individual errors)
            qf1_squared_errors = ((qf1_a_values - next_q_value) ** 2)
            qf2_squared_errors = ((qf2_a_values - next_q_value) ** 2)

            # Apply weights and take mean to get scalar losses
            qf1_loss = (
                    weights * qf1_squared_errors).mean()  # This gives a scalar
            qf2_loss = (
                    weights * qf2_squared_errors).mean()  # This gives a scalar
            qf_loss = qf1_loss + qf2_loss

            # Compute the largest TD error (largest surprise) and update the priorities
            td_errors = torch.max(torch.abs(qf1_a_values - next_q_value),
                                  torch.abs(qf2_a_values - next_q_value))

            data.set('td_error', td_errors)

            # Optimize the model by 'sandwiching double Q-learner' with optimizers
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            rb.update_tensordict_priority(data)

            # TD 3 Delayed update support
            # compensate for the delay by doing 'actor_update_interval' instead of 1
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
                            _, log_pi, _ = actor.get_action(
                                    data['observations'])
                        alpha_loss = (-log_alpha.exp() * (
                                log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(),
                                               qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (
                            1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(),
                                               qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (
                            1 - args.tau) * target_param.data)

            # Write some stuff (for plotting purposes)
            if global_step % 100 == 0:
                # log the max td error
                writer.add_scalar("losses/max_td_error",
                                  torch.max(td_errors).item(), global_step)
                writer.add_scalar("losses/qf1_values",
                                  qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values",
                                  qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(),
                                  global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(),
                                  global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0,
                                  global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(),
                                  global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS",
                                  int(global_step / (time.time() - start_time)),
                                  global_step, )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(),
                                      global_step)

    envs.close()
    writer.close()
