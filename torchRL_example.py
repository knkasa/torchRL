# https://medium.com/chat-gpt-now-writes-all-my-articles/easy-reinforcement-learning-with-brand-new-torchrl-code-example-included-43cc40983c1a

"""
!pip3 install torchrl
!pip3 install gym[mujoco]
!pip3 install tqdm
"""

import torch
import multiprocessing

is_fork = multiprocessing.get_start_method() == "fork"
device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 1000
total_frames = 10_000
sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


from torchrl.envs import GymEnv, TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter

base_env = GymEnv("InvertedDoublePendulum-v4", device=device)
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
    )
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
        },
    return_log_prob=True,
    )
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
    )
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
    )

from torchrl.collectors import SyncDataCollector

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    )

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
    )

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )
loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
    )
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
    )

from collections import defaultdict
from tqdm import tqdm

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""
for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
                )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()
    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # Evaluation code here
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(1000, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                        )
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {logs['eval step_count'][-1]}"
                        )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        
            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()