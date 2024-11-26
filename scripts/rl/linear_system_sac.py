import datetime
import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seal.examples.linear_system import LinearSystemMPC, LinearSystemOcpEnv
from seal.mpc import MPC, MPCParameter, MPCState
from seal.nn.modules import MPCSolutionModule
from seal.rl.replay_buffer import ReplayBuffer
from seal.rl.sac_modules import SACActor, SACConfig, SACQNet, SACTrainer
from seal.torch_utils import TanhNormal, create_mlp, string_to_activation
from seal.util import create_dir_if_not_exists, tensor_to_numpy


class FOUMPCNetwork(nn.Module):
    """
    Uses a neural network to predict PARAMETERS(!) of the MPC solver and a standard deviation for the ACTION(!) of the MPC.
    The MPC solver is then used to calculate the action which is used as mean in a tanh normal distribution,
    together with the standard deviation from the neural network.
    This actor is trained by differentiating through the MPC (by using sensitivities),
    hence treating the MPC as kind of "layer".
    NOTE: All learnable parameter must be global for this architecture to work, such that we can calculate the sensitivities for them.
    Also see the MPCSolutionModule documentation.
    """

    def __init__(
        self,
        mpc: MPC,
        param_mean_action_std_model: nn.Module,
        param_factor: np.ndarray,
        param_shift: np.ndarray,
        minimal_std: float = 1e-3,
    ):
        """
        Parameters:
            mpc: The MPC to be used for solving the OCP.
            param_mean_action_std_model: A model that predicts the parameters for the MPC and a standard deviation for the actions of the MPC.
                Possibly also outputs a stats dict (see forward).
            param_factor: A factor to scale the parameters, before putting them into the MPC.
            param_shift: A shift to add to the scaled parameters, before putting them into the MPC.
            minimal_std: The minimal standard deviation of the action distribution.
        """
        super().__init__()
        self.param_mean_action_std_model = param_mean_action_std_model

        self.mpc_layer = MPCSolutionModule(mpc)
        self.tanh_normal = TanhNormal(minimal_std=minimal_std)

        if not param_factor.shape == param_shift.shape:
            raise ValueError("param_scaling and param_shift must have the same shape.")

        self.param_factor = nn.Parameter(
            torch.tensor(param_factor), requires_grad=False
        )
        self.param_shift = nn.Parameter(torch.tensor(param_shift), requires_grad=False)

    def forward(
        self,
        obs: tuple[torch.Tensor, MPCParameter],
        param_transform_kwargs: dict | None = None,
        mpc_initialization: list[MPCState] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Parameters:
            obs: A tuple containing the input to the mean/std model and the stagewise Parameters to set in the MPC before solving.
                NOTE: Despite being of the class MPCParameter, if p_global is not None,
                it will be overridden, because p_global will be set by the prediction from the network.
            param_transform_kwargs: The keyword arguments for the final_param_transform_before_mpc (useful if overwritten).
            initialization: The initialization for the MPC. If not None, it should be a list of length batch_size.
            deterministic: If True, the output will just be tanh(MPC(mean)), no sampling is taking place.

        Returns the action, the log probability of the action, the status of the MPC and some statistics.
        """
        x, param = obs
        p_stagewise = MPCParameter(
            p_stagewise=param.p_stagewise.astype(np.float64)  # type:ignore
            if param.p_stagewise is not None
            else None,
            p_stagewise_sparse_idx=param.p_stagewise_sparse_idx,
        )
        net_out = self.param_mean_action_std_model(obs)
        if isinstance(net_out[-1], dict):
            stats = net_out[-1]
            net_out = net_out[:-1]
        else:
            stats = dict()
        params_unscaled, action_std = net_out

        param_transform_kwargs = (
            dict() if param_transform_kwargs is None else param_transform_kwargs
        )
        params_scaled = self.final_param_transform_before_mpc(
            params_unscaled, x, **param_transform_kwargs
        )
        stats["params_input_to_mpc"] = params_scaled

        action_mean, _, mpc_status = self.mpc_layer(
            x,
            p_global=params_scaled,
            p_stagewise=p_stagewise,
            initializations=mpc_initialization,
        )
        if torch.any(mpc_status):
            print("WARNING: Status != 0 encountered.")
        stats["mean_from_mpc"] = action_mean
        stats["std_from_net"] = action_std

        real_action, real_log_prob = self.tanh_normal(
            action_mean, action_std, deterministic=deterministic
        )
        # NOTE: One could use a TruncatedNormal instead of a TanhNormal to retain the constraint satisfaction of the MPC at test-time (deterministic mode),
        # but this results in much more training needed, probably due to bad gradients?
        # We did not check if this is also a problem if we just use a "differentiable clamp" in the manner described below to clamp the Normal distribution.
        # This could be another way to retain the constraint satisfaction of the MPC at test-time.
        # Scale the log prob for the clamp
        # if action < -Fmax:
        #     real_log_prob = torch.log(dist.cdf(torch.Tensor(-Fmax, device=self.device)))
        # elif action > Fmax:
        #     real_log_prob = torch.log(1 - dist.cdf(torch.Tensor(Fmax, device=self.device)))
        # else:
        #     real_log_prob = log_prob
        # real_action = dclamp(action, -self.main_mpc.Fmax, self.main_mpc.Fmax) / Fmax # Scale it to [-1, 1]

        # return_ac = real_action / Fmax  # Scale it to [-1, 1]

        return real_action, real_log_prob, mpc_status, stats

    def final_param_transform_before_mpc(
        self, params_unscaled: torch.Tensor, state: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """The final transformation of the parameters before they are used in the MPC.
        Amounts to a tanh transformation to squash the parameters between [-1, 1] followed by a linear transformation.
        This can be overridden in subclasses to fit your needs, but it is expected to be differentiable (meaning torch.autograd)!

        Parameters:
            params: The parameters from the neural network.
            state: The state of the system, e.g., needed for using deltas (param = x + delta) instead of explicitly predicting the parameters.
        """
        return F.tanh(params_unscaled) * self.param_factor + self.param_shift


class TanhNormalNetwork(nn.Module):
    """Uses a Network to predict the mean and standard deviation of a TanhNormal distribution."""

    def __init__(
        self,
        mean_std_module: nn.Module,
        minimal_std: float = 1e-3,
    ):
        super().__init__()
        self.mean_std_module = mean_std_module
        self.tanh_normal = TanhNormal(minimal_std=minimal_std)

    def forward(
        self, obs: tuple[torch.Tensor, MPCParameter], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            obs: The input to the mean/std model.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            an output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        mean, std = self.mean_std_module(obs)
        return self.tanh_normal(mean, std, deterministic=deterministic)


class MeanStdMLP(nn.Module):
    """An MLP with a little syntactic sugar (being the splitting of the output in the correct dimensions),
    ment to predict the mean and standard deviation of a distribution."""

    def __init__(
        self,
        s_dim: int,
        mean_dim: int,
        std_dim: int,
        hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        self.mean_dim = mean_dim
        self.std_dim = std_dim
        self.mlp = create_mlp(s_dim, mean_dim + std_dim, hidden_dims, activation, False)

    def forward(
        self, obs: tuple[torch.Tensor, MPCParameter]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NOTE: Ignores MPCParameter"""
        return torch.split(self.mlp(obs[0]), [self.mean_dim, self.std_dim], dim=-1)  # type:ignore


class LinearWithActivation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = string_to_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


class StateEmbed(LinearWithActivation):
    def forward(self, obs: tuple[torch.Tensor, MPCParameter]) -> torch.Tensor:
        """NOTE: Ignores the parameters."""
        state, params = obs
        return self.activation(self.fc(state))


class Scenario(Enum):
    STANDARD_SAC = 0
    FO_U_SAC = 1


@dataclass(kw_only=True)
class LinearSystemSACConfig(SACConfig):
    # General
    name: str

    # env
    max_time: float
    dt: float
    a_dim: int
    s_dim: int
    param_dim: int
    # MPC-specific
    accept_n_nonconvergences_per_batch: int

    # Networks, i.e., mlps
    hidden_dims: list[int]
    q_embed_size: int  # should be half of hidden size
    activation: str

    # Buffer
    dtype_buffer: torch.dtype


def create_qnet(config: LinearSystemSACConfig) -> SACQNet:
    s_embed = StateEmbed(config.s_dim, config.q_embed_size, config.activation)
    a_embed = LinearWithActivation(config.a_dim, config.q_embed_size, config.activation)
    embed_to_q = create_mlp(
        input_dim=config.hidden_dims[0],
        output_dim=1,
        hidden_dims=config.hidden_dims[1:],
        activation=config.activation,
        use_activation_on_output=False,
    )
    return SACQNet(
        state_embed=s_embed, action_embed=a_embed, embed_to_q=embed_to_q, tau=config.tau
    )


class LinearSystemSACTrainer(SACTrainer):
    def act(
        self, obs: tuple[np.ndarray, MPCParameter], deterministic: bool = False
    ) -> np.ndarray:
        state, params = obs
        state_tensor = torch.tensor(
            state, dtype=self.replay_buffer.obs_dtype, device=self.device
        )
        return tensor_to_numpy(
            self.actor((state_tensor, params), deterministic=deterministic)[0]
        )


def run_linear_system_sac(
    scenario: Scenario, device: str, seed: int, savefile_path: str
):
    """Run the linear system SAC.
    Parameters:
        scenario: The scenario to run.
        device: The device to run on.
        seed: The seed for reproducibility.
        savefile_path: The path where the models (networks) will be saved.
    """
    params = {
        "A": np.array([[1.0, 0.25], [0.0, 1.0]]),
        "B": np.array([[0.03125], [0.25]]),
        "Q": np.identity(2),
        "R": np.identity(1),
        "b": np.array([[0.0], [0.0]]),
        "f": np.array([[0.0], [0.0], [0.0]]),
        "V_0": np.array([1e-3]),
    }
    learnable_params = ["b"]
    dt = 0.1
    max_time = 10.0  # => 100 steps
    discount_factor = 0.99
    batch_size = 64
    mpc = LinearSystemMPC(
        params=params,
        learnable_params=learnable_params,
        discount_factor=discount_factor,
        n_batch=batch_size,
    )
    env = LinearSystemOcpEnv(mpc, dt=dt, max_time=max_time)
    param_dim = env.p_learnable_space.shape[0]  # type:ignore

    a_space = env.action_space
    a_dim = a_space.shape[0]  # type:ignore
    s_space = env.state_space
    s_dim = s_space.shape[0]  # type:ignore
    target_entropy = -a_dim

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savefile_directory_path = os.path.join(
        savefile_path, scenario.name + "_" + timestamp
    )
    create_dir_if_not_exists(savefile_directory_path)

    param_factor = np.array([1.0, 1.0], dtype=np.float32)
    param_shift = np.array([0.0, 0.0], dtype=np.float32)
    config = LinearSystemSACConfig(
        # General and training loop
        name=scenario.name,
        seed=seed,
        device=device,
        max_episodes=1000,
        training_steps_per_episode=20,
        dont_train_until_this_many_transitions=1000,
        val_interval=50,
        n_val_rollouts=5,
        max_eps_length=100,  # TODO: Usually this should be in the gym, right?
        save_interval=100,
        save_directory_path=savefile_directory_path,
        no_grad_during_rollout=False,
        crude_memory_debugging=False,
        # SACTrainer
        actor_lr=5 * 1e-4,
        critic_lr=5 * 1e-4,
        entropy_scaling_lr=1e-3,
        discount_factor=discount_factor,
        batch_size=batch_size,
        # Replay Buffer
        replay_buffer_size=10000,
        dtype_buffer=torch.float32,
        # Actor
        init_entropy_scaling=0.01,
        target_entropy=target_entropy,
        minimal_std=1e-3,  # Will be used in every TanhNormal to avoid collapse of the distribution
        param_factor=param_factor,  # Only in Actor with MPC
        param_shift=param_shift,  # Only in Actor with MPC
        # Critic
        tau=1e-2,  # For target network soft update
        soft_update_frequency=1,
        # Environment
        max_time=max_time,
        dt=dt,
        a_dim=a_dim,
        s_dim=s_dim,
        param_dim=param_dim,
        # MPC-specific
        accept_n_nonconvergences_per_batch=0,
        # Networks
        q_embed_size=64,  # should be half of hidden size
        hidden_dims=[128, 128],
        activation="leaky_relu",
    )

    if scenario.value == Scenario.STANDARD_SAC.value:
        mlp = MeanStdMLP(
            s_dim=config.s_dim,
            mean_dim=config.a_dim,
            std_dim=config.a_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )
        actor_net = TanhNormalNetwork(
            mean_std_module=mlp, minimal_std=config.minimal_std
        )
        actor = SACActor(
            obs_to_action_module=actor_net,
            module_contains_mpc=False,
            init_entropy_scaling=config.init_entropy_scaling,
            target_entropy=config.target_entropy,
        )

    elif scenario.value == Scenario.FO_U_SAC.value:
        mlp = MeanStdMLP(
            s_dim=config.s_dim,
            mean_dim=config.param_dim,
            std_dim=config.a_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        )
        actor_net = FOUMPCNetwork(
            mpc=mpc,
            param_mean_action_std_model=mlp,
            param_factor=config.param_factor,
            param_shift=config.param_shift,
            minimal_std=config.minimal_std,
        )
        actor = SACActor(
            obs_to_action_module=actor_net,
            module_contains_mpc=True,
            init_entropy_scaling=config.init_entropy_scaling,
            target_entropy=config.target_entropy,
            accept_n_nonconvergences_per_batch=config.accept_n_nonconvergences_per_batch,
            num_batch_dimensions=1,
        )

    if not config.q_embed_size * 2 == config.hidden_dims[0]:
        raise ValueError("The q_embed_size should be half of the hidden size.")
    critic1 = create_qnet(config)
    critic2 = create_qnet(config)
    critic1_target = create_qnet(config)
    critic2_target = create_qnet(config)
    buffer = ReplayBuffer(
        buffer_limit=config.replay_buffer_size,
        device=config.device,
        obs_dtype=config.dtype_buffer,
    )  # type:ignore
    trainer = LinearSystemSACTrainer(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        critic1_target=critic1_target,
        critic2_target=critic2_target,
        replay_buffer=buffer,
        config=config,
    )
    trainer.training_loop(env, config)


if __name__ == "__main__":
    scenario = Scenario.FO_U_SAC
    device = "cuda:5"
    seed = 1337

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    savefile_directory_path = os.path.join(os.getcwd(), "output")
    create_dir_if_not_exists(savefile_directory_path)
    savefile_directory_path = os.path.join(savefile_directory_path, "linear_system_sac")
    create_dir_if_not_exists(savefile_directory_path)

    run_linear_system_sac(scenario, device, seed, savefile_directory_path)
