import datetime as dt
import os
from enum import Enum

import numpy as np

from seal.examples.linear_system import LinearSystemMPC, LinearSystemOcpEnv
from seal.ocp_env import ConstantParamCreator, MPCInput
from seal.rl.critics_base import QNetMLP
from seal.rl.replay_buffer import ReplayBuffer
from seal.rl.run import training_loop
from seal.rl.sac_modules import SACActorFOMPCWithActionNoise, SACActorMLP, SACTrainer
from seal.util import create_dir_if_not_exists


class Scenario(Enum):
    STANDARD_SAC = 0
    FO_U_SAC = 1


def get_action_scaling_from_a_space(a_space):
    a_mean = (a_space.high + a_space.low) / 2
    a_scale = (a_space.high - a_space.low) / 2

    def action_scaling(action):
        return a_scale * action + a_mean

    return action_scaling


def get_target_entropy_from_env(env):
    a_space = env.action_space
    a_dim = a_space.shape[0]
    target_entropy = -a_dim
    return target_entropy


def map_policy_to_env(u: np.ndarray) -> np.ndarray:
    """The policies output actions between -1 and 1, and the environment expects actions between -1 and 1."""
    return u


def map_ocp_env_output_to_trainer_input(ocp_output: MPCInput) -> np.ndarray:
    """Maps MPCInput to state."""
    return ocp_output[0]


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
    mpc = LinearSystemMPC(params=params, learnable_params=learnable_params)
    param_dim = 0
    for param in learnable_params:
        param_dim += params[param].size
    params.pop("Q")
    params.pop("R")
    default_params = np.concatenate([param.flatten() for param in params.values()])
    param_creator = ConstantParamCreator(default_params)
    env = LinearSystemOcpEnv(mpc, param_creator)

    a_space = env.action_space
    a_dim = a_space.shape[0]  # type:ignore
    s_space = env.state_space
    s_dim = s_space.shape[0]  # type:ignore
    target_entropy = -a_dim

    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    savefile_directory_path = os.path.join(
        savefile_path, scenario.name + "_" + timestamp
    )
    create_dir_if_not_exists(savefile_directory_path)

    config = dict(
        name=scenario.name,
        seed=seed,
        learning_rate=5 * 1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_limit=10000,
        max_episodes=1000,
        training_steps_per_episode=20,
        max_eps_length=100,  # TODO: Usually this should be in the gym, right?
        dont_train_until_this_many_transitions=1000,
        val_interval=50,
        n_val_rollouts=5,
        lr_alpha=0.001,  # for automated alpha update
        init_alpha=0.01,
        tau=0.01,  # for target network soft update
        target_entropy=target_entropy,
        a_dim=a_dim,
        s_dim=s_dim,
        param_dim=param_dim,
        q_embed_size=64,  # should be half of hidden size
        hidden_dims=[128, 128],
        activation="leaky_relu",
        soft_update_frequency=1,
        device=device,
        accept_n_nonconvergences_per_batch=0,
        save_interval=100,
        save_directory_path=savefile_directory_path,
    )

    if scenario.value == Scenario.STANDARD_SAC.value:
        actor = SACActorMLP(**config)  # type:ignore
    elif scenario.value == Scenario.FO_U_SAC:
        actor = SACActorFOMPCWithActionNoise(mpc=mpc, **config)  # type:ignore

    critic1 = QNetMLP(**config)  # type:ignore
    critic2 = QNetMLP(**config)  # type:ignore
    critic1_target = QNetMLP(**config)  # type:ignore
    critic2_target = QNetMLP(**config)  # type:ignore
    buffer = ReplayBuffer(buffer_limit=config["buffer_limit"], device=device)  # type:ignore
    trainer = SACTrainer(
        actor,
        critic1,
        critic2,
        critic1_target,
        critic2_target,
        buffer,
        config["soft_update_frequency"],  # type:ignore
        config["gamma"],  # type:ignore
        config["batch_size"],  # type:ignore
        device,
    )

    training_loop(
        ocp_env=env,
        trainer=trainer,
        map_policy_to_env=map_policy_to_env,
        map_ocp_env_state_to_trainer_input=map_ocp_env_output_to_trainer_input,
        **config,  # type:ignore
    )


if __name__ == "__main__":
    scenario = Scenario.STANDARD_SAC
    device = "cuda:5"
    seed = 1337

    savefile_directory_path = os.path.join(os.getcwd(), "output")
    create_dir_if_not_exists(savefile_directory_path)
    savefile_directory_path = os.path.join(savefile_directory_path, "linear_system_sac")
    create_dir_if_not_exists(savefile_directory_path)

    run_linear_system_sac(scenario, device, seed, savefile_directory_path)
