import os
import sys
import time
import argparse

import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FlattenObservation
import numpy as np
import matplotlib.pyplot as plt

from sac_torch import Agent
sys.path.append("./")
from CarlaDrivingEnv import CarlaDrivingEnv


def plot_learning_curve(scores, filename):
    x = [i + 1 for i in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    plt.plot(x, running_avg)
    plt.xlabel("Episode")
    plt.ylabel("Running Average Score")
    plt.title("Training Performance")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this run; will create tmp/sac/<run-name>')
    parser.add_argument('--chkpt-dir', type=str, default=None,
                        help='Custom checkpoint directory to save/load (overrides run-name)')
    parser.add_argument('--load-run', type=str, default=None,
                        help='Name of a run under tmp/sac to load from (does not change save dir)')
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='If set, attempt to load models from the checkpoint directory')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Reward scaling factor passed to Agent')
    parser.add_argument('--start-steps', type=int, default=2000,
                        help='Number of initial environment steps to take random actions')
    parser.add_argument('--render', action='store_true',
                        help='Enable CARLA spectator rendering')
    args = parser.parse_args()

    env = CarlaDrivingEnv()

    if args.chkpt_dir:
        chkpt_dir = args.chkpt_dir
    elif args.run_name:
        chkpt_dir = os.path.join('tmp', 'sac', args.run_name)
    else:
        run_id = int(time.time())
        chkpt_dir = os.path.join('tmp', 'sac', f'run_{run_id}')

    os.makedirs(chkpt_dir, exist_ok=True)

    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        reward_scale=args.reward_scale,
        chkpt_dir=chkpt_dir,
    )

    n_games = 1500
    figure_file = os.path.join("plots", "finalplease.png")

    best_score = -np.inf
    score_history = []

    autosave_interval = 50

    if args.load_run:
        load_dir = os.path.join('tmp', 'sac', args.load_run)
        print(f"Loading checkpoints from run dir: {load_dir}")
        if os.path.isdir(load_dir):
            agent.load_models_from(
                load_dir,
                load_actor=True,
                load_critics=True,
                load_alpha=False,
                load_buffer=False,
            )
        else:
            print(f"Error: load run directory does not exist: {load_dir}")

    elif args.load_checkpoint:
        print("Loading checkpoint from save directory...")
        try:
            agent.load_models(load_alpha=False, load_buffer=False)
        except Exception as e:
            print(f"Warning: failed to load checkpoints from {chkpt_dir}: {e}")

    print(f"Checkpoints for this run will be in: {chkpt_dir}")

    total_steps = 0
    start_steps = int(args.start_steps)

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0.0

        episode_reward_breakdown = {
            "time_penalty": 0.0,
            "progress": 0.0,
            "speed_bonus": 0.0,
            "align_reward": 0.0,
            "standstill_penalty": 0.0,
            "steer_penalty": 0.0,
            "waypoint_bonus": 0.0,
            "goal_bonus": 0.0,
            "collision_penalty": 0.0,
            "wrong_way": 0.0,
            "route_align": 0.0,
            "route_recover": 0.0,
            "heading_fix": 0.0,
            "stuck_bad_heading": 0.0,
        }

        step_actor_losses = []
        step_critic_losses = []
        step_alpha_losses = []
        step_alphas = []

        while not done:
            if args.render:
                env.render()

            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(observation, evaluate=False)

            observation_, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            for k, v in info["reward_breakdown"].items():
                episode_reward_breakdown.setdefault(k, 0.0)
                episode_reward_breakdown[k] += float(v)

            score += float(reward)
            done = bool(terminated or truncated)

            agent.remember(observation, action, reward, observation_, bool(terminated))

            losses = agent.learn()
            if losses is not None:
                step_actor_losses.append(losses.get("actor_loss", 0.0))
                step_critic_losses.append(losses.get("critic_loss", 0.0))
                step_alpha_losses.append(losses.get("alpha_loss", 0.0))
                step_alphas.append(losses.get("alpha", 0.0))

            observation = observation_

        score_history.append(score)
        avg_score = float(np.mean(score_history[-100:]))

        avg_actor_loss = float(np.mean(step_actor_losses)) if step_actor_losses else 0.0
        avg_critic_loss = float(np.mean(step_critic_losses)) if step_critic_losses else 0.0
        avg_alpha = float(np.mean(step_alphas)) if step_alphas else float(agent.log_alpha.exp().item())
        avg_alpha_loss = float(np.mean(step_alpha_losses)) if step_alpha_losses else 0.0

        steps_this_ep = env.current_step

        print(
            f"episode {i} score {score:.1f} avg_score {avg_score:.1f} steps {steps_this_ep} "
            f"act_loss {avg_actor_loss:.4f} critic_loss {avg_critic_loss:.4f} "
            f"alpha {avg_alpha:.6f} alpha_loss {avg_alpha_loss:.6f}"
        )

        print("episode reward totals:", episode_reward_breakdown)

        if avg_score > best_score:
            best_score = avg_score
            print("Saving improved model...")
            agent.save_models()

        if (i % autosave_interval) == 0:
            agent.save_models()
            print("Autosaved.")

    plot_learning_curve(score_history, figure_file)