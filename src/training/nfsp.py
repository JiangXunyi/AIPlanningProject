import os
import itertools
import copy
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.agents import RandomAgent, NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(hidden_layer_sizes, q_mlp_layers, learning_rate, discount_factor, exploration_rate):
    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(42)

    # Make the environment with seed
    env = rlcard.make(
        "limit-holdem",
        config={
            'seed': 42,
        }
    )

    agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=hidden_layer_sizes,
                q_mlp_layers=q_mlp_layers,
                device=device,
                rl_learning_rate=learning_rate,
                q_discount_factor=discount_factor,
                q_epsilon_start=exploration_rate
            )
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    losses = []

    max_performance = float('-inf')
    best_agent = None
    # Start training
    with Logger("experiments/limit_holdem_nfsp_result") as logger:
        for episode in range(5000):

            agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                loss = agent.feed(ts)
                losses.append(loss)

            # Evaluate the performance. Play with random agents.
            if episode % 50 == 0:
                performance = tournament(env, 1000)[0]
                logger.log_performance(episode, performance)

                # Update the best performing agent
                if performance > max_performance:
                    max_performance = performance
                    best_agent = copy.deepcopy(agent)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        loss_file_path = os.path.join("experiments/limit_holdem_nfsp_result", 'losses.txt')
        with open(loss_file_path, 'w') as f:
            for loss in losses:
                f.write(str(loss) + '\n')

    # Plot the learning curve
    plot_curve(csv_path, fig_path, "nfsp")


    plt.plot(losses)
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    # Save model
    save_path = os.path.join("experiments/limit_holdem_nfsp_result", 'best_model.pth')
    torch.save(best_agent, save_path)
    print('Best performing model saved in', save_path)

def main():
    # Define the parameter grid for tuning
    hidden_layer_sizes_grid = [[64, 64, 64]]
    q_mlp_layers_grid = [[64, 64, 64]]
    learning_rate_grid = [0.001]
    discount_factor_grid = [0.9]
    exploration_rate_grid = [0.2]

    # Perform grid search over parameter combinations
    for params in itertools.product(hidden_layer_sizes_grid, q_mlp_layers_grid, learning_rate_grid, discount_factor_grid, exploration_rate_grid):
        print(f"Training with parameters: {params}")
        train(*params)

if __name__ == '__main__':
    main()
