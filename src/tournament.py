import os
import itertools
import copy
import torch
import matplotlib.pyplot as plt
from typing import List

import rlcard
from rlcard.agents import RandomAgent, NFSPAgent, DQNAgent, human_agents, dmc_agent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.envs import Env


class tournaments():
    def __init__(self, agent_paths : List[str], env_name = "limit-holdem", num_games = 100, seed = 42):
        self.agent_paths = agent_paths
        self.env_name = env_name
        self.num_games = num_games
        self.env = rlcard.make(
            env_name,
            config={
                'seed': seed,
                'game_num_players': len(agent_paths),
            }
        )


    def game(self):
        """ Evaluate the performance of the agents in the environment

        Args:
            agents (list): The list of agents to be evaluated
            env (Env): The environment, which is limitHoldem in this case
            num_games (int): The number of games to be played

        Returns:
            list: A list of average payoffs for each agent
        """
        self.agents = [self.agent_init(agent_path, self.env) for agent_path in self.agent_paths]
        self.env.set_agents(self.agents)
        rewards = rlcard.utils.tournament(self.env, self.num_games)
        print(rewards)


    def agent_init(self, model_path : str, env : Env):
        """ Initialize the agent

        Args:
            agent_name (str): The name of the agent

        Returns:
            Agent: The agent
        """
        device = get_device()
                
        return torch.load(model_path)
    

if __name__ == '__main__':
    # 5 dqn agents tournament
    # astral-sweep-5: GitHub/AIPlanningProject/src/results/limited_holdem_results/lr=5e-05_mlp=[32,32]_bs=128_df=0.95/model.pth
    # glowing-sweep-4: GitHub/AIPlanningProject/src/results/limited_holdem_results/lr=1e-05_mlp=[128,128]_bs=32_df=0.99/model.pth
    # expert-sweep-3: GitHub/AIPlanningProject/src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=32_df=0.99/model.pth
    # atomic-sweep-2: GitHub/AIPlanningProject/src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=32_df=0.9/model.pth
    # silver-sweep-1: GitHub/AIPlanningProject/src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=128_df=0.95/model.pth

    # agent_names = ["dqn", "nfsp", "dmc"]
    agent_paths = [
        "src/results/limited_holdem_results/lr=5e-05_mlp=[32,32]_bs=128_df=0.95/model.pth",
        "src/results/limited_holdem_results/lr=1e-05_mlp=[128,128]_bs=32_df=0.99/model.pth",
        "src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=64_df=0.99/model.pth",
        "src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=32_df=0.9/model.pth",
        "src/results/limited_holdem_results/lr=0.0001_mlp=[64,64]_bs=32_df=0.95/model.pth"
    ]
    tournament = tournaments(agent_paths = agent_paths)
    tournament.game()
