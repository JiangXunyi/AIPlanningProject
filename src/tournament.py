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
    def __init__(self, agent_names : List[str], env_name = "limit-holdem", num_games = 100, seed = 42):
        self.agent_names = agent_names
        self.env_name = env_name
        self.num_games = num_games
        self.env = rlcard.make(
            env_name,
            config={
                'seed': seed,
                'game_num_players': len(agent_names),
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
        self.agents = [self.agent_init(agent_name, self.env) for agent_name in agent_names]
        self.env.set_agents(self.agents)
        rewards = rlcard.utils.tournament(self.env, self.num_games)
        print(rewards)


    def agent_init(self, agent_name : str, env : Env):
        """ Initialize the agent

        Args:
            agent_name (str): The name of the agent

        Returns:
            Agent: The agent
        """
        device = get_device()
        model_path = "src/models/" + agent_name + ".pth"

        # if agent_name == "dqn":
        #     agent = torch.load(model_path)
        # elif agent_name == "nfsp":
        #     agent = NFSPAgent.from_checkpoint(torch.load(model_path))
        # elif agent_name == "random":
        #     agent = RandomAgent.from_checkpoint(torch.load(model_path))
        # elif agent_name == "human":
        #     agent = human_agents()
        # elif agent_name == "dmc":
        #     agent = dmc_agent.from_checkpoint(torch.load(model_path))
        
        return torch.load(model_path)
    

if __name__ == '__main__':
    agent_names = ["dqn", "nfsp", "dmc"]
    tournament = tournaments(agent_names = agent_names)
    tournament.game()
