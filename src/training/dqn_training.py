import os
import argparse
import matplotlib.pyplot as plt
import pickle

import torch
import wandb

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def create_log_dir_name(args):
    """Generate a directory name from parameters."""
    params_short = ['lr', 'mlp', 'bs', 'df']
    params_to_include = ['learning_rate', 'mlp_layers', 'batch_size', 'discount_factor']
    # Format each parameter and its value into a string, join with underscores
    parts = [f"{params_short[i]}={getattr(args, params_to_include[i])}" for i in range(len(params_to_include))]
    return '_'.join(parts).replace(" ", "")


def train(args = None):
    with wandb.init(
        # set the wandb project where this run will be logged
        project="AIPlanningProject",
    ):
        args = wandb.config
        log_dir_name = create_log_dir_name(args)
        log_dir = "src/results/limited_holdem_results/" + log_dir_name
        os.makedirs(log_dir, exist_ok=True)


        # Check whether gpu is available
        device = get_device()

        # Seed numpy, torch, random
        set_seed(args.seed)

        # Make the environment with seed
        env = rlcard.make(
            args.env,
            config={
                'seed': args.seed,
                'game_num_players': args.num_players,
            }
        )

        # Initialize the agent and use dmc agents as opponents
        if args.algorithm == 'dqn':
            from rlcard.agents import DQNAgent
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=args.mlp_layers,
                device=device,
                learning_rate=args.learning_rate,
                discount_factor=args.discount_factor,
                update_target_estimator_every = args.update_target_estimator_every,
                batch_size=args.batch_size,
                save_path=log_dir,
            )
        elif args.algorithm == 'nfsp':
            from rlcard.agents import NFSPAgent
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64,64],
                q_mlp_layers=[64,64],
                device=device,
            )
        agents = [agent]
        # use a random agent and a dmc agent as opponents
        for _ in range(1, env.num_players):
            agents.append(torch.load("src/models/dmc.pth"))
            agents.append(RandomAgent(num_actions=env.num_actions))
        env.set_agents(agents)

        # Start training

        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
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
                wandb.log({"loss": loss})

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                performance = tournament(
                                        env,
                                        args.num_eval_games,
                                    )
                DQN_performance = performance[0]
                DMC_performance = performance[1]
                Random_performance = performance[2]

                wandb.log({
                    "DQN_performance": DQN_performance,
                    "DMC_performance": DMC_performance,
                    "Random_performance": Random_performance,
                })

        # Save model
        
        save_path = os.path.join(log_dir, 'model.pth')
        print(save_path)
        torch.save(agent, save_path)
        print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN hyper-parameter search in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='src/results/limited_holdem_results/test1',
    )
    parser.add_argument(
        '--num_players',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00005,
    )
    parser.add_argument(
        '--mlp_layers',
        nargs='*',
        type=int,
        default=[64,64],
    )
    parser.add_argument(
        '--update_target_estimator_every',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--train_every',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--eps_start',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--discount_factor',
        type=float,
        default=0.99,
    )

    args = parser.parse_args()

    if args.debug:
        args.num_episodes = 1000
        args.num_eval_games = 10
        args.evaluate_every = 1
        args.log_dir = 'src/results/limited_holdem_results/debug'
        args.num_players = 4
        args.algorithm = 'dqn'
        args.env = 'limit-holdem'

    # init wandb board
    wandb.login(key="9d5c944ff4fa253bd79df0f97c512303800dc345")
    # using bayesian method for hyper-parameter search
    sweep_config = {
        'method': 'random',
    }
    metric = {
        'name': 'DQN_performance',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric

    parameters_dict = args.__dict__
    # add values for each parameter
    for key in parameters_dict:
        parameters_dict[key] = {
            'value': parameters_dict[key]
        }
    # parameters_dict.update({
    #     'mlp_layers': {
    #         'values': [[32,32], [64,64], [128,128]]
    #         },
    #     'learning_rate': {
    #         'values': [0.0001, 0.00005, 0.00001]
    #     },
    #     'batch_size': {
    #         'values': [32, 64, 128]
    #     },
    #     'discount_factor': {
    #         'values': [0.99, 0.95, 0.9]
    #     },
    # })

    # test update_target_estimator_every
    parameters_dict.update({
        'mlp_layers': {
            'value': [64,64]
            },
        'learning_rate': {
            'value': 0.0001
        },
        'batch_size': {
            'value': 32
        },
        'discount_factor': {
            'value': 0.9
        },
        'update_target_estimator_every': {
            'values': [1, 500, 1000, 2000]
        }
    })

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="AIPlanningProject")

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # train(args)
    wandb.agent(sweep_id, train, count=4)