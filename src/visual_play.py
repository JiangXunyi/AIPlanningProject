import numpy as np
import torch

import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent
from rlcard.utils.utils import print_card

# Make environment
player_num = input("Please input the number of players (default is 2): ")

# set up agents. If 2 palyer(contains human) then use dmc, otherwise randomly
# select the number of agent from ['dmc','random','nfsp','dqn']
if player_num == '2':
    agent_names = ['dmc']
else:
    agent_names = np.random.choice(['dmc','nfsp','dqn','dqn_best'], int(player_num)-1, replace=True)

model_paths = ["src/models/" + agent_name + ".pth" for agent_name in agent_names]
agents = [torch.load(model_path) for model_path in model_paths]
env = rlcard.make(
            'limit-holdem',
            config={
                'seed': 42,
                'game_num_players': len(agents) + 1,
            }
        )
human_agent = HumanAgent(env.num_actions)
env.set_agents([
    human_agent,
] + agents)

print(">> Limit Hold'em random agent")               

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # print(trajectories)
    # If the human does not take the final action, we need to
    # print other players action
    if len(trajectories[0]) != 0:
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            """
            if action_record[-i][0] == state['current_player']:
                break
            """
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('=============     Random Agent    ============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")