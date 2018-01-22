import envs.BasicEnv as Env
import random
# Params
INTERACTION_ROUND = 100000

AGENT_NUM = 100
BANDWIDTH = 5
NEIGHBORHOOD_SIZE = 10

REWIRING_COST = 20
REWIRING_PROBABILITY = 0.01

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 1

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

if __name__ == '__main__':

    for repeat in range(EPISODE):
        print('Episode: ' + str(repeat))

        # init Env
        env = Env.BasicEnv(AGENT_NUM,
                           NETWORK_TYPE,
                           BANDWIDTH,
                           NEIGHBORHOOD_SIZE,
                           REWIRING_COST,
                           REWIRING_PROBABILITY,
                           DISTRIBUTION_TYPE)

        phi = env.phi

        for iteration in range(INTERACTION_ROUND):
            # print('Interaction round: ', iteration)
            # to avoid the the problem that decision order may influence the fairness
            # shuffle the order at every iteration
            agents = env.network.nodes()
            random.shuffle(agents)
            # print('Env initialized.')

            # rewiring phase
            # print('Rewiring phase.')
            for i in agents:
                if random.uniform(0, 1) < phi:
                    # do rewire
                    env._rewire(i)

            # interaction phase
            # print('Interaction phase.')
            for i in env.network.nodes():
                # do interaction
                neighbors_num = len(env.network.neighbors(i))
                if neighbors_num > 0:
                    # 1) randomly choose a opponent in S (choose the best opponent)
                    oppo_index = random.randint(0, neighbors_num - 1)
                    oppo_agent_no = env.network.neighbors(i)[oppo_index]

                    # sort the players
                    left = min(i, oppo_agent_no)
                    right = max(i, oppo_agent_no)

                    # 2) agent i interacts with certain opponent
                    env._interact(left, right)
                    # env._interact(i, oppo_agent_no)
                else:
                    print('agent ', i, ' has no neighbor.')

                    # statistic
        group_reward, group_rewiring = env.printAgentInfo()
        average_reward.append(sum(group_reward) / AGENT_NUM)
        highest_reward.append(max(group_reward))
        lowest_reward.append(min(group_reward))

        average_rewiring = average_rewiring + group_rewiring / AGENT_NUM

    print('--------------------------------------------------------------------')
    print('--------------------------------------------------------------------')
    print('Final outputs:')
    print('Mean average rewiring: ' + str(average_rewiring / EPISODE))
    print('Mean average reward: ' + str(sum(average_reward) / EPISODE))
    print('Mean highest reward: ' + str(sum(highest_reward) / EPISODE))
    print('Mean lowest reward: ' + str(sum(lowest_reward) / EPISODE))