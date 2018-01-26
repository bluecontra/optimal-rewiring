import envs.NewEnv as Env
import random
# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 80
REWIRING_PROBABILITY = 0.01

K = 2

REWIRING_STRATEGY = 1

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 10

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = 0

if __name__ == '__main__':

    for repeat in range(EPISODE):
        if DEBUG > -1:
            print('Episode: ' + str(repeat))

        # init Env
        env = Env.NewEnv(AGENT_NUM,
                           NETWORK_TYPE,
                           BANDWIDTH,
                           NEIGHBORHOOD_SIZE,
                           REWIRING_COST,
                           REWIRING_PROBABILITY,
                           DISTRIBUTION_TYPE,
                           REWIRING_STRATEGY,
                           K)

        phi = env.phi

        for iteration in range(INTERACTION_ROUND):
            if DEBUG > 0:
                print('-- Interaction round: ', iteration)
            # to avoid the the problem that decision order may influence the fairness
            # shuffle the order at every iteration
            agents = env.network.nodes()
            # random.shuffle(agents)
            # print('Env initialized.')

            # rewiring phase
            # print('Rewiring phase.')

            # do rewiring
            # should be good with sparse rewiring
            for i in agents:
                agent = env.network.node[i]
                if random.uniform(0, 1) < phi:
                    neighbors_num = len(env.getNeighbors(i))
                    # print(network.neighbors(i))
                    if neighbors_num > 0:
                        if len(agent['S_'] + agent['BL']) > 0:
                            # do rewire
                            if DEBUG > 0:
                                print('Agent ' + str(i) + ' does rewiring.')
                            env._rewire(i)
                        else:
                            if DEBUG > 0:
                                print('No more available potential peers.')
                    else:
                        if DEBUG > 0:
                            print('Agent ' + str(i) + ' is isolated.')
                        # TO-DO
                        if len(agent['S_'] + agent['BL']) > 0:
                            # do rewire
                            if DEBUG > 0:
                                print('Agent ' + str(i) + ' does rewiring.')
                            env._rewire(i)
                        else:
                            if DEBUG > 0:
                                print(i, 'has no more available potential peers.')

            # TO-DO
            # more reasonable situation, but complex
            # 1) raise rewiring proposals.
            # 2) decide rewiring target

            # interaction phase
            # print('Interaction phase.')
            for i in env.network.nodes():
                # do interaction
                neighborhood = env.getNeighbors(i)
                if len(neighborhood) > 0:
                    # 1) randomly choose a opponent in S (choose the best opponent)
                    oppo_index = random.randint(0, len(neighborhood) - 1)
                    oppo_agent_no = neighborhood[oppo_index]

                    # sort the players
                    left = min(i, oppo_agent_no)
                    right = max(i, oppo_agent_no)

                    # 2) agent i interacts with certain opponent
                    env._interact(left, right)
                    # env._interact(i, oppo_agent_no)
                else:
                    if DEBUG > 0:
                        print('agent ', i, ' has no neighbor.')

                    # statistic
        group_reward, group_rewiring = env.printAgentInfo(1)
        average_reward.append(sum(group_reward) / AGENT_NUM)
        highest_reward.append(max(group_reward))
        lowest_reward.append(min(group_reward))

        average_rewiring = average_rewiring + group_rewiring / AGENT_NUM

        # print(env.oppActionDis)

    print('--------------------------------------------------------------------')
    print('--------------------------------------------------------------------')
    print('Final outputs:')
    print('Mean average rewiring: ' + str(average_rewiring / EPISODE))
    print('Mean average reward: ' + str(sum(average_reward) / EPISODE))
    print('Mean highest reward: ' + str(sum(highest_reward) / EPISODE))
    print('Mean lowest reward: ' + str(sum(lowest_reward) / EPISODE))

