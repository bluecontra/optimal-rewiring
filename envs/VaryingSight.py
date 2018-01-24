import envs.BasicEnv as Env
import random
import matplotlib.pyplot as plt


# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 40
REWIRING_PROBABILITY = 0.01

K = 2

REWIRING_STRATEGY = 2

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 10

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

# c_list = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
# c_list = [20.0, 40.0, 60.0, 80.0, 100.0]
c_list = [20.0, 40.0, 80.0, 120.0, 160.0, 200.0]
# c_list = [120.0, 140.0, 160.0, 180.0, 200.0]
# c_list = [150.0, 250.0, 300.0, 350.0, 400.0]
sight_list = []
for s in range(1, 21):
    sight_list.append(s*0.1)

# for m in range(1, 41):
#     sight_list.append(2000 + m*200)

y_list = []

if __name__ == '__main__':

    if REWIRING_STRATEGY == 0:
        print('--Rewiring strategy: Random')
    if REWIRING_STRATEGY == 1:
        print('--Rewiring strategy: HE')
    if REWIRING_STRATEGY == 2:
        print('--Rewiring strategy: Opt1')
    if REWIRING_STRATEGY == 3:
        print('--Rewiring strategy: Opt2')
    if REWIRING_STRATEGY == 4:
        print('--Rewiring strategy: Opt3')

    for c in c_list:
        y_axi = []
        print('For cost: ', c)
        REWIRING_COST = c
        for sight in sight_list:
            print('For K: ', sight)
            K = sight

            average_reward = []
            highest_reward = []
            lowest_reward = []
            average_rewiring = 0.0

            for repeat in range(EPISODE):
                if DEBUG > -1:
                    print('Episode: ' + str(repeat))

                # init Env
                env = Env.BasicEnv(AGENT_NUM,
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
                group_reward, group_rewiring = env.printAgentInfo(0)
                average_reward.append(sum(group_reward) / AGENT_NUM)
                highest_reward.append(max(group_reward))
                lowest_reward.append(min(group_reward))

                average_rewiring = average_rewiring + group_rewiring / AGENT_NUM

                # print(env.oppActionDis)

            # print('--------------------------------------------------------------------')
            # print('--------------------------------------------------------------------')
            # print('Final outputs:')
            # print('Mean average rewiring: ' + str(average_rewiring / EPISODE))
            print('Mean average reward: ' + str(sum(average_reward) / EPISODE))
            # print('Mean highest reward: ' + str(sum(highest_reward) / EPISODE))
            # print('Mean lowest reward: ' + str(sum(lowest_reward) / EPISODE))
            # print('--------------------------------------------------------------------')
            # print('--------------------------------------------------------------------')

            y_axi.append(sum(average_reward) / EPISODE)
        y_list.append(y_axi)

    print('======================================')
    print('Results below.')
    for l in y_list:
        print(l)
    x_axi = sight_list

    # plt.xlim(0.0, 1.0)
    # plt.ylim(-0.1, 1.0)

    # for y in y_list:
    #     plt.plot(x_axi, y)
    # plt.title('(50,3,9) - Optimal - k-sight')
    #
    # plt.show()




############################################################
# Results

# 18/01/24
# Varying sight