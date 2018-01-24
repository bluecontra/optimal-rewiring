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

REWIRING_STRATEGY = 0

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 30

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

c_sight_pair = []
for x in range(21):
    c_sight_pair.append([x * 10.0, 200])
print(c_sight_pair)


sg_result = []

if __name__ == '__main__':

    for sg in [0,1,2,3,4]:
        result = []
        REWIRING_STRATEGY = sg

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

        for [c, sight] in c_sight_pair:
            y_axi = []

            average_reward = []
            highest_reward = []
            lowest_reward = []
            average_rewiring = 0.0
            print('C-Sight pair:' + str([c, sight]))
            REWIRING_COST = c

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

            result.append(sum(average_reward) / EPISODE)
        print('Interim result:', result)
        sg_result.append(result)

    print('======================================')
    print('Results below.')

    for l in sg_result:
        print(l)

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
# Pure Env
# c-sight
# [[0.0, 200], [10.0, 200], [20.0, 200], [30.0, 200], [40.0, 200], [50.0, 200], [60.0, 200], [70.0, 200], [80.0, 200], [90.0, 200], [100.0, 200], [110.0, 200], [120.0, 200], [130.0, 200], [140.0, 200], [150.0, 200], [160.0, 200], [170.0, 200], [180.0, 200], [190.0, 200], [200.0, 200]]

# Ran
# [1104.1263331865787, 1039.8735986027439, 992.6071637516186, 960.40673918904963, 905.19377726482344, 853.58961364876927, 806.81203224934779, 743.76348882532, 705.95493459435306, 651.15802173572206, 611.75853016003271, 550.36526718486414, 505.83065360670128, 442.21462771905442, 389.48240234984416, 355.13186160851518, 308.6080692960237, 225.55845106507076, 212.8217195587626, 140.59622614592988, 115.25286537198072]
