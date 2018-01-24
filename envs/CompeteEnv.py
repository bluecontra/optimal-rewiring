import envs.MixEnv as Env
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

REWIRING_STRATEGY = [0.2,0.2,0.2,0.2,0.2]

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

c_list = []
for x in range(0,11):
    c_list.append(x*20.0)
print(c_list)

s0_axi = []
s1_axi = []
s2_axi = []
s3_axi = []
s4_axi = []

if __name__ == '__main__':

    for [c, sight] in c_sight_pair:
        s0 = []
        s1 = []
        s2 = []
        s3 = []
        s4 = []

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
            env = Env.MixEnv(AGENT_NUM,
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

            s0_reward, s1_reward, s2_reward, s3_reward, s4_reward = env.printAgentInfoEveryStg()

            if s0_reward != []:
                s0.append(sum(s0_reward) / len(s0_reward))

            if s1_reward != []:
                s1.append(sum(s1_reward) / len(s1_reward))

            if s2_reward == []:
                s2.append(sum(s2_reward) / len(s2_reward))

            if s3_reward != []:
                s3.append(sum(s3_reward) / len(s3_reward))

            if s4_reward != []:
                s4.append(sum(s4_reward) / len(s4_reward))

        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('Final outputs:')

        print('Mean Random reward: ' + str(sum(s0) / len(s0)))
        print('Mean HE reward: ' + str(sum(s1) / len(s1)))
        print('Mean Opt1 reward: ' + str(sum(s2) / len(s2)))
        print('Mean Opt2 reward: ' + str(sum(s3) / len(s3)))
        print('Mean Opt3 reward: ' + str(sum(s4) / len(s4)))

        s0_axi.append(sum(s0) / len(s0))
        s1_axi.append(sum(s1) / len(s1))
        s2_axi.append(sum(s2) / len(s2))
        s3_axi.append(sum(s3) / len(s3))
        s4_axi.append(sum(s4) / len(s4))

    print('======================================')
    print('Results below.')

    print('Ran: ', s0_axi)
    print('HE: ', s1_axi)
    print('Opt1: ', s2_axi)
    print('Opt2: ', s3_axi)
    print('Opt3: ', s4_axi)


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

