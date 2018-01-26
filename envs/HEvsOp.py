import envs.MixEnv as Env
import random
import matplotlib.pyplot as plt


# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 80
REWIRING_PROBABILITY = 0.01

K = 2

REWIRING_STRATEGY = [0.2,0.2,0.2,0.2,0.2]

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 50

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

pc_list = []
for x in range(1,10):
    pc_list.append(x*0.1)


# c_list = []
# for x in range(0,11):
#     c_list.append(x*20.0)
# print(c_list)

# s0_axi = []
s1_axi = []
# s2_axi = []
# s3_axi = []
s4_axi = []

if __name__ == '__main__':

    for pc in pc_list:
        REWIRING_STRATEGY = pc
        print('For pc - ', pc)
        # s0 = []
        s1 = []
        # s2 = []
        # s3 = []
        s4 = []

        average_reward = []
        highest_reward = []
        lowest_reward = []
        average_rewiring = 0.0

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

            # if s0_reward != []:
            #     s0.append(sum(s0_reward) / len(s0_reward))

            if s1_reward != []:
                s1.append(sum(s1_reward) / len(s1_reward))

            # if s2_reward != []:
            #     s2.append(sum(s2_reward) / len(s2_reward))
            #
            # if s3_reward != []:
            #     s3.append(sum(s3_reward) / len(s3_reward))

            if s4_reward != []:
                s4.append(sum(s4_reward) / len(s4_reward))

        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('Final outputs:')

        # print('Mean Random reward: ' + str(sum(s0) / len(s0)))
        print('Mean HE reward: ' + str(sum(s1) / len(s1)))
        # print('Mean Opt1 reward: ' + str(sum(s2) / len(s2)))
        # print('Mean Opt2 reward: ' + str(sum(s3) / len(s3)))
        print('Mean Opt3 reward: ' + str(sum(s4) / len(s4)))

        # s0_axi.append(sum(s0) / len(s0))
        s1_axi.append(sum(s1) / len(s1))
        # s2_axi.append(sum(s2) / len(s2))
        # s3_axi.append(sum(s3) / len(s3))
        s4_axi.append(sum(s4) / len(s4))

    print('======================================')
    print('Results below.')

    # print('Ran: ', s0_axi)
    print('HE: ', s1_axi)
    # print('Opt1: ', s2_axi)
    # print('Opt2: ', s3_axi)
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

# 18/01/25
# HE vs. Opt3

# K = 2, c = 40
# HE:  [1228.8055163977012, 1225.2641675133145, 1231.439381165781, 1227.5285693303667, 1230.2779712287659, 1224.9028825611879, 1230.1122406450793, 1229.5504651421652, 1218.8067444039789]
# Opt3:  [1251.2110090722499, 1251.5725025828317, 1245.2644729964709, 1240.6531361715213, 1243.1344380429036, 1251.3408068248025, 1237.8866020476423, 1231.5032391342954, 1238.5376239070979]

# K = 2, c = 60
# HE:  [1126.6128120052015, 1118.9976076648372, 1125.3659098314661, 1118.2389785788368, 1124.8214719255129, 1117.6937071415093, 1124.219585268437, 1132.7422760589764, 1126.3684328672864]
# Opt3:  [1187.8279926568437, 1182.6378854230268, 1162.6651336805664, 1184.6530204263729, 1194.0658291951636, 1187.7307266881955, 1178.7029782477975, 1198.0955946866095, 1181.9650586796372]

# K = 1, c = 40
# HE:  [1229.4124087435439, 1237.4030578951063, 1234.4871580281977, 1251.3893890826937, 1223.5740050740737, 1231.4368210176094, 1231.5801284518109, 1227.1054543409361, 1237.220867128492]
# Opt3:  [1247.0431873761099, 1239.9699277223963, 1247.9214939214985, 1241.5740719546627, 1236.0518517211333, 1248.7761527661255, 1235.3596326140746, 1245.0040959707869, 1252.0089227835647]

# K = 2, c = 80