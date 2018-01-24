import envs.BasicEnv as Env
import random
# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 40
REWIRING_PROBABILITY = 0.01

K = 2

REWIRING_STRATEGY = 4

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 10

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

agent_num_set = [100, 500, 1000]
bandwidth_set = [4, 8, 12]
neighborhood_set = [8, 12, 16]



if __name__ == '__main__':

    if REWIRING_STRATEGY == 0:
        print('Rewiring strategy: Random')
    if REWIRING_STRATEGY == 1:
        print('Rewiring strategy: HE')
    if REWIRING_STRATEGY == 2:
        print('Rewiring strategy: Opt1')
    if REWIRING_STRATEGY == 3:
        print('Rewiring strategy: Opt2')
    if REWIRING_STRATEGY == 4:
        print('Rewiring strategy: Opt3')


    for m in [1]:
        AGENT_NUM = agent_num_set[2]
        NEIGHBORHOOD_SIZE = neighborhood_set[2]
        BANDWIDTH = bandwidth_set[m]

        print('Params:', AGENT_NUM, BANDWIDTH, NEIGHBORHOOD_SIZE)

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

        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('Final outputs:')
        print('Mean average rewiring: ' + str(average_rewiring / EPISODE))
        print('Mean average reward: ' + str(sum(average_reward) / EPISODE))
        print('Mean highest reward: ' + str(sum(highest_reward) / EPISODE))
        print('Mean lowest reward: ' + str(sum(lowest_reward) / EPISODE))
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')


############################################################
# Results

# 18/01/24
# HE
# Params: 100 4 8
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 10.017999999999999
# Mean average reward: 1006.5406462
# Mean highest reward: 1595.46335446
# Mean lowest reward: 244.2311123
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 12
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 10.049000000000001
# Mean average reward: 1050.83207558
# Mean highest reward: 1702.76615131
# Mean lowest reward: 352.734701688
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 9.839
# Mean average reward: 1088.19218829
# Mean highest reward: 1726.43557468
# Mean lowest reward: 276.743408503
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Opt 1
# Params: 100 4 8
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.153
# Mean average reward: 1219.77660779
# Mean highest reward: 1749.50018075
# Mean lowest reward: 642.46949292
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 12
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.18
# Mean average reward: 1222.99766676
# Mean highest reward: 1764.48070058
# Mean lowest reward: 577.165933893
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.18599999999999997
# Mean average reward: 1234.27311578
# Mean highest reward: 1821.70580286
# Mean lowest reward: 655.890204446
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Opt 2
# Params: 100 4 8
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.417
# Mean average reward: 1248.42946585
# Mean highest reward: 1889.41057553
# Mean lowest reward: 572.248306306
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 12
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.601
# Mean average reward: 1280.52498768
# Mean highest reward: 1918.04214903
# Mean lowest reward: 604.108048603
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.684
# Mean average reward: 1287.9215326
# Mean highest reward: 1968.80580873
# Mean lowest reward: 700.64667917
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Opt 3
# Params: 100 4 8
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 1.8779999999999997
# Mean average reward: 1311.75950273
# Mean highest reward: 2128.45243574
# Mean lowest reward: 602.309328201
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 12
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 2.3789999999999996
# Mean average reward: 1328.60257575
# Mean highest reward: 2299.8484594
# Mean lowest reward: 493.30308373
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 2.601
# Mean average reward: 1313.92180435
# Mean highest reward: 2174.36966048
# Mean lowest reward: 314.305324603
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Ran
# Params: 100 4 8
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 5.011000000000001
# Mean average reward: 904.131103864
# Mean highest reward: 1490.93117947
# Mean lowest reward: 420.560767079
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 12
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 4.995
# Mean average reward: 914.316226352
# Mean highest reward: 1416.22691936
# Mean lowest reward: 409.159700604
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 100 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 5.163
# Mean average reward: 896.823935062
# Mean highest reward: 1450.8611726
# Mean lowest reward: 457.664626974
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Random
# Params: 500 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 5.042
# Mean average reward: 896.353452046
# Mean highest reward: 1562.81607017
# Mean lowest reward: 281.239976441
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 500 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 4.977600000000001
# Mean average reward: 880.255210484
# Mean highest reward: 1385.44522615
# Mean lowest reward: 380.976814414
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: HE
# Params: 500 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 9.9768
# Mean average reward: 1085.00970196
# Mean highest reward: 1825.50672418
# Mean lowest reward: 114.342828166
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 500 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 9.972399999999999
# Mean average reward: 953.417393471
# Mean highest reward: 1459.549141
# Mean lowest reward: 329.761073826
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt1
# Params: 500 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.18599999999999997
# Mean average reward: 1231.54761531
# Mean highest reward: 1882.11429031
# Mean lowest reward: 477.695868429
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 500 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.1244
# Mean average reward: 1135.58168111
# Mean highest reward: 1560.1612022
# Mean lowest reward: 673.710521905
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt2
# Params: 500 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.6994
# Mean average reward: 1293.29843594
# Mean highest reward: 2125.61506441
# Mean lowest reward: 362.586741225
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 500 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.5038
# Mean average reward: 1177.60704915
# Mean highest reward: 1608.00062787
# Mean lowest reward: 729.31088752
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt3
# Params: 500 4 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 2.588
# Mean average reward: 1330.34847577
# Mean highest reward: 2401.19293252
# Mean lowest reward: 162.335671676
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Params: 500 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 4.652000000000001
# Mean average reward: 1138.10195292
# Mean highest reward: 1603.8855518
# Mean lowest reward: 576.717843225
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Random
# Params: 1000 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 5.016399999999999
# Mean average reward: 876.938018814
# Mean highest reward: 1395.7628627
# Mean lowest reward: 369.838770816
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: HE
# Params: 1000 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 9.948699999999999
# Mean average reward: 953.484302729
# Mean highest reward: 1539.34097381
# Mean lowest reward: 261.927949096
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt1
# Params: 1000 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.1246
# Mean average reward: 1136.43778669
# Mean highest reward: 1592.77321605
# Mean lowest reward: 647.531208295
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt2
# Params: 1000 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 0.4958
# Mean average reward: 1173.44008924
# Mean highest reward: 1640.03590314
# Mean lowest reward: 658.731835109
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# Rewiring strategy: Opt3
# Params: 1000 8 16
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Final outputs:
# Mean average rewiring: 4.6157
# Mean average reward: 1139.89445933
# Mean highest reward: 1646.11965647
# Mean lowest reward: 557.19911801
# --------------------------------------------------------------------
# --------------------------------------------------------------------
