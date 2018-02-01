import envs.FinalEnv as Env
import random
# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 30
REWIRING_PROBABILITY = 0.01

K = 1

REWIRING_STRATEGY = 4
LEARNING_STRATEGY = 0

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 30

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = 0

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

    for repeat in range(EPISODE):
        if DEBUG > -1:
            print('Episode: ' + str(repeat))

        # init Env
        # env = Env.NewEnv(AGENT_NUM,
        env = Env.BasicEnv(AGENT_NUM,
                           NETWORK_TYPE,
                           BANDWIDTH,
                           NEIGHBORHOOD_SIZE,
                           REWIRING_COST,
                           REWIRING_PROBABILITY,
                           DISTRIBUTION_TYPE,
                           REWIRING_STRATEGY,
                           LEARNING_STRATEGY,
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

# diff network param

# c = 40, phi = 0.01
# (100,4,8)

# ran
# Mean average reward: 955.384987926
# Mean highest reward: 1498.61436177
# Mean lowest reward: 446.997369289

# khe k = 1
# Mean average reward: 1122.60149291
# Mean highest reward: 1715.35028143
# Mean lowest reward: 408.644097543

# Mean average reward: 1110.4222923
# Mean highest reward: 1738.62958526
# Mean lowest reward: 256.59030994

# opt-min
# Mean average reward: 1182.58407656
# Mean highest reward: 1679.90247767
# Mean lowest reward: 639.606463236

# Mean average reward: 1158.11998202
# Mean highest reward: 1645.46994451
# Mean lowest reward: 676.662846156

# opt-max
# Mean average reward: 1254.01419246
# Mean highest reward: 1850.04740801
# Mean lowest reward: 629.781272974


# c = 30, \phi = 0.01
# (100,4,8)

# Mean average reward: 974.176392323
# Mean highest reward: 1509.78715112
# Mean lowest reward: 473.501462241
#
# Mean average reward: 1126.55626183
# Mean highest reward: 1723.02307334
# Mean lowest reward: 372.319349937
#
# Mean average reward: 1183.07154471
# Mean highest reward: 1686.13911437
# Mean lowest reward: 623.12370091
#
# Mean average reward: 1254.85165621
# Mean highest reward: 1869.47080074
# Mean lowest reward: 574.323476222
#
# Mean average reward: 1270.92668092
# Mean highest reward: 1846.96998994
# Mean lowest reward: 623.920049599

# (100,4,12)

# Mean average reward: 945.451728423
# Mean highest reward: 1424.44228852
# Mean lowest reward: 435.874635603
#
# Mean average reward: 1157.41123228
# Mean highest reward: 1760.91091121
# Mean lowest reward: 396.903623781
#
# Mean average reward: 1191.17450163
# Mean highest reward: 1701.16726356
# Mean lowest reward: 624.54947424
#
# Mean average reward: 1282.66695097
# Mean highest reward: 1934.72865151
# Mean lowest reward: 595.252565645
#
# Mean average reward: 1310.35538593
# Mean highest reward: 1947.8080632
# Mean lowest reward: 617.913926612

# (100, 4, 16)
# Mean average reward: 952.40721285
# Mean highest reward: 1457.34284206
# Mean lowest reward: 412.799098141
#
# Mean average reward: 1181.89631538
# Mean highest reward: 1762.34895266
# Mean lowest reward: 392.731483036
#
# Mean average reward: 1190.27327921
# Mean highest reward: 1691.52256896
# Mean lowest reward: 601.869094817
#
# Mean average reward: 1293.79833885
# Mean highest reward: 1932.0819695
# Mean lowest reward: 601.544530919
#
# Mean average reward: 1330.39164474
# Mean highest reward: 1999.24706211
# Mean lowest reward: 619.904592129

# (500, 4, 16)
# Mean average reward: 945.894272674
# Mean highest reward: 1609.82382727
# Mean lowest reward: 330.71701522
#
# Mean average reward: 1185.77326471
# Mean highest reward: 1897.9227319
# Mean lowest reward: 182.91919701
#
# Mean average reward: 1188.81910794
# Mean highest reward: 1769.10034777
# Mean lowest reward: 509.11199724
#
# Mean average reward: 1300.02613657
# Mean highest reward: 2151.963564
# Mean lowest reward: 449.14469517
#
# Mean average reward: 1332.91549254
# Mean highest reward: 2158.11936435
# Mean lowest reward: 435.622022395

# (500, 8, 16)
# Mean average reward: 929.670781093
# Mean highest reward: 1399.23760928
# Mean lowest reward: 447.653212554
#
# Mean average reward: 1050.99397362
# Mean highest reward: 1530.87070652
# Mean lowest reward: 477.173694633
#
# Mean average reward: 1122.48841584
# Mean highest reward: 1550.95389678
# Mean lowest reward: 685.661994062
#
# Mean average reward: 1186.66748686
# Mean highest reward: 1615.36375831
# Mean lowest reward: 698.472744888
#
# Mean average reward: 1207.34233871
# Mean highest reward: 1655.96896995
# Mean lowest reward: 707.284851933

# (1000, 8, 16)
# Mean average reward: 929.492234086
# Mean highest reward: 1423.54343635
# Mean lowest reward: 432.934325662

# Mean average reward: 1052.36888124
# Mean highest reward: 1571.16926307
# Mean lowest reward: 443.859139709
#
# Mean average reward: 1121.35555122
# Mean highest reward: 1566.52066908
# Mean lowest reward: 650.275566126
#
# Mean average reward: 1184.20843413
# Mean highest reward: 1640.68807244
# Mean lowest reward: 658.097008663
#
# Group average reward: 1207.55022139
# Group highest reward: 1810.33807821
# Group lowest reward: 726.559745898