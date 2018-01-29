import envs.FinalEnv as Env
import random
import numpy as np
import matplotlib.pyplot as plt


# Params
INTERACTION_ROUND = 50000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 2000
REWIRING_PROBABILITY = 0.0001

K = 1

REWIRING_STRATEGY = 4
# LEARNING_STRATEGY = 0

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 1

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

sg_result = []

l00 = []
l11 = []
l22 = []

l0 = []
l1 = []
l2 = []

tpl0= 0.0
tpl1= 0.0
tpl2= 0.0


if __name__ == '__main__':

    for sg in [2]:
        result = []
        LEARNING_STRATEGY = sg

        if LEARNING_STRATEGY == 0:
            print('--Learning strategy: BR')
        if LEARNING_STRATEGY == 2:
            print('--Learning strategy: JAL')
        if LEARNING_STRATEGY == 1:
            print('--Learning strategy: JA-WoLF')

        print('C:', REWIRING_COST, 'K:', K)

        average_reward = []
        highest_reward = []
        lowest_reward = []
        average_rewiring = 0.0

        for repeat in range(EPISODE):
            tpl0 = 0.0
            tpl1 = 0.0
            tpl2 = 0.0

            l0 = []
            l1 = []
            l2 = []
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
                # average single-round payoff within nearest 2000 round
                if (iteration + 1) % 2000 == 0:
                    print(iteration + 1)
                    group_reward = env.getInteractionPayoff()
                    if sg == 0:
                        l0.append((sum(group_reward) - tpl0) / AGENT_NUM / 2000)
                        tpl0 = sum(group_reward)
                    if sg == 1:
                        l1.append((sum(group_reward) - tpl1) / AGENT_NUM / 2000)
                        tpl1 = sum(group_reward)
                    if sg == 2:
                        l2.append((sum(group_reward) - tpl2) / AGENT_NUM / 2000)
                        tpl2 = sum(group_reward)


            if sg == 0:
                l00.append(l0)
            if sg == 1:
                l11.append(l1)
            if sg == 2:
                l22.append(l2)

            # print(env.oppActionDis)
            print('Policy check:')
            n1 = 0
            n2 = 0
            n3 = 0
            for (i, j) in env.network.edges():

                if env.network.edge[i][j]['is_connected'] == 0:
                    gg = env.network.edge[i][j]['game']
                    pp =env.network.edge[i][j]['policy_pair']
                    p_i = env.oppActionDis[j, i]
                    p_j = env.oppActionDis[i, j]
                    n1 += 1
                    if LEARNING_STRATEGY == 1:
                        if gg[0,0] < gg[1,1] and pp[0] < 0.05 and pp[1] < 0.05:
                            n2 += 1
                        if gg[0,0] > gg[1,1] and pp[0] > 0.95 and pp[1] > 0.95:
                            n2 += 1
                        if pp[0] < 0.05 and pp[1] < 0.05:
                            n3 += 1
                        if pp[0] > 0.95 and pp[1] > 0.95:
                            n3 += 1
                    else:
                        if gg[0,0] < gg[1,1] and p_i < 0.05 and p_j < 0.05:
                            n2 += 1
                        if gg[0,0] > gg[1,1] and p_i > 0.95 and p_j > 0.95:
                            n2 += 1
                        if p_i < 0.05 and p_j < 0.05:
                            n3 += 1
                        if p_i > 0.95 and p_j > 0.95:
                            n3 += 1
                    print('edge: ', i, ',', j)
                    print('game:', gg)
                    print('policy pair:', pp)
                    print('estimated policy pair:', p_i , ',', p_j)
            print('total:', n1)
            print('NE:', n2)
            print('Coor:', n3)

        if sg == 0:
            for ii in l00:
                print('BR:', ii)
        if sg == 1:
            for ii in l11:
                print('JA-WoLF:',ii)
        if sg == 2:
            for ii in l22:
                print('JAL:',ii)



    print('======================================')
    print('Results below.')

    print('BR:')
    l000 = []
    l00 = np.array(l00).reshape(EPISODE, -1)

    (lx, ly) = l00.shape
    for l in range(ly):
        l000.append(sum(l00[:,l])/lx)
    print(l000)

    print('JA-WoLF:')
    l111 = []
    l11 = np.array(l11).reshape(EPISODE, -1)
    (lx, ly) = l11.shape
    for l in range(ly):
        l111.append(sum(l11[:,l])/lx)
    print(l111)

    print('JAL:')
    l222 = []
    l22 = np.array(l22).reshape(EPISODE, -1)
    (lx, ly) = l22.shape
    for l in range(ly):
        l222.append(sum(l22[:,l])/lx)
    print(l222)




############################################################
# Results

# 18/01/24
# BR, JAL, JA-WoLF

# first test EPISODE = 1
# BR:
# [1.1990762932712009, 1.2382387860878896, 1.2591179574901088, 1.2673569824903617, 1.3052084072728425, 1.3317567488812527, 1.331853086419895, 1.3315884081865534, 1.3339560758944438, 1.3386903776912811, 1.3487172919833381, 1.348089898012937, 1.3479236150257359, 1.3487866671655655, 1.3487986590070975, 1.3545727251605153, 1.3555896589889422, 1.3551333047507517, 1.356471979803308, 1.3553607241673766, 1.3562752366181092, 1.3568464859536011, 1.3571404451005253, 1.3576583928327215, 1.3552006215082948]
