import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt

#-------------------------------------------------------------------

# initialize the environment
# def initial


# initialize the network topology
# return a network(graph)
def initialNetworkTopology(agents_num, neighbors_num, bandwidth):
    N = nx.random_graphs.random_regular_graph(bandwidth, agents_num)
    initS__Size = neighbors_num - bandwidth
    for i in range(agents_num):
        S_ = []
        BL = []
        # links are initial links, the set S
        links = N.neighbors(i)
        # randomly add potential neighbors, the set S'
        for j in range(initS__Size):
            neighbor = random.randint(0, agents_num - 1)
            while neighbor == i or neighbor in links:
                neighbor = random.randint(0, agents_num - 1)
            S_.append(neighbor)
        N.node[i]['S_'] = S_
        # initial the BL for black-list
        N.node[i]['BL'] = BL
        # initial the accumulate reward and cost for agent(node)
        N.node[i]['ar'] = 0
        N.node[i]['ac'] = 0

        N.node[i]['expected_value_list'] = []
        N.node[i]['index_z_value_list'] = []
        N.node[i]['expected_value_list_S_'] = []

        N.node[i]['rebuild_mark'] = 0
        N.node[i]['rewiring_time'] = 0

        # additionally random initialize the strategy of agents, 10% random, 10% Never, 30% HE, and 50 % Optimal
        ran = random.uniform(0,1)
        if ran < 0.33:
            N.node[i]['rewiring_strategy'] = 0
        elif ran < 0.66:
            N.node[i]['rewiring_strategy'] = 1
        else:
            N.node[i]['rewiring_strategy'] = 2
        # N.node[i]['gaming_strategy'] = 0

    # ensure at least one for each strategy
    N.node[1]['rewiring_strategy'] = 0
    N.node[2]['rewiring_strategy'] = 1
    N.node[3]['rewiring_strategy'] = 2
    # N.node[4]['rewiring_strategy'] = 3
    return N

# initialize the reward distribution for each agent pair( k*(k-1)/2 ).
# return a numpy matrix
def initialDistributingSetting(agents_num, distribution_type=0):
    distribution = []
    # 0 for uniform distribution
    # for each agent pair, generate a U(a1,b1) and U(a2,b2) for respective reward distribution on different actions.
    if distribution_type == 0:
        # size = agents_num * (agents_num - 1) / 2
        size = agents_num * agents_num
        for i in range(int(size)):
            temp = []
            a1 = random.uniform(0,1)
            b1 = random.uniform(0,1)
            a2 = random.uniform(0,1)
            b2 = random.uniform(0,1)
            while (a1 >= b1):
                a1 = random.uniform(0, 1)
                b1 = random.uniform(0, 1)
            while (a2 >= b2):
                a2 = random.uniform(0, 1)
                b2 = random.uniform(0, 1)
            temp.append(a1)
            temp.append(b1)
            temp.append(a2)
            temp.append(b2)
            distribution.append(temp)
    distribution = np.matrix(distribution)

    return distribution

# reveal the unknown game by giving agent pair (i,j) (edge)
# add a new edge in network and set the attribute 'game' to the edge
def revealTheUnknownGameForEdge(network, agents_num, rewardDisMatrix, i, j):
    # sort the agent_no and neighbor, small one at left
    left = min(i, j)
    right = max(i, j)
    index = left * agents_num + right
    a = random.uniform(rewardDisMatrix[index, 0], rewardDisMatrix[index, 1])
    b = random.uniform(rewardDisMatrix[index, 2], rewardDisMatrix[index, 3])
    gameM = np.array([[a, 0], [0, b]])
    gameM = np.matrix(gameM)
    network.edge[i][j]['game'] = gameM

# initial the game in S, binding on the edge between agent(node) pair
def initialTheKnownGameInS(network, agents_num, rewardDisMatrix):
    edges = network.edges()
    for (i,j) in edges:
        # as i always < j, so directly
        revealTheUnknownGameForEdge(network, agents_num, rewardDisMatrix, i, j)

# initialize the opponents action distribution for every agent pair
# return a numpy matrix
def initialOpponentsActionDistribution(agents_num):
    distribution = []
    for i in range(agents_num):
        subdistribution = np.random.random(size=agents_num)
        distribution.append(subdistribution)
    distribution = np.matrix(distribution)
    return distribution

# check if the agent itself is the worst-behave one in its neighborhood
def checkSelfBehaviorInNeighborhood(network, agent_no):
    res = False
    S = network.neighbors(agent_no)
    my_ar = network.node[agent_no]['ar']
    neighbor_ar = []
    for n in S:
        neighbor_ar.append(network.node[n]['ar'])
    # print(neighbor_ar)
    # if my_ar < min(neighbor_ar):
    if my_ar < sum(neighbor_ar)/len(neighbor_ar):
        res = True
    return res

# get the expected reward list for set S
# update the node(agent) in network
def calculateExpectedRewardInS(network, agent_no, action):
    expectedReward_S = []
    S = network.neighbors(agent_no)
    # print(S)
    for oppo_no in S:
        # evaluate opponent's possibility to choose action a(for the sight of some agent)
        p = action[agent_no, oppo_no]
        game = network.edge[agent_no][oppo_no]['game']
        # print(p)
        # print(game)
        expectedReward_S.append(max(p * game[0,0], (1 - p) * game[1,1]))

    network.node[agent_no]['expected_value_list'] = expectedReward_S

# calculate Index for the unknown agent game
# a,b for U(a,b), c for rewiring cost
# for uniform distribution
# solve equation: (x - z)^2 | z = 2c(b-a)
# 1) if z not in (a,b)
#   z = -c + (a+b)/2
#   if z not in (a,b), z is the index
# 2) if z = -c + (a+b)/2 in (a,b)
#   z = b +(-) (2c(b-a))^1/2
#   if z in (a,b), z is the index
# return index z
def calculateIndex(a, b, c, sight):
    z = -c/sight + (a+b)/2
    if z > a:
        z = b - math.sqrt(2*c/sight*(b-a))
    return z

def calculateLambdaIndex(a, b, c, sight, y, yy):
    lam = 0

    # if yy <= a:
    #     yy_ = yy
    # if yy >= b:
    #     yy_ = x
    # if yy > a and yy <b:

    if y >= b:
        lam = (a + b)/2 - y - c/sight
    if y <= a:
        if yy <= a:
            lam = yy - y -c / sight
        if yy >= b:
            lam = (a + b) / 2 - y - c / sight
        if yy > a and yy < b:
            yy_ = (yy*yy - a*a) / (2*(b - a)) + yy*(b-yy) / (b - a)
            lam = yy_ - y -c /sight
    if y > a and y < b:
        # if yy <= a:
        #     lam = (y*y - a*a)/2 + yy*(b - c) - y - c / sight
        if yy >= b:
            lam = (a + b) / 2 - y - c / sight
        if yy > a and yy < b:
            yy_ = (yy * yy - a * a) / (2*(b - a)) + yy * (b - yy) / (b - a)
            lam = (y*y - a*a) / (2*(b - a)) - yy_*(b - y) / (b - a) - y - c/sight

    return lam

# get the index list for set S'
# update the node(agent) in network
def calculateIndexInS_(network, agents_num, agent_no, action, rewardDisMatrix, c, sight):
    # print('agent: ' + str(agent_no))
    index_S_ = []
    S_ = network.node[agent_no]['S_']
    # print(S_)
    for neighbor in S_:
        # print('calculate index for agent: ' + str(neighbor))
        p = action[agent_no, neighbor]
        # sort the agent_no and neighbor, small one at left
        left = min(agent_no, neighbor)
        right = max(agent_no, neighbor)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * agents_num + right
        # U(a1,b1) U(a2,b2)
        a1 = p * rewardDisMatrix[index_in_dis_matrix, 0]
        b1 = p * rewardDisMatrix[index_in_dis_matrix, 1]
        z1 = calculateIndex(a1, b1, c, sight)
        a2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 2]
        b2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 3]
        z2 = calculateIndex(a2, b2, c, sight)
        index_S_.append(max(z1, z2))
    network.node[agent_no]['index_z_value_list'] = index_S_
    # return index_S_

################
# new versions
# get the index list for set S' in Lambda way
# update the node(agent) in network
# calculateLambdaIndexInS_
def calculateLambdaIndexInS_(network, agents_num, agent_no, action, rewardDisMatrix, c, sight, minimum, sec, neighbors_num):
    # print('agent: ' + str(agent_no))
    index_S_ = []
    S_ = network.node[agent_no]['S_']
    # print(S_)
    for neighbor in S_:
        # print('calculate index for agent: ' + str(neighbor))
        p = action[agent_no, neighbor]
        # sort the agent_no and neighbor, small one at left
        left = min(agent_no, neighbor)
        right = max(agent_no, neighbor)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * agents_num + right
        # U(a1,b1) U(a2,b2)
        a1 = p * rewardDisMatrix[index_in_dis_matrix, 0]
        b1 = p * rewardDisMatrix[index_in_dis_matrix, 1]

        a2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 2]
        b2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 3]

        if neighbors_num == 1:
            z1 = (a1 + b1) / 2 - minimum - c / sight
            z2 = (a2 + b2) / 2 - minimum - c / sight
        else:
            # z1 = calculateIndex(a1, b1, c, sight)
            z1 = calculateLambdaIndex(a1, b1, c, sight, minimum, sec)
            # z2 = calculateIndex(a2, b2, c, sight)
            z2 = calculateLambdaIndex(a2, b2, c, sight, minimum, sec)
        index_S_.append(max(z1, z2))
    network.node[agent_no]['index_z_value_list'] = index_S_

# calculate the best response when player1 facing player2, game on edge (i,j)
def calculateBestResponse(network, player1, player2, action):
    game = network.edge[player1][player2]['game']
    p = action[player1, player2]
    # print(game)
    # print(p)
    if p * game[0, 0] >= (1 - p) * game[1, 1]:
        best_response = 0
    else:
        best_response = 1
    # print(best_response)
    return best_response

# get the expected value list for set S'
# update the node(agent) in network
def calculateExpectedValueInS_(network, agents_num, agent_no, action, rewardDisMatrix, c, sight):
    expected_value_S_ = []
    S_ = network.node[agent_no]['S_']
    # print(S_)
    for neighbor in S_:
        # print('calculate index for agent: ' + str(neighbor))
        p = action[agent_no, neighbor]
        # sort the agent_no and neighbor, small one at left
        left = min(agent_no, neighbor)
        right = max(agent_no, neighbor)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * agents_num + right
        # U(a1,b1) U(a2,b2)
        a1 = p * rewardDisMatrix[index_in_dis_matrix, 0]
        b1 = p * rewardDisMatrix[index_in_dis_matrix, 1]
        ev1 = (a1 + b1) / 2 - c/sight
        a2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 2]
        b2 = (1 - p) * rewardDisMatrix[index_in_dis_matrix, 3]
        ev2 = (a2 + b2) / 2 - c/sight
        expected_value_S_.append(max(ev1, ev2))
    network.node[agent_no]['expected_value_list_S_'] = expected_value_S_

# update opponent model as we can record opponent's action
# update the oppoActionDis
def updateOpponentActionModel(oppoActionDis, player1, player2, oppo_action, round):
    base_round = 100
    total_round = base_round + round
    # get the previous experiment
    p_old = oppActionDis[player1, player2]
    if oppo_action == 0:
        p_new = p_old * total_round / (total_round + 1) + 1 / (total_round + 1)
    else:
        p_new = p_old * total_round / (total_round + 1)
    oppActionDis[player1, player2] = p_new
#--------------------------------------------------------------------------------------

# 20, 50, 100
agents_num = 100
# 5, 10, 20
neighborhood = 12
# 3, 5, 10
bandwidth = 4

iso_damage = -1.0
c = 20.00
sight = 400
rebuild_threshold = 0.0

# 0 for random, 1 for highest-expect, 2 for optimal, 3 for never
# rewiring_strategy = 2

average_reward = []
highest_reward = []
lowest_reward = []

s0 = []
s1 = []
s2 = []
# s3 = []

average_rewiring = 0.0

repeat_time = 300

foo = 0


c_list = []
for x in range(1,11):
    c_list.append(x*20.0)

s0_axi = []
s1_axi = []
s2_axi = []
# s3_axi = []

# print(c_list)
for c in c_list:
    print('For c,sight pair - (' + str(c) + ',' + str(sight) +')')
    s0 = []
    s1 = []
    s2 = []
    # s3 = []
    for repeat in range(repeat_time):
        # print('Repeating: ' + str(repeat))
        # initial the network and unknown reward distribution
        network = initialNetworkTopology(agents_num, neighborhood, bandwidth)
        rewardDisMatrix = initialDistributingSetting(agents_num)
        # print(rewardDisMatrix)
        # print(len(rewardDisMatrix))

        # initial the know game in S
        initialTheKnownGameInS(network, agents_num, rewardDisMatrix)
        # for (i,j) in network.edges():
        #     print(network.edge[i][j]['game'])

        # initial the P-table for every agent pair
        oppActionDis = initialOpponentsActionDistribution(agents_num)

        for iteration in range(1000):
            # to avoid the the problem that decision order may influence the fairness
            # shuffle the order at every iteration
            agents = network.nodes()
            random.shuffle(agents)
            # print(network.nodes())
            # print(agents)

            if (iteration) % 100 == 0:
                # rewiring phase
                # print('Rewiring phase.')
                for i in agents:
                    # print('Agent: ' + str(i))
                    neighbors_num = len(network.neighbors(i))
                    # print(network.neighbors(i))
                    if neighbors_num > 0:
                        # check if itself is the worst one.
                        # if checkSelfBehaviorInNeighborhood(network, i) == True:
                        if network.node[i]['S_'] != []:

                            rewiring_strategy = network.node[i]['rewiring_strategy']
                            # randomly rewiring
                            if rewiring_strategy == 0:
                                unlink_agent_index = random.randint(0, len(network.neighbors(i)) - 1)
                                unlink_agent_no = network.neighbors(i)[unlink_agent_index]

                                # 2017/08/16
                                # guarantee each agent has at least 1 neighbor
                                # quit the rewiring if worst-agent only has 1 link
                                if len(network.neighbors(unlink_agent_no)) >= 2:
                                    rewiring_agent_index = random.randint(0, len(network.node[i]['S_']) - 1)
                                    rewiring_agent_no = network.node[i]['S_'][rewiring_agent_index]

                                    network.node[i]['BL'].append(unlink_agent_no)
                                    network.remove_edge(i, unlink_agent_no)

                                    network.node[i]['ac'] = network.node[i]['ac'] + c
                                    network.node[i]['S_'].remove(rewiring_agent_no)
                                    network.add_edge(i, rewiring_agent_no)
                                    # update information
                                    # reveal the reward matrix of the unknown agent and update
                                    revealTheUnknownGameForEdge(network, agents_num, rewardDisMatrix, i,
                                                                rewiring_agent_no)

                                    network.node[i]['rewiring_time'] = network.node[i]['rewiring_time'] + 1

                            # highest-expect rewiring
                            if rewiring_strategy == 1:
                                # 1) get the maximum(minimum) expected value in S
                                calculateExpectedRewardInS(network, i, oppActionDis)
                                minimun_expected_reward = min(network.node[i]['expected_value_list'])
                                # maximun_expected_reward = max(network.node[i]['expected_value_list'])

                                # 2) calculate the expected value of each neighbor in S' and get the maximum expected value ev
                                calculateExpectedValueInS_(network, agents_num, i, oppActionDis, rewardDisMatrix, c,
                                                           sight)
                                ev_m = max(network.node[i]['expected_value_list_S_'])
                                ev_max_index = np.array(network.node[i]['expected_value_list_S_']).argmax()

                                # 3) if the max_ev > minimum_expected_reward, do rewiring
                                # if ev_m > minimun_expected_reward:
                                if ev_m > 0:
                                    # print('Rewiring...')
                                    # do the rewiring

                                    # since the links number in MAS is limited,
                                    # so we unlink the worst one meanwhile we do the rewiring
                                    calculateExpectedRewardInS(network, i, oppActionDis)
                                    worst_agent_index = np.array(network.node[i]['expected_value_list']).argmin()
                                    worst_agent_no = network.neighbors(i)[worst_agent_index]

                                    # 2017/08/16
                                    # guarantee each agent has at least 1 neighbor
                                    # quit the rewiring if worst-agent only has 1 link
                                    if len(network.neighbors(worst_agent_no)) >= 2:
                                        # print('  Unlink the agent: ' + str(worst_agent_no))
                                        network.node[i]['BL'].append(worst_agent_no)
                                        network.remove_edge(i, worst_agent_no)

                                        rewiring_agent_no = network.node[i]['S_'][ev_max_index]
                                        # print('  Rewire the agent: ' + str(max_agent_no))

                                        network.node[i]['ac'] = network.node[i]['ac'] + c
                                        network.node[i]['S_'].pop(ev_max_index)
                                        network.add_edge(i, rewiring_agent_no)
                                        # update information
                                        # reveal the reward matrix of the unknown agent and update
                                        revealTheUnknownGameForEdge(network, agents_num, rewardDisMatrix, i,
                                                                    rewiring_agent_no)

                                        network.node[i]['rewiring_time'] = network.node[i]['rewiring_time'] + 1

                                else:
                                    # print('No rewiring.')
                                    foo = 0

                            # optimal rewiring
                            if rewiring_strategy == 2:
                                # 1) get the maximum(minimum) expected value in S
                                calculateExpectedRewardInS(network, i, oppActionDis)
                                expected_value_list = network.node[i]['expected_value_list']
                                minimun_expected_reward = min(expected_value_list)
                                sec_minimun_expected_reward = 0
                                if neighbors_num > 1:
                                    expected_value_list.remove(minimun_expected_reward)
                                    sec_minimun_expected_reward = min(expected_value_list)
                                # print(minimun_expected_reward)
                                # print('sec' + str(sec_minimun_expected_reward))
                                # maximun_expected_reward = max(network.node[i]['expected_value_list'])

                                # 2) calculate the index of each neighbor in S' and get the maximum index Z
                                # neighbors_num = network.neighbors(i)
                                # calculateIndexInS_(network, agents_num, i, oppActionDis, rewardDisMatrix, c, sight)
                                calculateLambdaIndexInS_(network, agents_num, i, oppActionDis, rewardDisMatrix, c,
                                                         sight, minimun_expected_reward,
                                                         sec_minimun_expected_reward, neighbors_num)

                                # print(network.node[i]['index_z_value_list'])
                                Z_m = max(network.node[i]['index_z_value_list'])
                                z_max_index = np.array(network.node[i]['index_z_value_list']).argmax()
                                # print(Z_m)

                                # 3) if the max_index > maximum_expected_reward(minimum_expected_reward), do rewiring

                                if Z_m > 0:
                                    # print(Z_m)
                                    # do the rewiring
                                    calculateExpectedRewardInS(network, i, oppActionDis)
                                    worst_agent_index = np.array(network.node[i]['expected_value_list']).argmin()
                                    worst_agent_no = network.neighbors(i)[worst_agent_index]

                                    # 2017/08/16
                                    # guarantee each agent has at least 1 neighbor
                                    # quit the rewiring if worst-agent only has 1 link
                                    if len(network.neighbors(worst_agent_no)) >= 2:
                                        # print('  Unlink the agent: ' + str(worst_agent_no))
                                        network.node[i]['BL'].append(worst_agent_no)
                                        network.remove_edge(i, worst_agent_no)

                                        # print(network.node[i]['index_z_value_list'])
                                        # print(network.node[i]['S_'])
                                        max_agent_no = network.node[i]['S_'][z_max_index]
                                        # print('  Rewire the agent: ' + str(max_agent_no))

                                        network.node[i]['ac'] = network.node[i]['ac'] + c
                                        network.node[i]['S_'].pop(z_max_index)
                                        network.add_edge(i, max_agent_no)
                                        # update information
                                        # reveal the reward matrix of the unknown agent and update
                                        revealTheUnknownGameForEdge(network, agents_num, rewardDisMatrix, i,
                                                                    max_agent_no)

                                        # print(network.node[i]['expected_value_list'])
                                        # print(network.neighbors(i))

                                        network.node[i]['rewiring_time'] = network.node[i]['rewiring_time'] + 1

                                        # print(network.neighbors(i))
                                        # print(network.node[i]['S_'])
                                        # print(network.node[i]['BL'])

                                else:
                                    # print('No rewiring.')
                                    foo = 0

                            # here is the never strategy
                            if rewiring_strategy == 3:
                                foo = 0

                        else:
                            # print('No more available potential neighbors.')
                            foo = 0

                    else:
                        # print('Agent ' + str(i) + ' is isolated.')
                        foo = 0

            # gaming phase
            # print('Gaming phase.')
            for i in network.nodes():
                # print('Agent ' + str(i) + ' on gaming.')

                neighbors_num = len(network.neighbors(i))
                if neighbors_num > 0:
                    # 1) randomly choose a opponent in S(choose the best opponent)

                    # print(network.neighbors(i))
                    oppo_index = random.randint(0, len(network.neighbors(i)) - 1)
                    # calculateExpectedRewardInS(network, i, oppActionDis)
                    # oppo_index = np.array(network.node[i]['expected_value_list']).argmax()

                    # print(oppo_index)
                    oppo_agent_no = network.neighbors(i)[oppo_index]
                    # print(oppo_agent_no)

                    # print('Interact with agent: ' + str(oppo_agent_no))

                    # 2) calculate best response
                    my_action = calculateBestResponse(network, i, oppo_agent_no, oppActionDis)
                    oppo_action = calculateBestResponse(network, oppo_agent_no, i, oppActionDis)

                    # update opponents action model for both 2 agents
                    updateOpponentActionModel(oppActionDis, i, oppo_agent_no, oppo_action, iteration)
                    updateOpponentActionModel(oppActionDis, oppo_agent_no, i, my_action, iteration)
                    # updateOpponentActionModel(oppActionDis, oppo_agent_no, i, oppo_action, iteration)

                    game = network.edge[i][oppo_agent_no]['game']
                    # print(game)
                    # print('action pair: ' + str(my_action) + ',' + str(oppo_action))
                    r = game[my_action, oppo_action]
                    # print('reward pair: ' + str(r) + ',' + str(r))

                    network.node[i]['ar'] = network.node[i]['ar'] + r
                    network.node[oppo_agent_no]['ar'] = network.node[oppo_agent_no]['ar'] + r

                    # print('-')
                else:
                    # print('Agent ' + str(i) + ' is isolated.')
                    network.node[i]['ar'] = network.node[i]['ar'] + iso_damage

            # rebuild phase
            # TO DO
            if (iteration + 1) % 100 == 0:
                # print('Rebuild phase.')
                for i in network.nodes():
                    total_reward = network.node[i]['ar'] - network.node[i]['ac']
                    # if the agent is isolated and has bad reward, we rebuild it
                    if total_reward < rebuild_threshold and len(network.neighbors(i)) == 0:
                        # print('Rebuild agent: ' + str(i))

                        # change the agent's color as a mark
                        network.node[i]['rebuild_mark'] = 1

                        # clear the old information
                        network.node[i]['ar'] = 0
                        network.node[i]['ac'] = 0
                        network.node[i]['expected_value_list'] = []
                        network.node[i]['index_z_value_list'] = []

                        # update the agent i's black-list
                        for a in network.node[i]['BL']:
                            network.node[i]['S_'].append(a)
                            network.node[i]['BL'].remove(a)

                        # rebuild the reward distribution and rebuild the oppoActionDis
                        for k in range(agents_num):
                            left = min(k, i)
                            right = max(k, i)
                            index = left * agents_num + right
                            a1 = random.uniform(0, 1)
                            b1 = random.uniform(0, 1)
                            a2 = random.uniform(0, 1)
                            b2 = random.uniform(0, 1)
                            while (a1 >= b1):
                                a1 = random.uniform(0, 1)
                                b1 = random.uniform(0, 1)
                            while (a2 >= b2):
                                a2 = random.uniform(0, 1)
                                b2 = random.uniform(0, 1)
                            rewardDisMatrix[index, 0] = a1
                            rewardDisMatrix[index, 1] = b1
                            rewardDisMatrix[index, 2] = a2
                            rewardDisMatrix[index, 3] = b2

                            oppActionDis[k, i] = random.uniform(0, 1)
                            oppActionDis[i, k] = random.uniform(0, 1)

                            # move agent i from neighbors black-list to S_
                            # let agent i be potential neighbor to its neighborhood
                            if k != i:
                                if i in network.node[k]['BL']:
                                    network.node[k]['BL'].remove(i)
                                    network.node[k]['S_'].append(i)

                                    # print('episode ends.')

        # Draw the result network and some outputs with analysis

        # 1) output every agent's ar & ac and neighbors
        # print('')
        # print('Here are the outputs below.')
        # print('----------------------------------')

        group_reward = []
        group_rewiring = 0.0

        s0_reward = []
        s1_reward = []
        s2_reward = []
        s3_reward = []

        for agent in network.nodes():
            # print('Agent: ' + str(agent))
            # print('  Neighbors: ' + str(network.neighbors(agent)))
            # print('  Rewiring time: ' + str(network.node[agent]['rewiring_time']))
            # print('  Rewiring strategy: ' + str(network.node[agent]['rewiring_strategy']))
            # print('  AR: ' + str(network.node[agent]['ar']))
            # print('  AC: ' + str(network.node[agent]['ac']))
            # print('  Reward: ' + str(network.node[agent]['ar'] - network.node[agent]['ac']))
            # print('----------------------------------')
            group_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            group_rewiring = group_rewiring + network.node[agent]['rewiring_time']
            if network.node[agent]['rewiring_strategy'] == 0:
                s0_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            if network.node[agent]['rewiring_strategy'] == 1:
                s1_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            if network.node[agent]['rewiring_strategy'] == 2:
                s2_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            # if network.node[agent]['rewiring_strategy'] == 3:
            #     s3_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])

        # print(s0_reward)
        # print(s1_reward)
        # print(s2_reward)
        # print(s3_reward)

        # print('Group average - Random: ' + str(sum(s0_reward) / len(s0_reward)))
        # print('Group average - Never: ' + str(sum(s3_reward) / len(s3_reward)))
        # print('Group average - HE: ' + str(sum(s1_reward) / len(s1_reward)))
        # print('Group average - Optimal: ' + str(sum(s2_reward) / len(s2_reward)))



        # print('Group average rewiring time: ' + str(group_rewiring/agents_num))
        # print('Group average reward: ' + str(sum(group_reward) / agents_num))
        # print('Group highest reward: ' + str(max(group_reward)))
        # print('Group lowest reward: ' + str(min(group_reward)))

        degree = nx.degree_histogram(network)
        # print('Group degree distribution: ' + str(degree))

        # colors = []
        # for i in network.nodes():
        #     if network.node[i]['rebuild_mark'] == 0:
        #         colors.append('red')
        #     else:
        #         colors.append('blue')
        # pos = nx.spectral_layout(network)
        # draw the regular graphy
        # nx.draw(network, pos, with_labels=True, node_size=300, node_color=colors)
        # plt.show()
        #
        average_reward.append(sum(group_reward) / agents_num)
        highest_reward.append(max(group_reward))
        lowest_reward.append(min(group_reward))

        s0.append(sum(s0_reward) / len(s0_reward))
        s1.append(sum(s1_reward) / len(s1_reward))
        s2.append(sum(s2_reward) / len(s2_reward))
        # s3.append(sum(s3_reward) / len(s3_reward))

        average_rewiring = average_rewiring + group_rewiring / agents_num

    print('--------------------------------------------------------------------')
    print('--------------------------------------------------------------------')
    print('Final outputs:')
    # print('Mean average rewiring: ' + str(average_rewiring/ repeat_time))
    # print('Mean average reward: ' + str(sum(average_reward)/ repeat_time))
    # print('Mean highest reward: ' + str(sum(highest_reward)/ repeat_time))
    # print('Mean lowest reward: ' + str(sum(lowest_reward)/ repeat_time))
    # print('--------------------------------------------------------------------')
    print('Mean Random reward: ' + str(sum(s0) / repeat_time))
    # print('Mean Never reward: ' + str(sum(s3) / repeat_time))
    print('Mean HE reward: ' + str(sum(s1) / repeat_time))
    print('Mean Optimal reward: ' + str(sum(s2) / repeat_time))

    s0_axi.append(sum(s0) / repeat_time)
    s1_axi.append(sum(s1) / repeat_time)
    s2_axi.append(sum(s2) / repeat_time)
    # s3_axi.append(sum(s3) / repeat_time)


x_axi = c_list

print(s0_axi)
print(s1_axi)
print(s2_axi)
# print(s3_axi)

# plt.xlim(0.0, 1.0)
# plt.ylim(-0.1, 1.0)
plt.plot(x_axi,s0_axi)
plt.plot(x_axi,s1_axi)
plt.plot(x_axi,s2_axi)
# plt.plot(x_axi,s3_axi)
plt.title('(100,4,12) - 33% - diff-strategy')

plt.show()


# print(average_reward)
# -----------------------------------------------------------------------------

# 2017/05/23
# do the researches on the situation that agents with different strategies, in the same environment

# test 1
# (50,3,9) - 25% per strategy - 30 iterations for 1000 rounds - c/sight = 20.0/100
# Final outputs:
# Mean Random reward: 778.101169229
# Mean Never reward: 1235.13120488
# Mean HE reward: 1058.3992298
# Mean Optimal reward: 1305.19335037

# test 2
# (50,3,9) - 25% per strategy - 300 iterations for 1000 rounds - c/sight = 0.0-100/100
# [899.498741895889, 864.14331582567218, 824.16265477981608, 816.84319965375721, 792.18799004440575, 767.23524383940776, 756.72203057291233, 739.97555895905714, 716.20248334406233, 685.44343746801417, 666.66651217699405]
# [1103.1151345758128, 1062.7934044420474, 1043.7489906517774, 1027.8845199432399, 1031.4342843890761, 1051.304223768075, 1089.6895032550799, 1129.0847043522676, 1173.4394054017939, 1195.2522797829886, 1208.4036959988514]
# [1303.8301316391085, 1308.8144797599391, 1303.9929926574553, 1305.9787167483942, 1288.3790476133606, 1285.4180678006944, 1284.9117097637936, 1279.7962771409714, 1264.5681652902149, 1267.6435953677787, 1247.7486048651872]
# [1213.2574846714424, 1224.4179947973646, 1224.7367353035302, 1221.47463606819, 1213.321325992436, 1218.4029520726597, 1238.513348161234, 1230.9145015242439, 1230.146766507414, 1239.0202709468003, 1211.4756848278607]
# [635.77128879897691, 600.46125692669057, 569.91648954601681, 538.02479923567125, 501.93804825024546, 472.07301953078382, 440.6447890962931, 414.67107799831132, 372.89253164061199, 348.25309047734709]
# [1219.1379646440771, 1209.4868075169406, 1203.3378863592875, 1192.1681647083335, 1176.5776485668155, 1176.854314320391, 1162.12600781848, 1163.4462302251864, 1160.3647550048654, 1156.4529077640439]
# [1224.6371186133076, 1212.1473371968295, 1190.1991029278108, 1183.9572079150021, 1172.2638193687903, 1165.6196451337776, 1159.6350248707608, 1176.2054977582336, 1164.2883021064856, 1163.4283747682068]
# [1208.5521279789286, 1189.0172312285583, 1189.1664049641411, 1189.4926696026751, 1164.7018848141799, 1166.4596895900856, 1159.9666797325885, 1170.6886481917429, 1161.3270355453683, 1161.7183932648359]

# test 3
# (100,4,12) - 33% per strategy - 300 iterations for 1000 rounds - c/sight = 0.0-100/100
# [898.82083278227185, 855.80345614546536, 820.39420848073928, 782.35671950080598, 748.57203024015973, 716.95566613220944, 683.56798873482114, 655.83823691677253, 621.74176244123498, 589.8166657909369, 555.71023396262433]
# [1138.3145101808741, 1098.8765719788789, 1059.1600671659553, 1034.5995152307833, 1022.7864410428743, 1031.9730956860938, 1057.5551022678826, 1092.4448191736922, 1128.7106014187766, 1162.8972494064174, 1191.1783059622719]
# [1284.0626505264697, 1271.3448491166405, 1267.0764882318626, 1255.112543235751, 1256.646570134126, 1256.2924127018387, 1254.9062173095178, 1253.9657296105986, 1247.3139326094713, 1243.4994706628254, 1225.2331972588579]
# [507.96546206539313, 474.46790343712792, 426.5098940540629, 382.49488472340414, 336.89369918606479, 291.30236108787375, 253.03852196490388, 207.55166757428009, 169.2778154531369, 127.48371231958899]
# [1195.4369255004424, 1188.8061784639442, 1176.2065300553579, 1167.8621189928269, 1149.9524609489786, 1142.1690527540254, 1136.6162277936203, 1130.065585839031, 1126.583844063693, 1128.3997225257676]
# [1203.6498012667093, 1190.7759250952049, 1169.0636085526639, 1155.0524693939067, 1140.1285257251634, 1130.4076201804296, 1131.5768189566604, 1130.704325411255, 1124.4317925230789, 1125.512874722355]

#2017/08/22

# [965.74120753361615, 924.30665789290754, 877.56590468522893, 844.14719235938503, 798.08408757021209, 762.1189979902914, 724.28016725396606, 690.15154813116817, 646.34960188582056, 607.53393151733201, 564.05969908982627]
# [1178.8713314196279, 1140.7276926401487, 1105.7453249773221, 1082.3910259715844, 1063.2708742218131, 1057.5079094919624, 1074.7628547606621, 1094.0444125251051, 1131.9081691767831, 1163.8894622007588, 1184.2509280197867]
# [1307.2124909908016, 1288.7634393110209, 1267.5276847965222, 1241.2908860028144, 1223.9058998908795, 1213.9473694916014, 1203.9224930037619, 1205.4645442799756, 1199.5131869442821, 1194.7340958013615, 1191.0269036992927]

# [525.76368923440123, 475.90689066536413, 433.13984475000382, 386.12380008672858, 344.06586113869832, 298.34089094002775, 260.43278284888839, 216.36418393027154, 177.15602975407418, 133.86308154783197]
# [1185.6199670090609, 1185.1969581261219, 1171.5096647643247, 1170.2786291061245, 1149.5290443281269, 1140.5473453659513, 1136.228123651284, 1136.9040185993229, 1133.4229356093169, 1126.3083212668828]
# [1177.4946150444475, 1169.6803920488794, 1162.5872339961568, 1146.9532376140539, 1148.2528839059059, 1138.7583467435143, 1134.1006619480445, 1133.735473642329, 1131.7045727491316, 1128.9456822971392]

# [1178.8713314196279, 1140.7276926401487, 1105.7453249773221, 1082.3910259715844, 1063.2708742218131, 1057.5079094919624, 1074.7628547606621, 1094.0444125251051, 1131.9081691767831, 1163.8894622007588, 1184.2509280197867,1185.6199670090609, 1185.1969581261219, 1171.5096647643247, 1170.2786291061245, 1149.5290443281269, 1140.5473453659513, 1136.228123651284, 1136.9040185993229, 1133.4229356093169, 1126.3083212668828]

# 2017/09/06
# c = [0,400] k = 400

# [800.70508847714268, 642.14687925565556, 480.67960017553327, 319.91937616026786, 163.97064445467146, 5.0370284013901525, -153.07050020709613, -311.34712153155635, -469.66953314163453, -631.0677507453488, -793.45371577790183, -959.62666478951485, -1127.687849403681, -1291.7091353576234, -1458.6436035975576, -1621.3353481795211, -1777.472448633175, -1940.499945033434, -2105.5067683233842, -2260.9896111151479]
# [1019.3540825030163, 869.63151015696712, 730.0860022939811, 637.98636697769064, 582.83920274406978, 583.81692455592201, 638.56548816831105, 719.93370144547544, 818.77979321556927, 904.73948066492949, 977.39537328592405, 1025.8829545660888, 1069.9595124986297, 1093.6252690953368, 1106.4990024423287, 1119.3263368539799, 1129.7812779148578, 1129.5729162727082, 1131.9516103144242, 1130.3953887614125]
# [1242.4182131211996, 1194.7617271821023, 1174.7361470205115, 1170.6391566209988, 1174.0280711777552, 1178.7516663318431, 1188.9064938953532, 1194.9318933836607, 1190.5442286926766, 1186.0787676454352, 1178.5993645920332, 1166.2521640952941, 1160.9129504653545, 1154.6900193542813, 1145.1148272528301, 1137.2616947982069, 1134.2393393043844, 1129.989570597656, 1132.3033437889512, 1128.3932001291132]
#
# [-793.45371577790183, -959.62666478951485, -1127.687849403681, -1291.7091353576234, -1458.6436035975576, -1621.3353481795211, -1777.472448633175, -1940.499945033434, -2105.5067683233842, -2260.9896111151479]
# [977.39537328592405, 1025.8829545660888, 1069.9595124986297, 1093.6252690953368, 1106.4990024423287, 1119.3263368539799, 1129.7812779148578, 1129.5729162727082, 1131.9516103144242, 1130.3953887614125]
# [1178.5993645920332, 1166.2521640952941, 1160.9129504653545, 1154.6900193542813, 1145.1148272528301, 1137.2616947982069, 1134.2393393043844, 1129.989570597656, 1132.3033437889512, 1128.3932001291132]

# c = [0,200] k = 200
# [958.01059957977816, 881.08646795754828, 797.66444579801498, 724.0013366787025, 639.75355382942735, 561.19338357207641, 479.79254326924888, 402.41077303719538, 327.39844145117519, 247.94769742909151, 161.23197295595185, 79.575766317358344, -1.0601179945052679, -89.829952578799748, -170.69342723558222, -255.4456882143252, -336.82880262871447, -422.97980142763862, -504.24933350612508, -587.55834453305158, -663.26871588047061]
# [1178.085220933071, 1095.5106783495653, 1029.3406836081908, 960.54643588281181, 919.44615636772187, 904.53894835067194, 915.50433766402716, 946.47652264101725, 991.56066506743684, 1043.3494364035225, 1091.4171439554873, 1120.7079826017837, 1137.7218858484107, 1137.4788327745523, 1138.0837122990588, 1138.0025682321145, 1133.9862713018051, 1129.3843943678601, 1131.2247945885936, 1134.3749717936496, 1134.139860127063]
# [1309.5600325599405, 1271.2547515734816, 1237.1772155723042, 1221.6348684186273, 1207.9319489482407, 1200.5751358709667, 1198.2353734717526, 1194.5286175134195, 1191.3901990510844, 1191.0834550412605, 1185.3916969979605, 1178.5556913623707, 1167.9386844987246, 1160.4159423700708, 1147.185832136026, 1146.7546526101551, 1138.0994958090298, 1132.9430511081803, 1127.7373980988207, 1129.2641927868065, 1133.1860115340382]
