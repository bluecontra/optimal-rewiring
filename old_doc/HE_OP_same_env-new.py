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
def initialNetworkTopology(agents_num, neighbors_num, bandwidth, percent):
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
        if ran < percent:
            N.node[i]['rewiring_strategy'] = 1
        else:
            N.node[i]['rewiring_strategy'] = 2
        # N.node[i]['gaming_strategy'] = 0

    # ensure at least one for each strategy
    N.node[1]['rewiring_strategy'] = 1
    N.node[2]['rewiring_strategy'] = 2

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
c = 40.00
sight = 200
rebuild_threshold = 0.0

# 0 for random, 1 for highest-expect, 2 for optimal, 3 for never
# rewiring_strategy = 2

average_reward = []
highest_reward = []
lowest_reward = []

s0 = []
s1 = []
s2 = []
s3 = []

average_rewiring = 0.0

repeat_time = 300

foo = 0


pc_list = []
for x in range(0,11):
    pc_list.append(x*0.1)

s1_axi = []
s2_axi = []

# print(c_list)
for pc in pc_list:
    print('For pc - ' + str(pc))
    s1 = []
    s2 = []
    for repeat in range(repeat_time):
        # print('Repeating: ' + str(repeat))
        # initial the network and unknown reward distribution
        network = initialNetworkTopology(agents_num, neighborhood, bandwidth, pc)
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

        s1_reward = []
        s2_reward = []

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

            if network.node[agent]['rewiring_strategy'] == 1:
                s1_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            if network.node[agent]['rewiring_strategy'] == 2:
                s2_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])



        # print(s1_reward)
        # print(s2_reward)

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

        s1.append(sum(s1_reward) / len(s1_reward))
        s2.append(sum(s2_reward) / len(s2_reward))

        average_rewiring = average_rewiring + group_rewiring / agents_num

    # print('--------------------------------------------------------------------')
    print('--------------------------------------------------------------------')
    print('Final outputs:')
    # print('Mean average rewiring: ' + str(average_rewiring/ repeat_time))
    # print('Mean average reward: ' + str(sum(average_reward)/ repeat_time))
    # print('Mean highest reward: ' + str(sum(highest_reward)/ repeat_time))
    # print('Mean lowest reward: ' + str(sum(lowest_reward)/ repeat_time))
    # print('--------------------------------------------------------------------')
    print('Mean HE reward: ' + str(sum(s1) / repeat_time))
    print('Mean Optimal reward: ' + str(sum(s2) / repeat_time))

    s1_axi.append(sum(s1) / repeat_time)
    s2_axi.append(sum(s2) / repeat_time)

print(s1_axi)
print(s2_axi)

x_axi = pc_list

# plt.xlim(0.0, 1.0)
# plt.ylim(-0.1, 1.0)
plt.plot(x_axi,s1_axi)
plt.plot(x_axi,s2_axi)
plt.title('(100,4,12) - HE vs OP - same env')

plt.show()


# print(average_reward)
# -----------------------------------------------------------------------------

# 2017/05/23
# compare the OP and HE in the same env

# test 1
# (50,3,9) - HE percent in [0% , 100%] - 100 iterations for 1000 rounds

# test 2
# (100,4,12) - HE percent in [0% , 100%] - 300 iterations for 1000 rounds
# [1000.2404923409406, 1014.7357814841927, 1008.2580973073229, 1019.4520548156854, 1020.3497228892452, 1025.0941616280575, 1020.6448467506518, 1022.6714136548977, 1020.2446358264932, 1019.4648666361315, 1017.7488864271971]
# [1389.3143094676839, 1379.1610922412046, 1369.2512430722861, 1355.0927784278288, 1343.6010572081427, 1330.8532274586262, 1322.0048763722293, 1303.6329153093645, 1295.0920072085867, 1276.1630703216122, 1284.8733997328441]

# 2017/08/22
# c = 25.0 k = 100
# [1040.059916039291, 1074.5217021265278, 1079.4097903551267, 1086.2318018459948, 1091.2431577511963, 1087.6866762936957, 1091.4984677352174, 1084.2874793361445, 1083.8613563254096, 1075.5247307848615, 1068.487325084252]
# [1206.9932372230642, 1230.359181565917, 1247.8879506543187, 1262.9393381940154, 1271.1325873561264, 1275.7917587944701, 1282.7623136446798, 1285.1803083546824, 1281.8489293588102, 1293.1238972414649, 1291.6766105487868]

# c = 40.0 k = 200
#
# [946.43659331805497, 932.3473192343597, 939.00526480752922, 943.42343793296686, 944.41240657239939, 940.70638781039122, 937.08500649234452, 934.97854333194903, 928.77188899135763, 916.4267136336432, 913.84528168217844]
# [1237.5421981910222, 1253.2145117264283, 1263.4984451264497, 1269.0432728462845, 1274.8433806044493, 1277.4640144844589, 1271.6595229137051, 1267.9919166160516, 1261.5978546475455, 1262.5413253726956, 1263.1512520991496]
