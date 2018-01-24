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
# c = 4.00
# sight = 10
rebuild_threshold = 0.0

# 0 for random, 1 for highest-expect, 2 for optimal
rewiring_strategy = 1



repeat_time = 500

foo = 0

c_sight_pair = []
for x in range(21):
    c_sight_pair.append([x * 10.0, 200])
print(c_sight_pair)

result = []

for [c, sight] in c_sight_pair:
    average_reward = []
    highest_reward = []
    lowest_reward = []
    print('C-Sight pair:' + str([c,sight]))

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

        for agent in network.nodes():
            # print('Agent: ' + str(agent))
            # print('  Neighbors: ' + str(network.neighbors(agent)))
            # print('  AR: ' + str(network.node[agent]['ar']))
            # print('  AC: ' + str(network.node[agent]['ac']))
            # print('  Reward: ' + str(network.node[agent]['ar'] - network.node[agent]['ac']))
            # print('----------------------------------')
            group_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])

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

    # print('--------------------------------------------------------------------')
    # print('--------------------------------------------------------------------')
    # print('Final outputs:')
    print('Mean average reward: ' + str(sum(average_reward) / repeat_time))
    # print('Mean highest reward: ' + str(sum(highest_reward) / repeat_time))
    # print('Mean lowest reward: ' + str(sum(lowest_reward) / repeat_time))

    result.append(sum(average_reward) / repeat_time)

print(result)
x_axi = []
for x in range(21):
    x_axi.append(x*0.05)

plt.xlim(0.0, 1.0)
# plt.ylim(-0.1, 1.0)
plt.plot(x_axi,result)
plt.title('(50,3,9) - Random - per 100 rounds - 500 sight')
plt.show()
# -----------------------------------------------------------------------------

# HE
# [1040.9986914676454, 878.57192433410592, 710.78201478534027, 542.63599985135716, 430.1952590580608, 407.26197926444291, 480.4703812027347, 617.87719627832212, 777.83577983112434, 920.50332199388617, 1036.6353614326056, 1111.8747675940547, 1153.5466797774277, 1172.0613865941577, 1180.5374218752215, 1179.2126018247147, 1177.4270911976139, 1172.3178430828648, 1170.7755702160946, 1170.8007300807171, 1169.6798846920499]

# optimal
# [1437.7778279408215, 1406.4493209246514, 1375.5416899813895, 1354.0250941704621, 1333.5817654514235, 1314.0386356548763, 1296.8682398135259, 1277.772399474303, 1256.4261611808718, 1235.384219340096, 1221.496993227099, 1204.190196031733, 1192.1555734807423, 1182.6984039593565, 1173.9625124611989, 1173.5802343335179, 1173.4465688871564, 1171.0607687735087, 1171.5841009547869, 1172.9315166025019, 1172.4660347021634]

# Random
# [897.52910641649862, 735.19815709490138, 546.3787856829947, 346.50430961033368, 152.84119922826827, -11.8658888925121, -168.87894338906403, -324.78132483843166, -484.77376239381158, -640.87924729183715, -791.32148342681808, -949.30680318886766, -1106.4118468901745, -1257.1985170723863, -1410.2581619296118, -1573.0067929025734, -1725.0124127066736, -1885.4401903509961, -2030.0993925029215, -2192.2139130012943, -2353.1116279508319]

###
# 2017/08/20
# Op
# [1455.2215742157284, 1407.2160841339846, 1373.356008565095, 1313.1398787901589, 1294.2469255565193, 1233.5222788997214, 1238.9300860352619, 1210.5594054541391, 1199.1812835169364, 1199.0859513757946, 1184.1124423187889, 1201.5865469650705, 1162.2343874022863, 1159.8733297792189, 1157.3116068628237, 1164.1683309138934, 1175.5645862548658, 1198.1086913889121, 1173.6145459429486, 1141.147802128798, 1180.9675391424539]

# HE
# [1136.9349771625205, 992.89083395041928, 848.66525488983734, 745.56835212535384, 648.51166767104746, 621.44600363396626, 664.06276680560507, 740.5415264635493, 851.32807834946414, 960.69584528783832, 1038.9334231661578, 1148.0662991095553, 1172.3266905146113, 1174.7348398307563, 1175.1399433487834, 1167.809378768706, 1170.0441846214446, 1158.0505091427087, 1176.8385921163272, 1164.9185864047945, 1172.7570678516622]

## 2017/08/22
# c=40.0 k=200
# K-HE
# [1160.7635698654919, 964.24037164507467, 773.66874782670118, 607.62170795756981, 488.18592309447433, 435.04964417052349, 457.15714855694961, 544.48202965163978, 668.98277429232712, 807.39007509368105, 933.77566943951376, 1026.9803629947291, 1092.7001793165589, 1127.7230267446384, 1148.163092655305, 1154.4260274265523, 1159.4085060987093, 1162.2976104221971, 1161.0156303827939, 1161.063343885997, 1161.3284882020512]

# Op
# [1423.811742020029, 1358.6440120059081, 1303.2348862673639, 1255.1003004423126, 1216.0474832996106, 1190.4562117811624, 1174.7128998991063, 1167.4489884086317, 1163.7520201612483, 1161.7336657571718, 1162.670710712272, 1161.6966258133912, 1160.3614235404973, 1161.7064119795807, 1163.3634751487139, 1160.9372957849212, 1160.0760452321176, 1162.4732768005151, 1159.7591609357744, 1159.7340624243379, 1160.8460583015658]

# 2017/08/23
# k = 200

# Op
# [1424.1182611253471, 1387.3813242894719, 1337.2429344166635, 1283.1522162968126, 1235.81388346569, 1201.5139004958482, 1181.8741635460822, 1172.836867122676, 1162.5719955029365, 1164.1026348913961, 1162.8639067782433, 1162.2427550970583, 1162.2786521048147, 1163.9431332537156, 1161.181531435793, 1161.8530701815048, 1162.6169427970467, 1164.8337759095887, 1159.6982045282361, 1161.6794924626872, 1164.6250753788684]

# K-HE
# [1162.8239460487819, 1083.061881885488, 1007.5193163139796, 946.61883869579151, 911.99420993170725, 909.94787144051782, 940.74328862026346, 1001.7608404474825, 1074.9294165361268, 1149.7817346315339, 1206.5906410556995, 1241.0301074664544, 1249.8647328425354, 1237.5682200212134, 1219.6626115847414, 1199.0473296089758, 1182.9008362300301, 1172.375030797569, 1163.9475470845359, 1162.6761187683801, 1160.696516077443]
