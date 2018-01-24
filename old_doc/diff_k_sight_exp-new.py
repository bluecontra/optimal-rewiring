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
c = 20.00
# sight = 40
rebuild_threshold = 0.0

# 0 for random, 1 for highest-expect, 2 for optimal
rewiring_strategy = 2

average_reward = []
highest_reward = []
lowest_reward = []

repeat_time = 200

foo = 0

# c_list = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
# c_list = [20.0, 40.0, 60.0, 80.0, 100.0]
# c_list = [120.0, 140.0, 160.0, 180.0, 200.0]
c_list = [150.0, 250.0, 300.0, 350.0, 400.0]
sight_list = []
for s in range(1, 41):
    sight_list.append(s*50)

for m in range(1, 41):
    sight_list.append(2000 + m*200)

# sight_list.append(1000)
# sight_list.append(2000)
# sight_list.append(3000)
# sight_list.append(4000)
# sight_list.append(5000)
# sight_list.append(10000)
# sight_list.append(50000)
# sight_list.append(100000)

print(sight_list)

y_list = []

for c in c_list:
    y_axi = []
    for sight in sight_list:
        average_reward = []
        highest_reward = []
        lowest_reward = []
        print('For sight: ' + str(sight))
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

        y_axi.append(sum(average_reward) / repeat_time)
    y_list.append(y_axi)

print(y_list)
x_axi = sight_list

# plt.xlim(0.0, 1.0)
# plt.ylim(-0.1, 1.0)
for y in y_list:
    plt.plot(x_axi,y)
plt.title('(50,3,9) - Optimal - k-sight')

plt.show()

# -----------------------------------------------------------------------------

# HE > 0 -> HE > minimum_expected_value

# [1174.2020932995938, 1189.232768506065, 1266.40901621426, 1329.7498698479237, 1367.5062261439487, 1389.8041687866576, 1403.3262400276476, 1409.9361458668511, 1416.2164787004929, 1419.7175266161221, 1421.2157610445333, 1422.5231024177435, 1420.2987524234568, 1422.9016429748042, 1423.2211758327937, 1424.1392539391159, 1424.1583967321474, 1424.1242906330792, 1421.6322818914596, 1424.5113273594, 1421.6101512454497, 1420.1404235379032, 1421.6761063099821, 1421.2125301242704, 1419.5789851666705, 1420.033890252967, 1418.376428494209, 1417.8774220151661, 1419.1821049864409, 1419.7327687323886, 1416.7554812592848, 1416.5976928741859, 1419.4280536723481, 1413.8430313648983, 1417.12185751944, 1414.3990778327536, 1415.7203852948921, 1415.4905691704603, 1413.8625039739161, 1414.6067484435675, 1413.9149762229079, 1412.96722987157, 1414.4667175858374, 1414.4857983605195, 1411.7961499069772, 1410.4489693638398, 1411.1208360157771, 1411.3668615126346, 1411.1099971973103]

# 2017/08/23

# c_list = [20.0, 40.0, 60.0, 80.0, 100.0]
# [[1161.8246398368531, 1160.5012196399859, 1159.8550453118053, 1166.1329602992664, 1175.6302744187083, 1192.7054230363062, 1212.2188547270755, 1223.869629366463, 1244.8245316996713, 1255.9991572229544, 1280.767279437232, 1283.6641381426323, 1296.8907192173431, 1307.8039301189037, 1311.0044259541298, 1321.3343939379431, 1331.6617768852541, 1332.3317130678197, 1335.9774893611873, 1343.0880732574535, 1346.0596709111967, 1353.9225604046189, 1348.2117270216806, 1353.9998948505558, 1357.1630429462361, 1355.0241498522578, 1355.8216327516741, 1360.722255706045, 1360.6564501839127, 1362.4051844877038, 1362.2478563353486, 1368.2993642590538, 1362.2719879939523, 1362.4866994793097, 1368.6505515994877, 1370.2568547747087, 1367.2175743958826, 1368.0405260197917, 1369.2466092653046, 1372.5584199182799, 1371.0988290400592, 1365.8929100780149, 1373.1209757383167, 1369.6767516205689, 1371.5298757696094, 1370.853998884618, 1371.5097245161282, 1369.1839886287248, 1369.9283089381204],
#  [1160.4966769416121, 1166.0836206970878, 1159.0538604111532, 1160.5951514980359, 1159.6191561007588, 1164.246973546667, 1159.3667932821636, 1164.8317037610809, 1170.1339656874247, 1165.8242200853852, 1180.1161936356457, 1179.2686428143411, 1192.1481579074818, 1196.4100668402298, 1201.3478013566973, 1206.6754695125931, 1219.0879355121078, 1226.6592530121795, 1236.4217515285327, 1243.2390308215042, 1247.1597555849889, 1258.0628287885179, 1258.6545480029024, 1272.3977939920183, 1271.6899602178532, 1275.0940716806658, 1281.0292530624581, 1282.7176952732611, 1285.0980477934979, 1291.908098107655, 1297.9415548181801, 1302.0430176222126, 1303.6545643224683, 1307.33533826589, 1314.2743278911121, 1311.0736165072576, 1304.1008042262342, 1315.860593487748, 1315.574429908015, 1313.2220657447206, 1318.625263748261, 1321.0195141832662, 1316.5349754507502, 1324.276624055856, 1318.0622944227632, 1327.5554736213535, 1328.3305342852518, 1320.4485792755142, 1325.5716125289539],
#  [1162.2338310396876, 1167.3566959242057, 1158.8780441650108, 1155.955424911848, 1161.2864198745415, 1164.4510009727712, 1157.4865889302835, 1164.0041419116915, 1166.0465543571281, 1161.0353843322168, 1159.1790337578525, 1163.7402905137433, 1163.0142424895296, 1166.2546662564234, 1169.6882963980136, 1170.8355172753606, 1172.2514256414001, 1178.3832413566499, 1185.3880577380071, 1186.0013423899816, 1189.0747007632933, 1196.5707078084954, 1202.3821843986509, 1206.6305953627243, 1211.8031181717697, 1218.079710311189, 1225.298167327654, 1227.8346139816642, 1226.9121432423567, 1233.4035437492439, 1235.6981303816606, 1241.5469987998356, 1242.4264347805486, 1249.6688907515816, 1252.6524448092619, 1253.8364084452419, 1255.3449748784553, 1262.5630778790501, 1261.4289753847002, 1262.8218452785666, 1267.9345321954377, 1269.134500455245, 1271.297438329548, 1273.4371856657222, 1272.6933839567175, 1280.4130910039949, 1276.2617386046563, 1285.5232842691412, 1283.6380942293167],
#  [1159.0709800900988, 1165.9254760518959, 1159.4450684525136, 1159.1400564932201, 1165.4200270142071, 1163.5368133244435, 1158.2169631751583, 1158.7813341639719, 1162.220904783876, 1162.4206182402136, 1165.9747798296717, 1159.3301151663582, 1159.0528643374523, 1162.9542794243127, 1160.5630700483889, 1162.3996451011496, 1161.1488795662233, 1162.5789030865908, 1164.3192020958452, 1173.7791750177885, 1165.9781868931068, 1169.0421762418523, 1172.8477533894488, 1177.58010098516, 1175.318464256565, 1176.7125020820358, 1187.8363809045884, 1189.4061206133335, 1189.5135058638778, 1193.6523844932726, 1195.9986425502173, 1199.3919737542644, 1207.6284880449175, 1209.9252025707046, 1212.123643956676, 1212.5103157029237, 1214.2177534383718, 1227.5348643208258, 1220.4633809326945, 1230.385507365836, 1227.5946378452441, 1229.992975608679, 1236.4338746641995, 1230.1249498651036, 1239.9464832876552, 1239.9487800424838, 1242.996757747534, 1240.5207445679955, 1243.7579123039286],
#  [1159.3299398823299, 1161.8552322641171, 1161.025846741474, 1162.1169537434043, 1154.942412594768, 1162.7467458836745, 1160.1008951037347, 1163.3630226740474, 1159.86563482616, 1162.2885729540396, 1162.0065171091412, 1167.0568836069572, 1164.7423230534639, 1158.6226903873553, 1160.5255735580936, 1162.5917903101297, 1166.3414817828341, 1166.2200562831072, 1161.1465682572054, 1166.3501015759318, 1160.2001439576854, 1162.3469941117555, 1168.6743404175586, 1164.1287333851492, 1160.0458548087008, 1173.3039201085783, 1168.1397454140294, 1168.9783352667487, 1172.7284079203957, 1172.7634076700365, 1175.0036838372976, 1172.3355410003692, 1179.4800619149164, 1182.2600695130411, 1185.8172341773331, 1186.7586768995184, 1190.9515974832759, 1193.3771622833719, 1192.1579348649354, 1198.9499509496748, 1193.0130310864683, 1207.8581149365962, 1200.618908399566, 1202.767848155989, 1203.101329550846, 1206.3993191098393, 1207.8313493129181, 1210.8584198846549, 1214.6538066035871]]

# c_list = [120.0, 140.0, 160.0, 180.0, 200.0]
# [[1164.8135675524243, 1160.4314382213956, 1159.9135858954758, 1162.6369598947388, 1159.9182527158084, 1155.3548538588959, 1162.3667637803037, 1164.0264788368784, 1159.375753400212, 1160.9026198061949, 1158.0325236777016, 1162.6105064374431, 1158.6157613734565, 1159.830753908748, 1161.8148717700237, 1162.8727666781335, 1159.9932584816117, 1168.3254596256693, 1164.7744674017688, 1157.2033190166551, 1160.0447708815038, 1165.0109116696037, 1163.3013379182789, 1164.4367229504799, 1160.6611708811877, 1156.8948607392681, 1169.4058810157533, 1160.5541246158427, 1163.0045337622751, 1169.5579778764547, 1163.84196831909, 1172.5843392819729, 1167.2803549342716, 1171.1085025833022, 1172.0184038221857, 1178.8183809077577, 1179.2373385857918, 1177.1275846797178, 1174.8012973822154, 1179.0955879412018, 1181.9839554369244, 1181.0452740799351, 1181.5373744889187, 1187.4667139871776, 1189.996887885806, 1192.1610407278013, 1191.5121663692128, 1191.2520490715171, 1193.5687452187908],
#  [1164.3150179497354, 1158.4566056951378, 1156.1446217027371, 1168.2048801064304, 1154.2534735990557, 1159.3262709539588, 1165.9875417133742, 1166.1198618030419, 1161.8932990251562, 1159.3016903318121, 1161.2288720423855, 1160.4271736678938, 1164.4486370706391, 1161.5338148083488, 1160.769801779805, 1162.1268866788712, 1158.1862198886945, 1161.2344576572384, 1164.1103040400819, 1161.5838110281195, 1156.5122750940689, 1162.2171139226366, 1163.0249327400386, 1162.7333217344362, 1164.8726674311599, 1163.3050008757405, 1165.6645193669817, 1160.0780452160457, 1159.0570204065666, 1161.1090313070329, 1165.8674473059673, 1162.1243548292489, 1165.6400455878015, 1164.8241065805178, 1166.0145201435566, 1169.888012226862, 1162.9574422533849, 1163.9554999390673, 1164.5453454382446, 1171.9777475320079, 1169.8963359337952, 1170.9028132354963, 1172.5912487953137, 1173.7961126742534, 1174.5757034796122, 1173.3470656055988, 1182.7587820791, 1177.3494757150502, 1178.8531749453768],
#  [1159.5483050678688, 1162.2246932194134, 1164.4484150778221, 1162.6391488875008, 1157.2466536043328, 1162.0611877904378, 1158.5782577416626, 1164.7265110058515, 1164.7325469205209, 1159.5731046516289, 1164.5074378836014, 1160.2286508571951, 1162.1488584848835, 1160.5243917514849, 1159.2457623700334, 1165.7497801511568, 1160.3068162820357, 1160.0707099700314, 1156.4060853917947, 1163.1262040714519, 1161.3150854909163, 1157.6329179553866, 1160.9983466458477, 1156.0295334675729, 1159.7346649743329, 1160.9841256198167, 1161.110516765787, 1165.3098592055703, 1164.4409177785353, 1159.2337435271477, 1162.6148815402037, 1159.9952809234162, 1157.6520846430935, 1159.8472676783169, 1163.634455862292, 1159.0181366934517, 1163.4437118249898, 1166.0402034453882, 1163.2478613506348, 1164.7090186680402, 1163.85914153506, 1169.0486913137879, 1163.4548340378542, 1167.8368163341429, 1165.0570714283961, 1174.4150612515118, 1174.5373645794941, 1168.1981677531048, 1170.64789772372],
#  [1156.6873618253296, 1160.481009075093, 1162.0108592147494, 1164.2984983711488, 1163.581853545772, 1161.0321178982113, 1164.9826041073488, 1158.742017372075, 1164.9177414633505, 1162.829941014739, 1161.0331985338346, 1162.7048730971565, 1154.9264073492877, 1157.1866907119161, 1165.2063259321355, 1160.5557042710782, 1163.832377608546, 1163.2536149011535, 1161.2546801019823, 1164.9847756213601, 1163.282384666599, 1154.2704800356883, 1160.4172594572692, 1162.9524988520143, 1163.2705547094058, 1162.4669735375883, 1157.0614500031429, 1157.8865162955005, 1161.7051558413136, 1162.3310613233559, 1164.9521922097597, 1167.7915068225668, 1161.9735083648804, 1160.7439680846358, 1165.1844741781904, 1160.7549440661053, 1163.2617161186006, 1163.1519997346668, 1160.0992984184072, 1162.7633163941257, 1164.31691810661, 1165.6782269944151, 1164.3993861086501, 1161.1178008650477, 1160.1838379185508, 1171.8843942576996, 1171.7686142812861, 1167.2104083909126, 1166.6705019765861],
#  [1163.0226940495427, 1160.5656436625438, 1160.8589492109704, 1161.8330026106921, 1161.2090381637809, 1155.6645007066095, 1165.0247326198159, 1161.2717618358956, 1158.8897150614287, 1166.7312526669527, 1159.1868862024269, 1164.0459563103741, 1165.0218823304519, 1160.3041473299231, 1160.5652736091254, 1165.0450075315769, 1164.0405118832757, 1160.6694273093308, 1163.9332423837923, 1162.9649677034058, 1159.6424697298505, 1165.2594549522053, 1160.8871441341078, 1160.4620816421398, 1163.2003710989939, 1161.2204521489664, 1162.0489641861091, 1161.0936864400583, 1163.8346595586088, 1168.7872783318455, 1164.0671984944915, 1162.2627596615089, 1160.5541020560056, 1164.7930740909512, 1161.9326142380735, 1163.8765245405696, 1161.0524604680381, 1162.5607342408618, 1161.6697229475224, 1161.8482192390252, 1160.0882368628247, 1158.6257678474187, 1159.7597216570052, 1159.8489202589155, 1160.3637329959497, 1162.3316599597317, 1167.6613388323726, 1160.7402489246467, 1166.0423470586875]]

# 2017/08/24

# c_list = [20.0, 40.0, 60.0, 80.0, 100.0]
# [[1157.4632259754087, 1160.9720305106184, 1173.632193388594, 1210.9489755873924, 1243.8096798596857, 1273.5502612380776, 1293.5585358494814, 1315.4019827380566, 1329.2014326160722, 1338.2433333002502, 1343.9076063252551, 1352.3470367550885, 1353.982840304835, 1354.3566871422074, 1361.7986235479295, 1363.3011988483563, 1364.2842885775337, 1368.705639027839, 1367.5203459450815, 1369.5075080235065, 1368.7219493195112, 1368.9371741410241, 1370.2792960224656, 1372.7364144707008, 1370.7501118042544, 1373.198530795129, 1372.1542084922964, 1374.4950514771751, 1375.4666001039898, 1375.810429852088],
#  [1160.8956701224361, 1161.2501375548236, 1157.3212357632756, 1161.5830022334762, 1167.0407070137462, 1178.0088809946367, 1191.5553594407559, 1201.6301324674846, 1216.4466198033913, 1236.4890520252802, 1250.3175394497612, 1263.539647343177, 1273.4687063868419, 1282.6183847502621, 1290.2131944378282, 1301.7359062566331, 1304.782381085801, 1306.590577431273, 1313.6362947822629, 1315.347868927021, 1319.1715073704058, 1320.2932140914211, 1318.677398611816, 1322.6906156996888, 1325.3656848091318, 1326.415232956623, 1326.6493788919597, 1325.3553904923604, 1329.2191945330399, 1331.9798026430565],
#  [1162.3299683288501, 1161.7603878763975, 1163.6650046043196, 1162.791408051296, 1163.2688263274517, 1161.1063989561012, 1166.2955549006674, 1168.8966840397632, 1170.6469880200409, 1180.2344124446245, 1190.1954884004285, 1200.295269593051, 1211.6278556505874, 1219.6288561410531, 1226.4057611043709, 1235.1074412312064, 1244.0440916908519, 1248.8407493383968, 1256.0777547987943, 1264.3380099471894, 1271.3684058453755, 1273.4126817820413, 1277.4267377919828, 1278.3977637886937, 1283.1126088231294, 1285.8531748371424, 1285.4906897022136, 1288.5030113357868, 1291.8207412893446, 1291.1212529709564],
#  [1163.2754072040832, 1162.1887150228938, 1161.2516957597816, 1161.8903838593042, 1163.1572640123625, 1160.1621041283945, 1162.2981693916311, 1164.1439189911764, 1167.5419751691136, 1164.9320709415808, 1165.9384691763778, 1172.0562854378556, 1178.0480949861435, 1184.7358306120479, 1190.3748821222234, 1201.4280685859974, 1202.1160697018818, 1211.9045458820929, 1216.2594092734355, 1225.6731263309759, 1226.8902868361822, 1232.4636164798278, 1239.2009633806824, 1244.2434378245177, 1244.3015096466263, 1250.2725888377072, 1246.4997743698566, 1253.6197293058988, 1257.9593386836316, 1259.8395927625206],
#  [1165.1352608792643, 1161.8498117219754, 1161.0241977185856, 1162.1356571193035, 1161.9859992840998, 1160.4255283457414, 1160.3128223156466, 1162.1843547210506, 1160.9421225194826, 1163.3763072283616, 1160.4039188449083, 1167.2446322534279, 1163.7345943728408, 1165.3628468537725, 1170.5360381403052, 1177.3092114292633, 1182.6164913598095, 1183.7035133627976, 1191.1883437725089, 1190.3473135840409, 1198.1851923282179, 1203.207625893411, 1210.835775861103, 1210.9480211951372, 1219.2132718819419, 1220.425447151295, 1222.9419369594286, 1223.4073568312497, 1229.3754865000035, 1232.1448892807523]]


# c_list = [120.0, 140.0, 160.0, 180.0, 200.0]
# [[1162.2543781518107, 1163.361609176143, 1160.7092164157359, 1160.1000191847495, 1162.6557123893604, 1159.5500130144974, 1160.1100037656261, 1163.0731757222466, 1163.290126253047, 1163.5459963032361, 1161.4675620179692, 1161.6098680766454, 1159.6165774389765, 1165.1744788875758, 1162.6261040059096, 1169.4172424235246, 1166.4772834058283, 1171.8801034812416, 1173.2797489340546, 1176.3401259390655, 1181.2459757742201, 1184.425349182397, 1182.1238737275291, 1189.9995666341099, 1195.8080735520018, 1198.2753373882417, 1201.6249063810985, 1204.1524699308729, 1204.555604140691, 1208.9149209449226],
#  [1160.1057309628204, 1163.014952923324, 1161.450516670858, 1163.0398176879064, 1161.74058370179, 1163.6197042784288, 1164.0524889556939, 1161.7889857039215, 1160.244481113828, 1162.1773232300861, 1162.4953629323518, 1161.5555840196912, 1161.3375369164191, 1160.8234733696577, 1159.8962107045497, 1165.2734699281968, 1162.5306364485091, 1164.4126769002637, 1165.8923716483898, 1165.9651997073629, 1169.3187430192352, 1171.6623957850109, 1172.3126119279405, 1174.9740986296881, 1182.4504866524287, 1182.7260710942878, 1181.1442511293044, 1188.6092869313586, 1187.7211812406622, 1191.5750395596503],
#  [1159.4857041879673, 1163.3966675847068, 1161.699013445374, 1160.301073992867, 1161.9442435713813, 1164.8742871609077, 1160.2665088054232, 1162.6778099005728, 1162.2113049241946, 1163.3062355108632, 1157.6602456863427, 1162.1949326100814, 1158.8103610117689, 1160.9058281361913, 1164.0562852957926, 1162.3446096905257, 1163.4206670874489, 1165.6497325844973, 1164.2175461073325, 1163.4342440476446, 1166.0863385095236, 1165.9258641273918, 1170.0304357671537, 1169.2958582535477, 1172.73875766008, 1170.6283342206568, 1172.0308323189522, 1178.6871779034532, 1181.2116314698073, 1180.4507980383692],
#  [1163.4750892480397, 1162.6389831776635, 1161.413938176049, 1163.8602166813923, 1160.4706828270992, 1159.7359405638881, 1162.111422192645, 1161.3657626107611, 1159.8874318941491, 1161.9310580418314, 1161.4742890756415, 1159.6481372402241, 1162.4559467056158, 1159.2700103678974, 1165.1173801729169, 1158.0077233285017, 1161.2078638129406, 1160.9647201019602, 1160.9085184293297, 1164.990170813737, 1164.1003472262337, 1163.2373618399627, 1164.3125547459363, 1164.2900567121442, 1161.9384329303712, 1166.5942066214996, 1170.0322972018971, 1166.0075034693466, 1169.527813611375, 1173.5222854940287],
#  [1160.4840724175563, 1160.9174188306049, 1161.0530344301592, 1162.5701571524705, 1161.693631664984, 1163.3036703471748, 1163.1363814764095, 1161.4682202240569, 1161.2967163427636, 1162.7466405489822, 1162.6734490823694, 1163.2281132163805, 1162.103579716673, 1156.8678180862478, 1160.2015283816363, 1162.4955902301556, 1162.2673290525684, 1161.6249923475552, 1160.731627228128, 1161.2443715584666, 1159.899236317169, 1163.2355742871118, 1162.479080449682, 1160.5699563601893, 1162.0549269877927, 1164.1456697475155, 1165.0646254842504, 1166.0249150952141, 1167.2286479361335, 1169.7761080893802]]

# 2017/08/27
# larger K

# sight_list.append(1000)
# sight_list.append(2000)
# sight_list.append(3000)
# sight_list.append(4000)
# sight_list.append(5000)
# sight_list.append(10000)
# sight_list.append(50000)
# sight_list.append(100000)

# [[1163.0739501384437, 1162.9537082281581, 1168.7937013924798, 1201.9810698526123, 1245.1498911939618, 1281.5935704835126, 1302.2130432116435, 1307.7415267645545, 1323.9119350200865, 1336.1135092147786, 1341.4227841426791, 1345.377982620779, 1350.8585291053353, 1361.1855039028057, 1364.0202297206968, 1368.823584869722, 1359.0201873634931, 1367.647730093614, 1359.3418023826184, 1366.362642004483, 1371.3551876611543, 1362.2811830703854, 1366.2671714943403, 1367.3614989458865, 1371.7398576505427, 1370.8877129994275, 1367.4018456943397, 1372.3824284032698, 1376.742709266369, 1371.3507025034019, 1375.9313504041709, 1378.3483837465988, 1375.5796837982564, 1382.8689625679533, 1380.5744018715404, 1378.0183723879368, 1377.4399955580682, 1375.2959179274942],
#  [1158.6165179031896, 1166.5223833772502, 1154.9436546752668, 1160.7817631267321, 1170.0166873624039, 1179.5564956424482, 1196.2039144589739, 1208.5161763953533, 1215.910587635052, 1245.2017955762158, 1249.4673412212401, 1264.0884528529027, 1277.5291156828209, 1283.6317942033797, 1279.0757830191658, 1292.0630631814022, 1296.7108134587429, 1311.4663821956808, 1317.8179640825097, 1310.7477631314023, 1317.71902255702, 1312.9049119030121, 1320.0080944264585, 1325.6700942760572, 1315.4723790818034, 1328.2385330105542, 1334.3503529934903, 1326.4141962702868, 1335.4479990340064, 1330.4013124706544, 1342.0324070908589, 1333.2342958107549, 1334.8794567237319, 1329.2901958539098, 1328.0558257761243, 1322.3202174693158, 1334.8271012574635, 1330.70004479096],
#  [1164.7553535929776, 1154.7166731144737, 1167.7862581765733, 1160.7951898102556, 1162.8782755852956, 1168.797265282614, 1177.6223949705791, 1166.4292955433552, 1186.9575069265873, 1186.2844471585665, 1188.2640870715368, 1215.0460126605753, 1208.7442514798759, 1227.442256807494, 1229.549730406434, 1242.2496332525166, 1242.2234480979414, 1251.1767710074373, 1265.3453270756916, 1268.5227718192336, 1272.3619995009269, 1266.7713546112032, 1283.3339949185247, 1273.5694723663046, 1281.4367212314253, 1290.1028968151879, 1284.8548200039936, 1282.1457494634435, 1291.9085235088967, 1300.0316504883865, 1294.4810208548079, 1301.0103818626437, 1284.4748412165268, 1284.4119371729187, 1285.4780197150794, 1281.9547727897082, 1283.3992394161698, 1289.9821751127529],
#  [1161.313752470506, 1161.0701758897399, 1163.4455004889726, 1157.1468759394418, 1160.6537223818814, 1157.605311938114, 1153.057323710977, 1162.6775547999018, 1166.3904369472552, 1165.5824764059009, 1167.7409524919346, 1171.6089062029459, 1172.4591693608666, 1185.4298189439191, 1199.2554327356368, 1193.7215805386998, 1195.2246825451657, 1203.6230163167406, 1220.1066868497694, 1232.4370071697001, 1220.582428238479, 1229.7437346747438, 1234.7716177299146, 1242.94239189679, 1239.1363487967487, 1251.8274721627261, 1244.7897890174172, 1255.6739005025133, 1259.929353241816, 1255.7811349634439, 1266.2521616254548, 1255.3875293097792, 1241.686478817702, 1251.4013255183177, 1237.9074597072326, 1236.4055869489669, 1230.7634247416213, 1220.0157486224425],
#  [1157.463970455879, 1169.0170322120678, 1161.0179684296706, 1158.8805802769682, 1166.6571213876855, 1164.6671776276153, 1166.0670964500202, 1158.0642855126705, 1159.5111189701688, 1159.8851968317781, 1162.1953339171087, 1161.2292101884402, 1167.5171044230535, 1175.8315673692664, 1159.3591557123725, 1175.4372110940592, 1179.3310502391198, 1182.15779379216, 1197.394156578071, 1198.5646649306675, 1196.7995348249196, 1195.6002840247347, 1199.521732140792, 1218.3277617821573, 1215.4152452812089, 1220.5781169983891, 1224.301851003537, 1218.8845708427471, 1229.4880132017893, 1235.990644274789, 1242.1839715140836, 1229.4641095269922, 1213.3824304206942, 1194.7672573392381, 1203.8426203416, 1189.656125463132, 1178.2941317575967, 1192.2223358721133]]

# 2017/08/27
# more reasonable K (not smooth)
#
# [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000]

# [[1166.0306774119329, 1240.9207517140983, 1311.4774744862657, 1332.8085062898861, 1354.2833587807038, 1358.8426321127781, 1371.263903855543, 1367.8572420192011, 1361.8298606307224, 1373.1834494252512, 1378.6874402810674, 1375.5628114887911, 1373.4029038157305, 1368.2867531462828, 1376.2307472931498, 1377.9431136857604, 1374.0441121902961, 1379.7064742333209, 1374.7735547544657, 1373.1086934688428, 1373.6165979865023, 1381.0746882257274, 1366.4050807701863, 1375.6655683854685, 1375.6801371550105, 1380.0653655708456, 1377.6567599354967, 1376.902823471685, 1376.4120802238936, 1381.1667227472888, 1378.6044886624118, 1366.8363033202529, 1379.5534852285248, 1371.5380458718605, 1383.9318580462334, 1381.1355651244889, 1378.6429395875869, 1377.6843233900427, 1375.971984463426, 1378.3350173490844, 1385.4812456147336, 1377.5819612874548, 1373.1684066875471, 1378.0126793398815, 1373.7231459483357, 1380.1787300386225, 1374.4762107584174, 1369.0285579233646, 1370.0759878567872, 1372.9934130794722, 1376.0178684108807, 1370.4551045872981, 1375.2850423207028, 1372.6144223190327, 1372.7173945287032, 1371.337433826842, 1372.1797825110646, 1371.9803299386172, 1375.4458747224448, 1383.740402582064, 1369.1107248095873, 1365.965587584823, 1379.8550346136558, 1375.5318143780376, 1369.3111722245626, 1375.8184565111378, 1373.9614223634971, 1374.3501553675844, 1366.28210579243, 1380.3960695714786, 1378.879930516784, 1373.4053836600758, 1374.6646253875515, 1371.7219566314329, 1378.6445475271487, 1365.6860181272716, 1376.7988306078587, 1375.1225483966591, 1371.1877570652864, 1373.9036800719523], [1156.0589540050871, 1161.0230167697414, 1197.5351413336653, 1229.235476919386, 1264.5394651120189, 1294.5366426863939, 1310.5796658188754, 1318.2793317039091, 1315.6075641320067, 1321.3856583623269, 1330.4207910705779, 1321.5775075950583, 1329.3179128475115, 1333.3217590613053, 1331.8794199687288, 1335.7251137840492, 1334.7776781636735, 1335.0444109779869, 1330.3262572253889, 1337.6746593461028, 1323.741317124425, 1340.5109744050742, 1338.0420891364327, 1326.497749000094, 1330.6393365164536, 1342.6075399868164, 1332.6218447385911, 1337.4929032240668, 1329.8317511962334, 1339.5254387631978, 1328.0356116620098, 1337.3934696994211, 1335.7029848950322, 1331.8512589630909, 1333.741773735637, 1328.3813537599542, 1332.8557483395891, 1334.7588615316167, 1329.0359021375596, 1338.7915864872832, 1339.6282211115008, 1328.581423339655, 1334.3933043548957, 1332.3000859877413, 1332.2396260206297, 1334.3807104102389, 1334.8918886479919, 1324.4801931817776, 1329.95693008898, 1322.7613189602023, 1324.2625623310175, 1329.6396400202318, 1334.2497142810437, 1329.6692802104747, 1336.0232889763661, 1326.6162418057997, 1331.8893323258505, 1325.5938496274516, 1332.9219367113908, 1324.0038226822523, 1324.4894148789076, 1329.8712168649142, 1331.8091920144393, 1320.6809050404256, 1330.7906628408653, 1331.0754860107415, 1329.4487033144853, 1322.4111394854865, 1319.5819706100224, 1332.7022353389207, 1323.998077379759, 1331.4010334620359, 1326.9254141885779, 1325.6797267852089, 1332.693952712601, 1332.6623251959541, 1330.0301625795121, 1327.6648199859346, 1331.3373703442326, 1322.5827832582859], [1160.3138358536883, 1158.7963540089725, 1176.7500414544691, 1183.1674978615679, 1209.9853172409014, 1235.2828611086129, 1262.789749631632, 1262.6247611464391, 1277.3669010368778, 1284.2709612258543, 1292.0205902870223, 1292.6270774789095, 1296.8048027466693, 1298.6853948898156, 1294.2426132271844, 1310.4917921634512, 1307.4705382696286, 1298.1471150178115, 1294.1772312249645, 1292.4900910455422, 1296.9536651165267, 1294.46546533896, 1298.6982531578403, 1301.0049879985718, 1298.2967023519045, 1290.1034022552401, 1291.1838743223577, 1298.6311156307181, 1301.4077259332742, 1295.5238577221744, 1295.6221075216868, 1293.7292876815818, 1295.7671522694031, 1293.826919838114, 1291.1919188542661, 1284.5391244972584, 1296.3411689392026, 1290.9690933906729, 1289.5434026622172, 1286.5000615744377, 1295.1546809481429, 1290.7098373379933, 1290.9082752600079, 1288.6062684587175, 1291.2707865958769, 1287.6303636174691, 1285.0043512171369, 1287.5185210678053, 1283.1764187237056, 1295.2791081860225, 1291.5686890518637, 1288.5629683973534, 1282.2458282421676, 1290.0174939077003, 1286.4337425212339, 1281.251722125349, 1291.1804221357218, 1291.9681178915155, 1279.2114474819573, 1284.1043250349123, 1284.5180360955617, 1283.3565558872613, 1281.9843977565981, 1279.348497811661, 1277.6948316879002, 1290.9893646984267, 1283.1015712776284, 1276.5439395005612, 1290.1384953421768, 1285.2437512222482, 1282.830963721663, 1284.3203514550992, 1279.7683327764855, 1283.3444763301109, 1285.1791076116053, 1285.86575496259, 1278.7743498743514, 1285.6732327955167, 1282.6134675005701, 1278.8962585550626], [1168.4114050900941, 1172.34597999875, 1176.9036963798071, 1169.6643351670759, 1178.0529445543918, 1194.571259222568, 1204.1056112263036, 1226.9404317735782, 1230.864724167127, 1237.226005376102, 1247.0822958486831, 1266.6750125167378, 1261.3421091820753, 1275.1985236700523, 1265.3213132088863, 1263.2810744728799, 1266.2703624355515, 1260.4001829838126, 1271.5804049926717, 1275.2396447781318, 1267.7570550810647, 1260.2185608640002, 1265.1065568449035, 1265.8527657920638, 1257.8227831616377, 1267.3830426366662, 1268.4847310441287, 1264.4407125563775, 1269.7540450261442, 1270.2360085508526, 1261.0443885500119, 1264.1205212675109, 1261.0151602043072, 1257.7712896601731, 1260.1763119202444, 1259.9726032226847, 1253.5639025822809, 1256.7611963171541, 1262.5878087424242, 1259.9162081958834, 1252.4785884300002, 1258.5821623649099, 1254.221671438657, 1256.8355476302595, 1243.4454370792268, 1254.1200836380174, 1239.401549595957, 1243.7557093104442, 1242.961403824035, 1251.238245598202, 1253.3811390168323, 1249.0460487961425, 1249.2671415903892, 1243.00117218419, 1250.7002641535764, 1226.9040941066221, 1243.9095837288667, 1244.5510817949155, 1230.1967000946402, 1235.8786076183681, 1240.861517506288, 1242.282826356327, 1244.4974302394282, 1244.5949955106155, 1241.6745008267605, 1241.8476715414083, 1241.0676603598142, 1241.3115608692792, 1240.8397231916033, 1229.7100360552645, 1239.7970053046604, 1236.2673833515739, 1240.9370224947256, 1235.1346829964432, 1244.1634704505975, 1241.232545922353, 1243.9844887866682, 1243.9591310437929, 1241.496997118028, 1226.6095612897414], [1171.1345634089637, 1158.6944805898725, 1171.2851921477331, 1161.1175966564881, 1157.237291359075, 1167.0201896180902, 1175.4816547159028, 1196.9880955785641, 1204.5533616478481, 1223.8781132037504, 1219.1913048179217, 1233.1157442105823, 1229.7262499279243, 1235.9680842989562, 1238.0177131625992, 1247.4695399775824, 1239.742498333969, 1241.4793372767931, 1233.351539287434, 1249.0634017954583, 1246.8322902684433, 1244.3869801348649, 1240.5473066661996, 1240.3434150941018, 1243.8388649894646, 1244.6576556162572, 1236.2048473623113, 1244.6403288377505, 1228.7978286177308, 1243.1199707099654, 1241.5575789863194, 1234.4102852573626, 1221.1827420509767, 1229.2103727688698, 1238.6741038499933, 1225.7414095605693, 1233.1337659451178, 1223.7778104549902, 1222.163072316019, 1236.5149077134736, 1227.2198741785694, 1212.6224280471681, 1217.5929394895745, 1218.4208526613154, 1213.5792349374335, 1205.1230836448024, 1209.0275378140163, 1200.9070195907077, 1207.4055633112171, 1205.1353388326693, 1204.7100162219203, 1204.1192856782404, 1199.2787104064289, 1195.7317935659801, 1199.6233184504147, 1200.2642999027134, 1206.0842060949954, 1205.1312158328324, 1202.4700481082052, 1196.8968911416312, 1210.9360022412773, 1198.0644898775722, 1196.5535902137644, 1194.8448386456644, 1189.9229828326147, 1190.9537873218483, 1197.0703749321442, 1188.2749224690535, 1204.13188870003, 1190.6321254246263, 1198.7291678495428, 1189.8553039685439, 1191.8018325802655, 1195.6143492382355, 1202.9948070584696, 1190.5438707902226, 1191.6043641663043, 1188.8273117834472, 1201.5225665127105, 1191.0976920907272]]

# 2017/08/28
# more iterations
#
# [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000]

# [[1167.781826719179, 1240.5559769406882, 1307.4054409286859, 1334.4608564888258, 1351.5408338564314, 1361.6285512712684, 1364.9062604584947, 1370.1931533320944, 1373.5564551460868, 1369.1089167470582, 1373.0075901653393, 1372.3146242833609, 1374.3124142975535, 1370.3872849679349, 1374.4548313186312, 1374.004583721181, 1374.3332360456425, 1373.9297762537601, 1373.9524061204379, 1374.9816932887779, 1373.7262925184268, 1377.4081669134571, 1372.1956256809065, 1373.8634375094414, 1377.4929776510808, 1373.7989083448383, 1376.4850429097146, 1377.6332002678737, 1375.9173864280424, 1377.2566968448443, 1378.7477105476457, 1378.9773142319159, 1377.6308733173023, 1376.9372941027366, 1376.2653152791061, 1375.0239213522318, 1375.4043852670379, 1373.6385361388852, 1376.1966468741984, 1376.9613794063555, 1378.5083072738839, 1378.133408969366, 1377.980022931757, 1374.9981806692583, 1375.6722866465216, 1375.814649679586, 1377.9179582949828, 1377.2007818263965, 1371.7737498097215, 1377.3031051368487, 1372.7539332501892, 1374.5297201290846, 1375.7715822186208, 1377.2608208163581, 1378.0320562688748, 1375.6383673306264, 1374.3043082108413, 1379.2592123371851, 1373.2631524806902, 1375.3333912663036, 1376.214064809835, 1375.2846341108161, 1374.4149608197563, 1375.2374510874765, 1375.0938147517015, 1375.6885873437222, 1377.1125364656214, 1375.8211474666814, 1376.2539036416524, 1372.9457659243924, 1376.0160084473748, 1373.273509428333, 1374.3759226345564, 1376.2394822472645, 1377.8542458155512, 1374.5250867422928, 1375.7157280542847, 1374.4835049151923, 1378.5158589897917, 1372.2900566899195],
#  [1160.7640702514491, 1164.4024340879268, 1200.3693548345911, 1236.2279372544936, 1272.2085594319233, 1289.0520763122433, 1305.0090405127562, 1316.625895014706, 1321.715461906826, 1321.9479725161043, 1329.3634115336265, 1332.8815939263579, 1332.4618789414267, 1326.6754898331844, 1329.0423686769091, 1330.4686595785513, 1330.9760075372369, 1331.0430187079382, 1332.8369908374423, 1332.7352039948412, 1331.6635919960033, 1332.9578256947955, 1333.9029136716431, 1331.1687929378706, 1331.7660960461692, 1331.7559396360502, 1331.489752914774, 1331.0372142174333, 1330.8678078145726, 1332.5904684807267, 1332.6007376699129, 1329.8729576479038, 1331.4975104926657, 1334.7214079466276, 1333.9579721842952, 1334.6342892150838, 1332.1274030883685, 1328.6913821849739, 1329.2825521909472, 1333.4172695857699, 1332.6526841913919, 1330.8604256418421, 1331.7433320285775, 1332.6904753392289, 1329.7709534852206, 1329.6341126507459, 1332.6425974173815, 1331.1728715656154, 1328.8366928078817, 1329.7098804213572, 1330.7971119915046, 1329.1136678485407, 1332.0087673583414, 1328.8946747581231, 1328.6521243274533, 1327.3873981416536, 1328.4147705440078, 1333.3717535146209, 1327.9736994702128, 1328.4425175337758, 1330.091226476067, 1330.3176004281841, 1331.4447712056149, 1328.8488844843105, 1331.4768612271187, 1330.802810997164, 1329.5180228414536, 1325.8728316019601, 1329.9467609566755, 1330.2484723099219, 1329.4765166279294, 1326.0987219497149, 1330.8126511769653, 1327.3304390616197, 1328.431984150116, 1328.6086536988653, 1331.733176579093, 1327.8805704252127, 1330.9941520254711, 1332.2420525831062],
#  [1158.4385076637054, 1160.6412921842241, 1167.0146337327092, 1184.2850718969128, 1208.8022389960076, 1228.2135702708249, 1247.6526540374746, 1263.4026584293167, 1273.6176989205828, 1280.4344467575138, 1284.4871096991355, 1292.0478257361103, 1293.1799362043162, 1296.9316481882781, 1296.2915841856368, 1295.0438664042433, 1297.1672876992475, 1299.3017741978952, 1300.0820951337057, 1299.7937443640976, 1299.3086086635578, 1298.7204949454372, 1297.574386473987, 1297.900282977658, 1296.8751806211585, 1294.0184034178769, 1297.003009893946, 1295.9174252394935, 1296.9845847381732, 1293.9147056931463, 1291.5723909855683, 1294.0421208294611, 1297.6031870570298, 1292.633190477967, 1293.8844632766679, 1289.2649699466776, 1291.6641862146037, 1294.0953241382267, 1293.3684410435858, 1291.1299782294293, 1290.9141429530737, 1292.9668341336949, 1292.0154350809844, 1291.7737595122107, 1289.0917794910993, 1290.2355999945291, 1287.6632871609554, 1287.3599326358071, 1285.4787663104787, 1287.0442163959167, 1286.967653530304, 1288.3514100295099, 1282.3451254618874, 1288.6127160365818, 1286.7125373006993, 1288.3963190377935, 1284.207645832053, 1284.5164574222813, 1286.6270170396151, 1284.7715422528038, 1286.2434083268981, 1284.6829289352465, 1283.867773926685, 1282.1673680764295, 1284.3094861804209, 1286.5524447535201, 1281.5180742248654, 1284.5829456299064, 1282.5042553824901, 1284.558425870131, 1280.2219747017029, 1283.8605029689156, 1282.9847914597358, 1279.4254803949477, 1285.1050413827252, 1285.4982433600301, 1283.2165616105362, 1283.5759808561045, 1283.7579752451754, 1277.016462535682],
#  [1160.7642602823084, 1162.5280786754456, 1160.7037986902087, 1167.2970266280627, 1179.1071352253002, 1191.2067023894672, 1209.1122791048663, 1219.9039914497798, 1234.4672475698803, 1248.7135809895972, 1253.3927869991267, 1255.7419834490058, 1262.4262431497691, 1265.0273994537558, 1266.0870782178331, 1267.3426249127201, 1271.2552778948821, 1268.6267955469762, 1269.8248062348368, 1270.913768967459, 1269.5917122823823, 1268.2566599582183, 1268.110332787759, 1264.0910218064357, 1265.4075649940708, 1263.2586708651593, 1264.8833375076358, 1265.5562902155548, 1262.4060339877572, 1263.9179492503388, 1259.8759220373511, 1261.1951514828911, 1261.7844557366409, 1261.0321307538275, 1259.2498785397106, 1257.4830266637689, 1254.924701311784, 1256.9942863971987, 1256.5751533329073, 1251.6599179601913, 1256.4428279536005, 1255.4921787735364, 1249.5112259480729, 1250.0129047574169, 1250.472387974974, 1252.0753573492104, 1249.7714078917377, 1246.9297059046978, 1249.570319296914, 1241.1419973789552, 1247.5561649499325, 1243.573316664093, 1244.2098810222924, 1242.2475451214925, 1243.1774970249339, 1243.4209970621812, 1250.1757207575254, 1240.2750229465175, 1238.3310259462335, 1239.5527282149785, 1242.2696732276934, 1242.5080865271805, 1238.2311818151704, 1240.8676698071113, 1238.4643746615291, 1238.8109246420775, 1238.8138322370764, 1238.2004397428523, 1235.3515295440357, 1237.8840267526241, 1237.7142869844172, 1241.1177623409819, 1239.4228639012422, 1235.359620829709, 1235.8669591405319, 1239.6100967984169, 1240.0153018065587, 1236.4833367505182, 1236.8688686262085, 1237.7232364340446],
#  [1162.719641804179, 1160.5785561998955, 1162.7389714855815, 1160.5363131205575, 1167.2363825334205, 1173.4726444964281, 1182.9262980562621, 1194.7862641370509, 1205.8738330266874, 1215.1978677755135, 1226.0952420973538, 1229.7928319900429, 1237.3438939053556, 1237.3620398245478, 1239.8689634174305, 1245.5102694555992, 1243.9452241231413, 1248.6501265947343, 1246.2940343752236, 1242.1727049763608, 1239.7031697248242, 1244.8502502649853, 1242.2671627320462, 1245.0257158092672, 1242.4016060726201, 1240.5294899212076, 1241.7937514872965, 1236.2157828134668, 1239.7791425188666, 1237.4362162199911, 1234.4428758172764, 1231.8693562554499, 1234.5093439323934, 1231.9426801759573, 1231.4021175299501, 1229.6198234395893, 1228.4069093667924, 1222.6537021218601, 1226.8942463772894, 1226.4399365211375, 1223.7101907512397, 1219.8004335077467, 1215.517689179002, 1215.7594260216408, 1215.2918081850887, 1211.8656610477663, 1211.3506991398783, 1208.1792900746191, 1207.6771083861559, 1203.9381679633291, 1204.8543992485104, 1201.7948175101014, 1208.8805580635899, 1202.4809512266545, 1200.5743162102228, 1200.4575897261152, 1199.3319628526538, 1200.9932913368614, 1203.9587058812185, 1197.1840406142585, 1197.3332381754337, 1199.7585814982867, 1196.7771365951203, 1197.4804058873178, 1200.4009886082897, 1197.0556082461733, 1198.4742959526816, 1198.4304433570783, 1200.6345362961886, 1197.9377872676089, 1196.0962768704464, 1195.9645203590201, 1192.3728935470717, 1195.041132331218, 1191.174036902037, 1194.5742650120044, 1196.6818130949134, 1196.2732755116458, 1195.3800573686374, 1194.6072903918478]]

# 2017/08/29
# larger c [100,400]

# [120.0, 140.0, 160.0, 180.0, 200.0]
# [[1162.7598627292109, 1166.4493327018815, 1163.7291595647241, 1161.5642305012018, 1167.4767025496708, 1164.4205629091989, 1170.407431204145, 1179.1196834410609, 1187.0221132766578, 1195.172096006203, 1201.1173943230806, 1209.4402978314397, 1215.6657769490005, 1213.4846379358526, 1221.3961148295014, 1221.7956171593778, 1224.8503511917172, 1222.4912318873562, 1226.0056525152102, 1225.0787303871227, 1223.5056468134194, 1227.7735845911811, 1223.1829649455897, 1220.1666908975933, 1220.1507962104154, 1219.7895346807702, 1218.5703775263255, 1217.3738694433598, 1216.026333386943, 1216.7579147471367, 1211.2140886227216, 1212.0005534480511, 1211.7450087308821, 1204.7700660428461, 1204.3915890164417, 1203.4982059105243, 1198.3835212960125, 1201.6065824109055, 1201.5840790617558, 1196.9599797481669, 1195.4941151323007, 1191.6224721037847, 1179.585802225939, 1181.2456548279988, 1180.4352489909261, 1176.5851629063991, 1173.7367375004494, 1171.434049602432, 1172.2175122050573, 1167.0160871007463, 1168.6992011689147, 1166.8793614747617, 1163.4874906380498, 1164.0774388521702, 1165.6671152694171, 1161.1388440251092, 1159.225156865022, 1158.7441212019446, 1160.4787210456454, 1162.2492257499684, 1161.4614069560166, 1158.1799260231064, 1156.1534797767315, 1157.890156935777, 1153.6114164761809, 1156.1173799607307, 1152.2086566096516, 1153.0487374118802, 1151.4039699740172, 1151.204380811507, 1150.7816437694291, 1152.0885784825994, 1152.738905244179, 1152.8331996708694, 1151.855770373448, 1149.8395614545705, 1152.7595672195087, 1149.4093547387058, 1151.697348597319, 1149.5108264386599],
#  [1159.6776540008168, 1165.6034131343195, 1164.3201075447528, 1160.6549533468158, 1162.5127774258874, 1161.930214917229, 1166.3922511116421, 1167.8023444515459, 1171.2191479446008, 1180.0768539306771, 1183.4616159492173, 1192.270820471547, 1199.7051536707111, 1204.1082945015739, 1203.6502982044308, 1210.0353441182656, 1206.9742228715952, 1213.0204627187063, 1208.9932504115154, 1211.0911691858062, 1209.0429151879041, 1209.7649229424728, 1208.7563867446881, 1207.1353334546759, 1205.9623501273384, 1204.1850552281642, 1203.5130938161055, 1200.5647728827869, 1199.9248358339992, 1194.4450978266595, 1192.6171623596902, 1190.4259284136201, 1190.8583923588037, 1187.7786564463509, 1186.2838193575026, 1184.9505094539995, 1183.7499812725148, 1178.5202100417819, 1179.4392436635633, 1172.3040898863549, 1167.4515892552556, 1166.3929378642479, 1162.2659167413708, 1152.3419913782991, 1149.6788573716497, 1143.568787680307, 1140.2998803337096, 1139.3860559660823, 1137.195290951242, 1135.8684113898921, 1131.286924650406, 1133.3119468491395, 1129.7380262395357, 1128.3316892950479, 1127.7707574997569, 1121.8957926318953, 1123.655885937013, 1122.7198572774621, 1118.4501765089171, 1118.5122057758829, 1119.0119870077872, 1119.8520119481334, 1115.3513155921928, 1118.2819533796528, 1117.4157737410587, 1114.7629978851976, 1111.8549324608312, 1111.3839476548521, 1116.4106644766878, 1114.7910412629165, 1112.3946616692137, 1109.3745336165987, 1109.6484414627967, 1108.9644311221737, 1108.9034069310112, 1106.670039683891, 1108.744689025996, 1109.3773676015232, 1105.8792384107032, 1106.2328602753216],
#  [1161.3932837814139, 1159.5672484445454, 1160.3910094441007, 1160.6711506142306, 1161.6488752757109, 1157.6957422555784, 1161.4685961177272, 1164.6764266209759, 1167.9440255933093, 1167.5342111531902, 1175.5526719424045, 1178.0930263458747, 1186.6570326561518, 1185.5496118543954, 1194.0232281617996, 1193.0956076433949, 1193.5044708955172, 1197.6588267282307, 1196.4904482079494, 1200.341622588991, 1198.1388932166915, 1194.7581147126498, 1192.5946373189827, 1197.0047969195753, 1193.2434379487549, 1191.569531471454, 1191.1133124735854, 1190.6265836325658, 1186.0175515857334, 1181.1608178217496, 1178.2052876837195, 1177.8771836370379, 1176.2162920177668, 1176.3222416933268, 1166.8845735261391, 1169.3536401662327, 1161.4345244132021, 1163.8181070838402, 1160.9914006428962, 1157.9036156999682, 1148.2654167470885, 1139.8219951585897, 1134.8989290525576, 1128.6849975782395, 1124.223348752151, 1116.6329465114193, 1113.5746795417974, 1110.228364193488, 1109.4825725176322, 1102.5476087981986, 1103.5064169088673, 1094.2601948401982, 1095.2676392350093, 1092.5202455212975, 1092.1075584971959, 1086.9663923335188, 1088.839919114554, 1081.9150443081492, 1084.0439859311202, 1081.0559284132514, 1081.4244162687539, 1080.8735815258824, 1080.925725737246, 1078.4791736013631, 1073.3390876175965, 1080.1960103695719, 1069.3698801619437, 1073.9699863909548, 1071.2275846425732, 1075.8857631717026, 1068.9827939348188, 1066.5392901447938, 1066.4498787902135, 1064.2109996471515, 1069.2381312223092, 1066.341638506852, 1066.3412159206696, 1064.1009182516418, 1065.7657165985204, 1066.5320576647737],
#  [1163.2996005273458, 1160.8449911115711, 1160.5024466758221, 1161.624534202112, 1159.2967406993632, 1163.3115124761357, 1162.8319326840422, 1160.6595337347169, 1164.7754357244696, 1164.2895438351316, 1168.101501876647, 1169.9804020275942, 1177.5974790905348, 1177.7681963628227, 1186.9136405400893, 1185.2257621636484, 1187.1686685947591, 1187.3922723483695, 1188.0932304565806, 1187.3583973606892, 1188.0906211758511, 1187.69341986159, 1187.0035120692116, 1186.164278307539, 1181.75083364482, 1178.9898626443028, 1180.1007960519303, 1174.5206155357907, 1177.8582601809603, 1174.3270968643085, 1169.3180136038663, 1165.4438132869559, 1164.5141788620399, 1164.9090753817582, 1157.2689396900412, 1154.8918666101467, 1152.2914035939307, 1147.2307336335648, 1150.1560971265408, 1144.3061694040857, 1129.9121225327374, 1125.0830663064846, 1113.2196655545999, 1107.6580666136006, 1100.8839980487044, 1095.9053301465196, 1088.6808538878661, 1086.1186734828395, 1075.8007070770627, 1077.6312618831525, 1070.1114679523268, 1065.810941406997, 1063.7352782199309, 1062.0740795640008, 1058.7449293668835, 1052.9302460583817, 1052.4532500791977, 1048.2522406221651, 1048.4584675303111, 1050.2000735609724, 1044.0752012614178, 1041.8729561439748, 1040.5005803108313, 1041.888264813206, 1035.736014472041, 1039.5583293642123, 1039.5128870129195, 1031.5582117375068, 1031.912044018135, 1028.9531848490144, 1029.3290816119918, 1031.4056131739867, 1029.4310197193422, 1032.4569205863024, 1026.7682102189187, 1028.5829302307932, 1028.7483697726059, 1022.7428882546722, 1029.61609027389, 1023.1390063314548],
#  [1161.5610027664961, 1161.1125324280486, 1164.0854274833794, 1162.779249825212, 1160.5294776824587, 1161.2945768516424, 1161.2398674721001, 1165.1065014323838, 1162.8031232354078, 1162.6983111997361, 1164.5470744221611, 1168.0322314632579, 1168.5533585601681, 1171.6759809697658, 1172.540342157168, 1181.8926478513808, 1174.8984801281179, 1178.2767494435122, 1181.8106851223138, 1179.0975837326623, 1182.3223229172518, 1181.2709285499959, 1176.7133011538272, 1175.0876020701035, 1175.3151247064191, 1174.8290281414704, 1172.7241788253184, 1171.0640869595447, 1166.85258460303, 1155.9077390805367, 1158.5114821090506, 1154.9605014722586, 1153.0766312344033, 1155.5999993063581, 1149.5082370744128, 1143.658843651034, 1140.4942534962834, 1138.1170122150072, 1135.8594296920471, 1129.0826488355362, 1120.6191572346265, 1106.7801740390321, 1094.3154418181032, 1086.9833009399902, 1075.3794793284517, 1070.6919837593985, 1060.3275537688919, 1056.8078339467006, 1046.1606702118161, 1043.9846394980568, 1039.0044664647041, 1035.4882100943025, 1030.9530650996871, 1031.7402776703659, 1023.4610543615537, 1023.3131414044105, 1019.0067458547207, 1020.9781145664653, 1016.8186893735557, 1012.6374098910144, 1011.393722054761, 1007.2520913812409, 1009.3585415639529, 1003.7449119567895, 1000.2109840441387, 1005.4307827415018, 1000.1024049086478, 996.05564883104478, 999.33012240406617, 997.19224063047443, 994.90316173787721, 996.6381363249078, 991.52473269523568, 992.6215923106729, 989.24288841598172, 983.65342372660064, 986.18645335858298, 986.34249579840355, 991.15972334624234, 986.39037125656114]]

# [150.0, 250.0, 300.0, 350.0, 400.0]
# [[1164.8277359624333, 1163.1972463776171, 1164.2113306937847, 1163.1701568011022, 1163.2056807204965, 1161.247685821319, 1163.6909771647565, 1164.3251573011744, 1170.3726306124265, 1173.4791601844954, 1183.5780842194119, 1183.3801718392149, 1191.7145902660036, 1194.5655489234789, 1196.6063672325756, 1200.6043061083521, 1204.836991561715, 1203.7079611712163, 1205.13568067452, 1203.5513786319009, 1205.0701690136725, 1206.9987721098532, 1198.1248990275067, 1199.7254009027413, 1201.8812847348968, 1197.8385141336178, 1194.2898053185838, 1196.7287053695636, 1191.4718736217746, 1189.4688986369761, 1187.0836577588677, 1183.3861167787618, 1183.7274298249295, 1179.9348033952642, 1177.8581512686442, 1170.8421075496155, 1171.7953480618762, 1171.4518401768096, 1170.9975138399132, 1167.7591240504103, 1159.9247840790274, 1153.6819278958301, 1145.2105361330139, 1141.3395441548569, 1137.1126430871288, 1132.1371437554199, 1126.6742432490764, 1123.0793115461163, 1120.7788755365029, 1117.8920509008769, 1117.5654881647536, 1113.2487681168134, 1111.9497065319449, 1109.3872786326347, 1106.6859376826187, 1110.2600096553208, 1102.3866911393482, 1103.8134520776825, 1101.6822050582987, 1102.3246182244186, 1099.1677506308667, 1100.0546782921865, 1097.2431892371415, 1096.9715672647485, 1097.70156831792, 1095.4555418059667, 1094.1693080299624, 1093.420458503163, 1095.0867662576156, 1090.9883687450103, 1090.8701936882792, 1090.7868975433169, 1091.1175204088452, 1087.5788413172379, 1090.2595833815562, 1089.2831556649828, 1089.2837601040962, 1092.0771991829881, 1084.5142539217775, 1087.493694539073],
#  [1160.2936971397476, 1159.6170902609258, 1161.8733604213976, 1161.4987518197993, 1162.5083955172802, 1160.2571830691274, 1158.63821674613, 1161.2631812314785, 1161.7336880679607, 1158.5243557318674, 1164.4732241298307, 1163.4856852865605, 1162.73967718326, 1163.5125321678577, 1166.7725467526827, 1164.6154123658823, 1163.5518451080015, 1167.9471317417299, 1166.0674945050873, 1165.4236914589005, 1168.2017536691801, 1170.1607792252771, 1168.9810846411979, 1164.6172291160742, 1161.9131921703718, 1161.8939377021884, 1159.6081483889809, 1156.1080314934939, 1153.1867526864801, 1150.1617916294972, 1146.2452775635454, 1143.0874169814467, 1142.7848601314199, 1136.6636464147853, 1133.956606274081, 1131.7391449367826, 1125.0002950498979, 1121.1209467571407, 1119.0317226947977, 1115.0840589060911, 1094.9925227133958, 1081.0886522595115, 1066.0352127810993, 1049.2628338490081, 1034.4376842960364, 1025.3110331223861, 1018.0439679081491, 1010.6350007254821, 997.64639897323354, 992.87333696486064, 985.35167580665006, 976.87281060901546, 968.26937007881327, 961.14672406720058, 958.38218339773186, 956.04840769143095, 945.37972269369391, 946.93353442572993, 938.95916671713144, 938.13771902922326, 927.82112138083971, 927.59374868060695, 923.79959692305886, 919.62168938439629, 912.87945484477063, 914.76831337584383, 915.03540393103674, 917.44517714882056, 910.12720561268304, 904.48442419401078, 902.8065302388577, 904.07455102737936, 900.38801600566455, 901.54684886878476, 897.08501753919336, 900.38998242136438, 895.98253009281518, 889.92059675187454, 892.57535449252987, 893.4550061688235],
#  [1166.651684794339, 1158.3385973169477, 1162.2253174041327, 1159.0388267977801, 1163.0431544546072, 1160.6063722558285, 1161.5308665215257, 1160.6756022676279, 1162.8793412492719, 1159.2142228040823, 1161.2115537337597, 1162.9960810640919, 1159.4084920265821, 1158.9004723525597, 1161.6910177197522, 1163.6726827418204, 1161.6595075059461, 1160.5255447917857, 1160.5332229634744, 1159.6189709975999, 1160.2733154907676, 1161.9087540305143, 1157.4592485518356, 1159.8774043202479, 1157.1137066274316, 1153.7111198430696, 1151.5939732330419, 1147.8496525940027, 1146.8282874251315, 1141.7089020593237, 1142.1741199251856, 1135.2004252623904, 1133.8305935877424, 1133.3632715399935, 1125.7571148689788, 1125.5548701624564, 1120.4074391991649, 1116.7144544330235, 1111.0542820265778, 1105.5316021461833, 1090.7825843706289, 1068.7830802253238, 1054.4330354170943, 1035.9960592531913, 1018.2411025269462, 999.52034575238099, 982.90037445390055, 971.9798315650911, 961.69706499531389, 944.45318843500991, 938.18940844306269, 926.47008202999916, 913.8096534527275, 909.00879521210027, 902.11344751856291, 892.99037762287276, 888.37849178706904, 880.15764260357844, 870.6110544946464, 870.66801297259792, 861.33712528432795, 858.87443319661281, 851.62132150280422, 849.23364485037177, 848.98911379743197, 838.93153814982065, 837.76221405958972, 837.63090280003667, 835.34199810930102, 822.99723885100593, 822.18010846753941, 819.26696822667316, 820.32976383915855, 811.50854057640458, 812.00504524756457, 807.95115002965463, 803.8632571993644, 807.92467463677303, 807.04039934733271, 798.72324191185851],
#  [1161.4455898477347, 1162.6099670486537, 1158.7134966792055, 1160.5507767070317, 1160.2305428272455, 1162.1836455123564, 1163.2770299515871, 1161.630904148124, 1161.3617853913779, 1160.623933925863, 1162.1308322606083, 1163.4013295662282, 1160.8937349686448, 1160.2956783431714, 1160.2435180185826, 1159.1813803196467, 1162.0198849466603, 1161.4792946021296, 1159.8845131999408, 1158.1712558591801, 1160.6246797142753, 1154.9637888221689, 1156.5388713694431, 1157.4971410675937, 1153.7348452904182, 1151.8419818472237, 1151.4088680066757, 1149.1024696559732, 1146.2903347989718, 1142.585055880761, 1139.1489033115047, 1138.5171507173868, 1135.0666570904139, 1134.3606319776327, 1130.4389882972139, 1122.099910815054, 1121.4366407843504, 1117.6959702050458, 1113.285789555171, 1105.4921262426371, 1087.4529972168034, 1062.5167523430875, 1049.535176058379, 1024.0907151918727, 1009.9315271422814, 985.04711581624099, 963.41740833313418, 957.57186487073568, 932.41580336547122, 922.54453692251036, 907.48669867194883, 890.76413169639625, 876.52982617274847, 866.17901311442165, 854.19327603012016, 841.80158269028061, 838.76595546669637, 831.26956130094914, 819.81807226631008, 813.68854344984948, 803.70085411400748, 794.06097968599261, 790.54272702395724, 786.51600464990963, 780.55325578033148, 771.12639144012257, 770.04846930007216, 764.05640896656985, 759.45440724739194, 749.75703226816108, 752.56076371438564, 747.50961824650892, 747.65664497306466, 737.10461017352611, 740.27606538467114, 732.77407173584606, 730.45617321089526, 728.02104794449872, 725.62210316795813, 721.70609093832502],
#  [1160.8245036393644, 1157.0606214354389, 1163.5799634053592, 1162.2365859669017, 1163.0239322226332, 1160.1960112512713, 1164.6661406231383, 1162.9263043243511, 1161.5885092568196, 1160.4723676503691, 1163.2833968541941, 1163.5715612286426, 1159.1198292484542, 1158.9169502141683, 1159.4146365454721, 1161.1482562984729, 1160.0210920225434, 1164.9709409547293, 1164.476184100195, 1159.1197971483266, 1156.6998220905582, 1160.5276098383508, 1159.7367662312452, 1159.4589149516705, 1153.0471021185606, 1154.0977966270216, 1150.3448498235939, 1150.6011639421083, 1148.1612176593505, 1145.9415796227838, 1140.9105389357596, 1138.0258790428904, 1134.3203382249185, 1130.5865366771918, 1130.3542440117358, 1124.4111000448206, 1124.0173222245378, 1119.568051501926, 1117.1279397052251, 1107.6688931631354, 1092.4855878133078, 1069.3250233943102, 1052.913250039114, 1028.1466425982917, 1004.2028224263761, 982.604730346834, 963.68993292918196, 940.87617872844839, 916.76949047645303, 901.10504070914408, 877.57472296932076, 863.75607549229539, 850.74309572666334, 841.24651234236069, 824.10226911502718, 813.90735892267435, 795.71319134031671, 788.03191946321681, 771.85325269388591, 762.97669429255529, 755.11717899431096, 743.22400356419678, 736.64502798294143, 734.2763800333538, 723.96479723063896, 710.04716435437365, 704.88953657433126, 702.96953276004194, 693.28600340486639, 693.71491507556698, 681.87593322610121, 684.43154515652338, 677.88349404006283, 665.65258547912595, 659.67750455094176, 663.0548726332795, 651.77671979127035, 654.86465766259062, 644.02306340314988, 645.22509182036674]]
