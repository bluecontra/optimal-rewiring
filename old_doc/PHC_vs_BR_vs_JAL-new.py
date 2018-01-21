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

        # NEW!!
        # 1/3 to 1/3 to 1/3
        N.node[i]['gaming_strategy'] = 0
        ran = random.uniform(0, 1)
        if ran <= 1/3:
            N.node[i]['gaming_strategy'] = 1
        elif ran <= 2/3:
            N.node[i]['gaming_strategy'] = 2

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

    network.edge[i][j]['policy_pair'] = [0.5, 0.5]
    network.edge[i][j]['avg_policy_pair'] = [0.0, 0.0]

    # q_table_pair = np.array([[0.0,0.0], [0.0,0.0]])
    q_table_pair = np.array([[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]])
    q_table_pair = np.matrix(q_table_pair)
    network.edge[i][j]['q-table_pair'] = q_table_pair
    network.edge[i][j]['e-epsilon'] = 1.0
    network.edge[i][j]['alpha'] = 0.1
    network.edge[i][j]['count'] = 0

    network.edge[i][j]['delta_w'] = 0.0001
    network.edge[i][j]['delta_l'] = 0.0002

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

# update opponent model as we can record opponent's action
# update the oppoActionDis
def updateOpponentActionModel100(oppoActionDis, player1, player2, oppo_action, round):

    # get the previous experiment
    p_old = oppActionDis[player1, player2]
    if oppo_action == 0:
        p_new = p_old * 99 / 100 + 1 / 100
    else:
        p_new = p_old * 99 / 100
    oppActionDis[player1, player2] = p_new

# calculate action on PHC-WOLF
# player1 < player2
def calculatePHCResponse(network, player1, player2):
    game = network.edge[player1][player2]['game']
    policy_pair = network.edge[player1][player2]['policy_pair']
    avg_policy_pair = network.edge[player1][player2]['avg_policy_pair']

    q_table_pair = network.edge[player1][player2]['q-table_pair']

    alpha = network.edge[player1][player2]['alpha']
    epsilon = network.edge[player1][player2]['e-epsilon']
    count = network.edge[player1][player2]['count']

    delta_w = network.edge[player1][player2]['delta_w']
    delta_l = network.edge[player1][player2]['delta_l']

    p1_p = policy_pair[0]
    p2_p = policy_pair[1]

    p1_avg_p = avg_policy_pair[0]
    p2_avg_p = avg_policy_pair[1]

    p1_q = q_table_pair[0]
    p2_q = q_table_pair[1]

    # if player1 < player2:
    #     p1_p = policy_pair[0]
    #     p2_p = policy_pair[1]
    #
    #     p1_q = q_table_pair[0]
    #     p2_q = q_table_pair[1]
    # else:
    #     p1_p = policy_pair[1]
    #     p2_p = policy_pair[0]
    #
    #     p1_q = q_table_pair[1]
    #     p2_q = q_table_pair[0]

    epsilon = epsilon - int(count/1000) * 0.1

    ran = random.uniform(0,1)
    if ran <= epsilon:
        p1_action = random.randint(0, 1)
    else:
        ran = random.uniform(0, 1)
        if ran <= p1_p:
            p1_action = 0
        else:
            p1_action = 1

    ran = random.uniform(0,1)
    if ran <= epsilon:
        p2_action = random.randint(0, 1)
    else:
        ran = random.uniform(0, 1)
        if ran <= p2_p:
            p2_action = 0
        else:
            p2_action = 1

    r = game[p1_action,p2_action]

    # update q-table
    p1_q[0, p1_action] = (1 - alpha) * p1_q[0, p1_action] + alpha * r
    p2_q[0, p2_action] = (1 - alpha) * p2_q[0, p2_action] + alpha * r

    # write back q-table
    network.edge[player1][player2]['q-table_pair'][0] = p1_q
    network.edge[player1][player2]['q-table_pair'][1] = p2_q

    # update avg policy
    count = count + 1

    p1_avg_p = p1_avg_p + 1 / count * (p1_p - p1_avg_p)
    p2_avg_p = p2_avg_p + 1 / count * (p2_p - p2_avg_p)

    # print(p1_action)
    # print(p2_action)
    # print(p1_q)

    # update policy
    p1_expect_value_current = p1_p * p1_q[0,0] + (1 - p1_p) * p1_q[0,1]
    p1_expect_value_avg = p1_avg_p * p1_q[0,0] + (1 - p1_avg_p) * p1_q[0,1]
    if p1_expect_value_current > p1_expect_value_avg:
        # print('p1 wins.')
        delta = delta_w
    else:
        # print('p1 loses.')
        delta = delta_l

    if p1_q[0,0] >= p1_q[0,1]:
        p1_p = min(p1_p + delta, 1.0)
    else:
        p1_p = max(p1_p - delta, 0.0)

    p2_expect_value_current = p2_p * p2_q[0,0] + (1 - p2_p) * p2_q[0,1]
    p2_expect_value_avg = p2_avg_p * p2_q[0,0] + (1 - p2_avg_p) * p2_q[0,1]
    if p2_expect_value_current > p2_expect_value_avg:
        # print('p1 wins.')
        delta = delta_w
    else:
        # print('p1 loses.')
        delta = delta_l

    if p2_q[0,0] >= p2_q[0,1]:
        p2_p = min(p2_p + delta, 1.0)
    else:
        p2_p = max(p2_p - delta, 0.0)

    # write back policy and count
    network.edge[player1][player2]['policy_pair'] = [p1_p, p2_p]
    network.edge[player1][player2]['avg_policy_pair'] = [p1_avg_p, p2_avg_p]
    network.edge[player1][player2]['count'] = count

    return [p1_action,p2_action,r]

def calculateJAPHCResponse(network, player1, player2, action):
    game = network.edge[player1][player2]['game']
    policy_pair = network.edge[player1][player2]['policy_pair']
    avg_policy_pair = network.edge[player1][player2]['avg_policy_pair']

    q_table_pair = network.edge[player1][player2]['q-table_pair']

    alpha = network.edge[player1][player2]['alpha']
    epsilon = network.edge[player1][player2]['e-epsilon']
    count = network.edge[player1][player2]['count']

    delta_w = network.edge[player1][player2]['delta_w']
    delta_l = network.edge[player1][player2]['delta_l']

    # p1, p2's strategies on choosing action 'a'
    p1_p = policy_pair[0]
    p2_p = policy_pair[1]

    # p1, p2's average strategies on choosing action 'a'
    p1_avg_p = avg_policy_pair[0]
    p2_avg_p = avg_policy_pair[1]

    # p1, p2's Q-table
    p1_q = q_table_pair[0]
    p2_q = q_table_pair[1]

    # p1, p2's beliefs on opponent's action distribution
    # evaluate opponent's possibility to choose action a(for the sight of some agent)
    p1_eva_p = action[player1, player2]
    p2_eva_p = action[player2, player1]

    # if player1 < player2:
    #     p1_p = policy_pair[0]
    #     p2_p = policy_pair[1]
    #
    #     p1_q = q_table_pair[0]
    #     p2_q = q_table_pair[1]
    # else:
    #     p1_p = policy_pair[1]
    #     p2_p = policy_pair[0]
    #
    #     p1_q = q_table_pair[1]
    #     p2_q = q_table_pair[0]

    epsilon = epsilon - int(count/1000) * 0.1

    # p1 chooses action
    ran = random.uniform(0,1)
    if ran <= epsilon:
        p1_action = random.randint(0, 1)
    else:
        ran = random.uniform(0, 1)
        if ran <= p1_p:
            p1_action = 0
        else:
            p1_action = 1

    # p2 chooses action
    ran = random.uniform(0,1)
    if ran <= epsilon:
        p2_action = random.randint(0, 1)
    else:
        ran = random.uniform(0, 1)
        if ran <= p2_p:
            p2_action = 0
        else:
            p2_action = 1

    # obtain payoff
    r = game[p1_action,p2_action]

    # update q-table
    p1_update_q_index = p1_action * 2 + p2_action
    p2_update_q_index = p2_action * 2 + p1_action
    p1_q[0, p1_update_q_index] = (1 - alpha) * p1_q[0, p1_update_q_index] + alpha * r
    p2_q[0, p2_update_q_index] = (1 - alpha) * p2_q[0, p2_update_q_index] + alpha * r

    # write back q-table
    network.edge[player1][player2]['q-table_pair'][0] = p1_q
    network.edge[player1][player2]['q-table_pair'][1] = p2_q

    # update avg policy
    count = count + 1

    p1_avg_p = p1_avg_p + 1 / count * (p1_p - p1_avg_p)
    p2_avg_p = p2_avg_p + 1 / count * (p2_p - p2_avg_p)

    # print(p1_action)
    # print(p2_action)
    # print(p1_q)

    # update policy
    p1_q_a = (p1_eva_p * p1_q[0,0] + (1 - p1_eva_p) * p1_q[0,1])
    p1_q_b = (p1_eva_p * p1_q[0,2] + (1 - p1_eva_p) * p1_q[0,3])
    p1_expect_value_current = p1_p * p1_q_a + (1 - p1_p) * p1_q_b
    p1_expect_value_avg = p1_avg_p * p1_q_a + (1 - p1_avg_p) * p1_q_b
    if p1_expect_value_current > p1_expect_value_avg:
        # print('p1 wins.')
        delta = delta_w
    else:
        # print('p1 loses.')
        delta = delta_l

    if p1_q_a >= p1_q_b:
        p1_p = min(p1_p + delta, 1.0)
    else:
        p1_p = max(p1_p - delta, 0.0)


    p2_q_a = (p1_eva_p * p1_q[0, 0] + (1 - p1_eva_p) * p1_q[0, 1])
    p2_q_b = (p1_eva_p * p1_q[0, 2] + (1 - p1_eva_p) * p1_q[0, 3])
    p2_expect_value_current = p2_p * p2_q_a + (1 - p2_p) * p2_q_b
    p2_expect_value_avg = p2_avg_p * p2_q_a + (1 - p2_avg_p) * p2_q_b
    if p2_expect_value_current > p2_expect_value_avg:
        # print('p1 wins.')
        delta = delta_w
    else:
        # print('p1 loses.')
        delta = delta_l

    if p2_q_a >= p2_q_b:
        p2_p = min(p2_p + delta, 1.0)
    else:
        p2_p = max(p2_p - delta, 0.0)

    # write back policy and count
    network.edge[player1][player2]['policy_pair'] = [p1_p, p2_p]
    network.edge[player1][player2]['avg_policy_pair'] = [p1_avg_p, p2_avg_p]
    network.edge[player1][player2]['count'] = count

    return [p1_action,p2_action,r]

def calculateJALResponse(network, player1, player2, action):
    game = network.edge[player1][player2]['game']
    # policy_pair = network.edge[player1][player2]['policy_pair']
    # avg_policy_pair = network.edge[player1][player2]['avg_policy_pair']

    q_table_pair = network.edge[player1][player2]['q-table_pair']

    alpha = network.edge[player1][player2]['alpha']
    epsilon = network.edge[player1][player2]['e-epsilon']
    count = network.edge[player1][player2]['count']

    # p1, p2's Q-table
    p1_q = q_table_pair[0]
    p2_q = q_table_pair[1]

    # p1, p2's beliefs on opponent's action distribution
    # evaluate opponent's possibility to choose action a(for the sight of some agent)
    p1_eva_p = action[player1, player2]
    p2_eva_p = action[player2, player1]

    epsilon = epsilon - int(count/1000) * 0.1


    # EV values
    EV1_a = p1_q[0, 0] * p1_eva_p + p1_q[0, 1] * (1 - p1_eva_p)
    EV1_b = p1_q[0, 2] * p1_eva_p + p1_q[0, 3] * (1 - p1_eva_p)

    EV2_a = p2_q[0, 0] * p2_eva_p + p2_q[0, 1] * (1 - p2_eva_p)
    EV2_b = p2_q[0, 2] * p2_eva_p + p2_q[0, 3] * (1 - p2_eva_p)

    # p1 chooses action
    ran = random.uniform(0,1)
    if ran <= epsilon:
        p1_action = random.randint(0, 1)
    else:
        if EV1_a >= EV1_b:
            p1_action = 0
        else:
            p1_action = 1

    # p2 chooses action
    ran = random.uniform(0,1)
    if ran <= epsilon:
        p2_action = random.randint(0, 1)
    else:
        if EV2_a >= EV2_b:
            p2_action = 0
        else:
            p2_action = 1

    # obtain payoff
    r = game[p1_action,p2_action]

    # update q-table
    p1_update_q_index = p1_action * 2 + p2_action
    p2_update_q_index = p2_action * 2 + p1_action
    p1_q[0, p1_update_q_index] = (1 - alpha) * p1_q[0, p1_update_q_index] + alpha * r
    p2_q[0, p2_update_q_index] = (1 - alpha) * p2_q[0, p2_update_q_index] + alpha * r

    # write back q-table
    network.edge[player1][player2]['q-table_pair'][0] = p1_q
    network.edge[player1][player2]['q-table_pair'][1] = p2_q

    # update avg policy
    count = count + 1

    # write back policy and count
    network.edge[player1][player2]['count'] = count

    return [p1_action,p2_action,r]

# sampling at watching points
def samplePHCFromWatchingPoints(PHC_results, network, iteration, agents_num):
    g_reward = []
    count = 0
    avg_reward = 0

    itera = iteration + 1
    for agent in network.nodes():
        if network.node[agent]['gaming_strategy'] == 1:
            count = count + 1
            g_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])

    # each network at N iteration
    # avg_reward = (sum(g_reward) / agents_num)
    avg_reward = (sum(g_reward) / count)


    if itera == 2000:
        update_index = 0
    elif itera == 5000:
        update_index = 1
    elif itera == 8000:
        update_index = 2
    elif itera == 10000:
        update_index = 3
    elif itera == 20000:
        update_index = 4
    elif itera == 30000:
        update_index = 5
    elif itera == 40000:
        update_index = 6
    elif itera == 50000:
        update_index = 7
    elif itera == 60000:
        update_index = 8
    elif itera == 70000:
        update_index = 9
    elif itera == 80000:
        update_index = 10
    elif itera == 90000:
        update_index = 11
    else:
        update_index = int(itera/100000) + 11


    PHC_results[update_index] = PHC_results[update_index] + avg_reward
    print(itera)
    print(update_index)
    print(PHC_results)

def sampleBRFromWatchingPoints(BR_results, network, iteration, agents_num):
    g_reward = []
    count = 0
    avg_reward = 0

    itera = iteration + 1
    for agent in network.nodes():
        if network.node[agent]['gaming_strategy'] == 0:
            count = count + 1
            g_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])

    # each network at N iteration
    # avg_reward = (sum(g_reward) / agents_num)
    avg_reward = (sum(g_reward) / count)

    if itera == 2000:
        update_index = 0
    elif itera == 5000:
        update_index = 1
    elif itera == 8000:
        update_index = 2
    elif itera == 10000:
        update_index = 3
    elif itera == 20000:
        update_index = 4
    elif itera == 30000:
        update_index = 5
    elif itera == 40000:
        update_index = 6
    elif itera == 50000:
        update_index = 7
    elif itera == 60000:
        update_index = 8
    elif itera == 70000:
        update_index = 9
    elif itera == 80000:
        update_index = 10
    elif itera == 90000:
        update_index = 11
    else:
        update_index = int(itera/100000) + 11


    BR_results[update_index] = BR_results[update_index] + avg_reward
    print(itera)
    print(update_index)
    print(BR_results)

def sampleJALFromWatchingPoints(JAL_results, network, iteration, agents_num):
    g_reward = []
    count = 0
    avg_reward = 0

    itera = iteration + 1
    for agent in network.nodes():
        if network.node[agent]['gaming_strategy'] == 2:
            count = count + 1
            g_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])

    # each network at N iteration
    # avg_reward = (sum(g_reward) / agents_num)
    avg_reward = (sum(g_reward) / count)

    if itera == 2000:
        update_index = 0
    elif itera == 5000:
        update_index = 1
    elif itera == 8000:
        update_index = 2
    elif itera == 10000:
        update_index = 3
    elif itera == 20000:
        update_index = 4
    elif itera == 30000:
        update_index = 5
    elif itera == 40000:
        update_index = 6
    elif itera == 50000:
        update_index = 7
    elif itera == 60000:
        update_index = 8
    elif itera == 70000:
        update_index = 9
    elif itera == 80000:
        update_index = 10
    elif itera == 90000:
        update_index = 11
    else:
        update_index = int(itera/100000) + 11


    JAL_results[update_index] = JAL_results[update_index] + avg_reward
    print(itera)
    print(update_index)
    print(JAL_results)
#--------------------------------------------------------------------------------------

# 20, 50, 100
agents_num = 100
# 5, 10, 20
neighborhood = 12
# 3, 5, 10
bandwidth = 4

iso_damage = -1.0
c = 2000.00
sight = 10000
rebuild_threshold = 0.0

# 0 for random, 1 for highest-expect, 2 for optimal
rewiring_strategy = 2

average_reward = []
highest_reward = []
lowest_reward = []

repeat_time = 10

foo = 0

gaming_strategy = 1

sample_points = [2000, 5000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
BR_results = len(sample_points) * [0]
PHC_results = len(sample_points) * [0]
JAL_results = len(sample_points) * [0]

for gaming_strategy in [2]:
    average_reward = []
    average_reward2 = []
    average_reward3 = []
    highest_reward = []
    highest_reward2 = []
    highest_reward3 = []
    lowest_reward = []
    lowest_reward2 = []
    lowest_reward3 = []
    # print('for gaming strategy: ' + str(gaming_strategy))
    # print('50% BR to 50% WoLF: ')
    for repeat in range(repeat_time):
        print('Repeating: ' + str(repeat))
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

        for iteration in range(1000000):

            if (iteration + 1) in sample_points:
                samplePHCFromWatchingPoints(PHC_results, network, iteration, agents_num)
                sampleBRFromWatchingPoints(BR_results, network, iteration, agents_num)
                sampleJALFromWatchingPoints(JAL_results, network, iteration, agents_num)

            # to avoid the the problem that decision order may influence the fairness
            # shuffle the order at every iteration
            agents = network.nodes()
            random.shuffle(agents)
            # print(network.nodes())
            # print(agents)

            if (iteration) % 10000 == 0:
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
                                    # print("sec...")
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

                gaming_strategy = network.node[i]['gaming_strategy']

                neighbors_num = len(network.neighbors(i))
                if neighbors_num > 0:

                    # JAL
                    if gaming_strategy == 2:
                        # 1) randomly choose a opponent in S(choose the best opponent)
                        oppo_index = random.randint(0, len(network.neighbors(i)) - 1)
                        oppo_agent_no = network.neighbors(i)[oppo_index]

                        # 2) calculate best response

                        # sort the players
                        left = min(i, oppo_agent_no)
                        right = max(i, oppo_agent_no)

                        # do gaming on JAL
                        [my_action, oppo_action, r] = calculateJALResponse(network, left, right, oppActionDis)
                        # print(str(my_action) + ',' + str(oppo_action) + ',' + str(r))

                        # print('reward pair: ' + str(r) + ',' + str(r))
                        # update opponents action model for both 2 agents
                        updateOpponentActionModel100(oppActionDis, i, oppo_agent_no, oppo_action, iteration)
                        updateOpponentActionModel100(oppActionDis, oppo_agent_no, i, my_action, iteration)

                        network.node[i]['ar'] = network.node[i]['ar'] + r
                        network.node[oppo_agent_no]['ar'] = network.node[oppo_agent_no]['ar'] + r

                    # JA-WoLF
                    if gaming_strategy == 1:
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

                        # sort the players
                        left = min(i, oppo_agent_no)
                        right = max(i, oppo_agent_no)

                        # do gaming on PHC-WoLF
                        [my_action, oppo_action, r] = calculateJAPHCResponse(network, left, right, oppActionDis)
                        # print(str(my_action) + ',' + str(oppo_action) + ',' + str(r))

                        # print('reward pair: ' + str(r) + ',' + str(r))
                        # update opponents action model for both 2 agents
                        # updateOpponentActionModel(oppActionDis, i, oppo_agent_no, oppo_action, iteration)
                        # updateOpponentActionModel(oppActionDis, oppo_agent_no, i, my_action, iteration)
                        updateOpponentActionModel100(oppActionDis, i, oppo_agent_no, oppo_action, iteration)
                        updateOpponentActionModel100(oppActionDis, oppo_agent_no, i, my_action, iteration)

                        network.node[i]['ar'] = network.node[i]['ar'] + r
                        network.node[oppo_agent_no]['ar'] = network.node[oppo_agent_no]['ar'] + r

                    # FP
                    if gaming_strategy == 0:
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
            if (iteration + 1) % 10000 == 0:
                # print('Rebuild phase.')
                for i in network.nodes():
                    total_reward = network.node[i]['ar'] - network.node[i]['ac']
                    # if the agent is isolated and has bad reward, we rebuild it
                    if total_reward < rebuild_threshold and len(network.neighbors(i)) == 0:
                        print('Rebuild agent: ' + str(i))

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
        group_reward2 = []
        group_reward3 = []

        br_num = 0
        wolf_num = 0
        jal_num = 0

        for agent in network.nodes():
            # print('Agent: ' + str(agent))
            # print('  Neighbors: ' + str(network.neighbors(agent)))
            # print('  AR: ' + str(network.node[agent]['ar']))
            # print('  AC: ' + str(network.node[agent]['ac']))
            # print('  Reward: ' + str(network.node[agent]['ar'] - network.node[agent]['ac']))
            # print('----------------------------------')
            if network.node[agent]['gaming_strategy'] == 0:
                br_num = br_num + 1
                group_reward.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            if network.node[agent]['gaming_strategy'] == 1:
                wolf_num = wolf_num + 1
                group_reward2.append(network.node[agent]['ar'] - network.node[agent]['ac'])
            if network.node[agent]['gaming_strategy'] == 2:
                jal_num = jal_num + 1
                group_reward3.append(network.node[agent]['ar'] - network.node[agent]['ac'])

        # print('Group average reward: ' + str(sum(group_reward) / agents_num))
        # print('Group highest reward: ' + str(max(group_reward)))
        # print('Group lowest reward: ' + str(min(group_reward)))

        degree = nx.degree_histogram(network)
        # print('Group degree distribution: ' + str(degree))


        # each network at N iteration
        average_reward.append(sum(group_reward) / br_num)
        highest_reward.append(max(group_reward))
        lowest_reward.append(min(group_reward))

        average_reward2.append(sum(group_reward2) / wolf_num)
        highest_reward2.append(max(group_reward2))
        lowest_reward2.append(min(group_reward2))

        average_reward3.append(sum(group_reward3) / jal_num)
        highest_reward3.append(max(group_reward3))
        lowest_reward3.append(min(group_reward3))


    # print('--------------------------------------------------------------------')
    # print('--------------------------------------------------------------------')
    # print('Final outputs:')

    # L networks at N iteration

    print('Mean average reward for BR: ' + str(sum(average_reward) / repeat_time))

    print('Mean average reward for WoLF: ' + str(sum(average_reward2) / repeat_time))

    print('Mean average reward for JAL: ' + str(sum(average_reward3) / repeat_time))
    # print('Mean highest reward: ' + str(sum(highest_reward) / repeat_time))
    # print('Mean lowest reward: ' + str(sum(lowest_reward) / repeat_time))


print('BR:')
print(BR_results)
print('PHC:')
print(PHC_results)
print('JAL:')
print(JAL_results)


for m in range(len(BR_results)):
    BR_results[m] = BR_results[m] / repeat_time
    BR_results[m] = BR_results[m] / sample_points[m]

for n in range(len(PHC_results)):
    PHC_results[n] = PHC_results[n] / repeat_time
    PHC_results[n] = PHC_results[n] / sample_points[n]

for m in range(len(JAL_results)):
    JAL_results[m] = JAL_results[m] / repeat_time
    JAL_results[m] = JAL_results[m] / sample_points[m]

print('BR:')
print(BR_results)
print('PHC:')
print(PHC_results)
print('JAL:')
print(JAL_results)

# plt.xlim(0.0, 1.0)
# plt.ylim(-0.1, 1.0)
x_axi = sample_points
plt.plot(x_axi,BR_results)
plt.plot(x_axi,PHC_results)
plt.plot(x_axi,JAL_results)
plt.title('(100,4,12) - FP - WoLF - JAL - 1000000')

plt.show()


# -----------------------------------------------------------------------------
# the experiment which compares the long-term reward between BR and PHC-WoLF

# BR:
# [125695.27468586387, 704663.49454501958, 1432300.8694134171, 2888305.5980568575, 4344158.6708032629, 5802066.9541265909, 7262747.7377777081, 8723412.9594228193, 10184093.850649239, 11644755.105199834, 13105422.194145752, 14566126.234660007]

# PHC:
# [57186.964953298258, 623754.45689820382, 1394172.8594316756, 2943065.6611432554, 4495902.1781113846, 6050141.0582304616, 7606152.2360368185, 9162663.0436742175, 10719245.648163948, 12275816.972023055, 13832380.510903519, 15388977.618702073]


# 2017/07/27 half-to-half
#
# BR:
# [72513.61366618851, 625615.28780890617, 1386212.3451960175, 2916589.5679184878, 4450791.2388797784, 5984657.6065104771, 7519728.938191819, 9055467.4759466387, 10591327.389159411, 12126922.490620159, 13662993.297226533, 15199095.485414764]
# PHC:
# [106812.61604921208, 662705.87743808108, 1408187.68456277, 2904214.4497633725, 4402701.4604794979, 5899933.1547690267, 7399156.6298749177, 8901422.5041313116, 10403945.955062957, 11906495.601428013, 13409464.263867395, 14912585.226659272]
# BR:
# [0.72513613666188514, 1.2512305756178124, 1.3862123451960173, 1.4582947839592437, 1.4835970796265929, 1.4961644016276192, 1.503945787638364, 1.5092445793244398, 1.5130467698799159, 1.5158653113275198, 1.5181103663585038, 1.5199095485414764]
# PHC:
# [1.0681261604921208, 1.3254117548761621, 1.4081876845627701, 1.4521072248816862, 1.467567153493166, 1.4749832886922567, 1.4798313259749836, 1.4835704173552187, 1.4862779935804227, 1.4883119501785018, 1.4899404737630439, 1.4912585226659272]

# 2017/08/23

# BR:
# [0.83866900576781356, 0.94080724381613601, 0.96902741482386956, 0.98063651573682165, 1.015678982111661, 1.0636462423588047, 1.1107551945042191, 1.1517209715516306, 1.1826793728460414, 1.2063573639619782, 1.224644929046824, 1.2387635852671723, 1.250489812555577, 1.3055959192380928, 1.32567398224286, 1.3358285478146479, 1.3418998420341965, 1.3459967068483161, 1.3488742174797295, 1.3511063349967769, 1.3529491434091405, 1.3545734000339897]
# PHC:
# [0.4853977220838574, 0.59225140722039582, 0.63165059145552971, 0.65545871622382168, 0.80023815101744233, 0.94143103961149832, 1.0287983812513544, 1.0903533664997995, 1.1342383683841353, 1.1665888914666092, 1.1912392588263727, 1.2109169438208169, 1.2265183082595479, 1.3005687902824266, 1.3266247120460732, 1.3397916431127777, 1.3477677401362644, 1.3532497455576484, 1.3573301643007096, 1.3604112992213508, 1.3628142413659536, 1.364723592353978]
# JAL:
# [0.46531597046518752, 0.57597558822670625, 0.62011986368006256, 0.64536981117391823, 0.78914549900971642, 0.92775751848882038, 1.0126476258624639, 1.0709857441143291, 1.1127736968780251, 1.1438628562625135, 1.1680879294252686, 1.187037565818194, 1.2024033817860247, 1.2721356143630036, 1.2958717337186496, 1.3077275297146946, 1.3149691724678207, 1.3198720252385545, 1.3233913840233942, 1.3261146706345517, 1.328243569040048, 1.3299473566971705]


