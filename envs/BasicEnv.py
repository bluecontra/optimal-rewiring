import networkx as nx
import random
import numpy as np
import utils as ut


class BasicEnv(object):
    def __init__(self, agent_num, network_type,
                 init_conn_num,
                 neighborhood_size,
                 rewiring_cost,
                 rewiring_probability,
                 distribution_type):
        self.foo = 0

        self.agent_num = agent_num
        self.network_type = network_type
        self.init_conn_num = init_conn_num
        self.neighborhood_size = neighborhood_size
        self.pot_peer_num = self.neighborhood_size - self.init_conn_num
        self.rewiring_cost = rewiring_cost
        self.phi = rewiring_probability
        self.distribution_type = distribution_type
        self.rewiring_sight = 1 / self.phi

        self.reward = None

        # initial the network and unknown reward distribution
        self.network = self._initialNetworkTopology()
        self.rewardDisMatrix = self._initialDistributingSetting()
        # initial the know game in S
        self._initialTheKnownGameInS()
        # initial the Policy for every agent pair
        self.oppActionDis = self._initialOpponentsActionDistribution()

    def _rewire(self, i):
        rewiring_strategy = self.network.node[i]['rewiring_strategy']
        # randomly rewiring
        if rewiring_strategy == 0:
            self._randomRewire(i)
        elif rewiring_strategy == 1:
            self._HERewire(i)
        else:
            self._OptimalRewire(i)

        return

    def _randomRewire(self, i):
        # random rewiring target
        rewiring_target = np.random.choice(np.array(self.network.node[i]['S_'] + self.network.node[i]['BL']))
        # rewiring_agent_index = random.randint(0, len(self.network.node[i]['S_']) - 1)
        # rewiring_agent_no = self.network.node[i]['S_'][rewiring_agent_index]

        # check if the rewiring target accept the proposal
        if self._getRewiringResponse(rewiring_target, i) == 1:
            # do rewiring
            # unlink the worst neighbor if reach the limitation
            current_neighbor_num = len(self.network.neighbors(i))
            if current_neighbor_num == self.network.node[i]['max_conn_num']:
                self._unlinkWorstNeighbor(i)

            # rewire target
            # update information
            self.network.node[i]['S_'].remove(rewiring_target) if rewiring_target in self.network.node[i]['S_'] \
                else self.network.node[i]['BL'].remove(rewiring_target)
            self.network.add_edge(i, rewiring_target)

            # reveal the reward matrix of the unknown agent and update
            self._revealTheUnknownGameForEdge(i, rewiring_target)
            # TO-DO
            # the rewiring pair shares the cost
            self.network.node[i]['ac'] += self.rewiring_cost / 2
            self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
            self.network.node[i]['rewiring_time'] += 1
        else:
            # do nothing
            print('Target agent ', rewiring_target, ' refused the rewiring.')


    def _HERewire(self, i):
        # pick HE target
        # calculate the expected value of each neighbor in S' and get the maximum expected value ev
        self.network.node[i]['expected_value_list_S_'] = self.calculateHEValueInS_(i)
        # TO-DO
        # calculate BL

        rewiring_target = np.random.choice(np.array(self.network.node[i]['S_'] + self.network.node[i]['BL']))
        # check if the rewiring target accept the proposal
        if self._getRewiringResponse(rewiring_target, i) == 1:
            # do rewiring
            # unlink the worst neighbor if reach the limitation
            current_neighbor_num = len(self.network.neighbors(i))
            if current_neighbor_num == self.network.node[i]['max_conn_num']:
                self._unlinkWorstNeighbor(i)

            # rewire target
            # update information
            self.network.node[i]['S_'].remove(rewiring_target) if rewiring_target in self.network.node[i]['S_'] \
                else self.network.node[i]['BL'].remove(rewiring_target)
            self.network.add_edge(i, rewiring_target)

            # reveal the reward matrix of the unknown agent and update
            self._revealTheUnknownGameForEdge(i, rewiring_target)
            # TO-DO
            # the rewiring pair shares the cost
            self.network.node[i]['ac'] += self.rewiring_cost / 2
            self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
            self.network.node[i]['rewiring_time'] += 1
        else:
            # do nothing
            print('Target agent ', rewiring_target, ' refused the rewiring.')
        return
    def _OptimalRewire(self, i):
        return

    # get rewiring response from target for i
    def _getRewiringResponse(self, target, i):
        rewiring_strategy = self.network.node[target]['rewiring_strategy']

        # accept at random
        if rewiring_strategy == 0:
            return 1 if random.uniform(0,1) < 0.5 else 0
        # accept with HE
        p = self.oppActionDis[target, i]
        game = self.network.edge[target][i]['game']
        if rewiring_strategy == 1:
            # if known agent
            if i in self.network.node[target]['BL']:
                expected_value = max(p * game[0,0] + (1 - p) * game[0,1], p * game[1,0] + (1 - p) * game[1,1])
                return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            if i in self.network.node[target]['S_']:
                HE_value = self.calculateHEValue(target, i)
                return 1 if HE_value * self.rewiring_sight > self.rewiring_cost / 2 else 0

        if rewiring_strategy == 2:
            # if known agent
            if i in self.network.node[target]['BL']:
                expected_value = max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1])
                return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            # TO-DO
            # check target connection numbers
            if i in self.network.node[target]['S_']:
                lamb_value = self.calculateLambdaValue(target, i)
                return 1 if lamb_value * self.rewiring_sight > self.rewiring_cost / 2 else 0



    # unlink the worst neighbor before rewiring
    def _unlinkWorstNeighbor(self, i):
        rewiring_strategy = self.network.node[i]['rewiring_strategy']

        # unlink a random neighbor
        if rewiring_strategy == 0:
            unlink_agent_index = random.randint(0, len(self.network.neighbors(i)) - 1)
            unlink_agent_no = self.network.neighbors(i)[unlink_agent_index]
            self.network.node[i]['BL'].append(unlink_agent_no)
            self.network.remove_edge(i, unlink_agent_no)
        # unlink the worst neighbor
        else:
            # 1) get the maximum(minimum) expected value in S
            self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
            worst_agent_index = np.array(self.network.node[i]['expected_value_list']).argmin()
            worst_agent_no = self.network.neighbors(i)[worst_agent_index]
            self.network.node[i]['BL'].append(worst_agent_index)
            self.network.remove_edge(i, worst_agent_index)
        return

    # i interacts with j, play the game G(i,j)
    def _interact(self, i, j):
        game = self.network.edge[i][j]['game']
        p_j = self.oppActionDis[i, j]
        p_i = self.oppActionDis[j, i]

        s_i = self.network.node[i]['gaming_strategy']
        s_j = self.network.node[j]['gaming_strategy']

        policy_pair = self.network.edge[i][j]['policy_pair']
        avg_policy_pair = self.network.edge[i][j]['avg_policy_pair']

        q_table_pair = self.network.edge[i][j]['q-table_pair']
        alpha = self.network.edge[i][j]['alpha']
        epsilon = self.network.edge[i][j]['e-epsilon']
        count = self.network.edge[i][j]['count']
        delta_w = self.network.edge[i][j]['delta_w']
        delta_l = self.network.edge[i][j]['delta_l']

        epsilon = epsilon - int(count / 1000) * 0.1

        # p1, p2's Q-table
        q_i = q_table_pair[0]
        q_j = q_table_pair[1]
        # p1, p2's strategies on choosing action 'a'
        pol_i = policy_pair[0]
        pol_j = policy_pair[1]
        # p1, p2's average strategies on choosing action 'a'
        avg_pol_i = avg_policy_pair[0]
        avg_pol_j = avg_policy_pair[1]

        # calculate actions
        if s_i == 0:
            i_action = ut.calculateBestResponse(game, p_j)
        elif s_i == 1:
            i_action = ut.calculateJAWoLFResponse(pol_i, epsilon)
        else:
            i_action = ut.calculateJALResponse(q_i, p_j, epsilon)

        if s_j == 0:
            j_action = ut.calculateBestResponse(game, p_i)
        elif s_j == 1:
            j_action = ut.calculateJAWoLFResponse(pol_j, epsilon)
        else:
            j_action = ut.calculateJALResponse(q_j, p_i, epsilon)

        # update accumulated reward
        r = game[i_action, j_action]
        self._updateReward(i, r)
        self._updateReward(j, r)

        # update opponent's policy
        self._updateOppoPolicy(i, j, j_action)
        self._updateOppoPolicy(j, i, i_action)

        count += 1

        # update Q
        if s_i > 0:
            q_i = self._updateQ(q_i, i_action, j_action, r, alpha)
        # update policy for WoLF
        if s_i == 1:
            # avg policy
            avg_pol_i = self._updateAvgPolicy(avg_pol_i, pol_i, count)
            # update policy
            pol_i = self._updatePolicy(q_i, p_j, pol_i, avg_pol_i, delta_w, delta_l)

        if s_j > 0:
            q_j = self._updateQ(q_j, j_action, i_action, r, alpha)
        # update policy for WoLF
        if s_j == 1:
            # avg policy
            avg_pol_j = self._updateAvgPolicy(avg_pol_j, pol_j, count)
            # update policy
            pol_j = self._updatePolicy(q_j, p_i, pol_j, avg_pol_j, delta_w, delta_l)

        # write back q-table
        self.network.edge[i][j]['q-table_pair'] = [q_i, q_j]
        # write back policy and count
        self.network.edge[i][j]['policy_pair'] = [pol_i, pol_j]
        self.network.edge[i][j]['avg_policy_pair'] = [avg_pol_i, avg_pol_j]
        self.network.edge[i][j]['count'] = count

    def calculateHEValueInS_(self, i):
        expected_value_S_ = []
        S_ = self.network.node[i]['S_']
        # print(S_)
        for j in S_:
            expected_value_S_.append(self.calculateHEValue(i, j))
        return expected_value_S_

    def calculateHEValue(self, i, j):
        # print('calculate index for agent: ' + str(neighbor))
        p = self.oppActionDis[i, j]
        # sort the agent_no and neighbor, small one at left
        left = min(i, j)
        right = max(i, j)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * self.agent_num + right
        if self.distribution_type == 0:
            # U(a1,b1) U(a2,b2)
            a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0]
            b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1]
            ev1 = (a1 + b1) / 2 - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]
            a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2]
            b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3]
            ev2 = (a2 + b2) / 2 - p * self.rewardDisMatrix[index_in_dis_matrix, 5]
            return max(ev1, ev2)

        if self.distribution_type == 1:
            # beta(a,b) E(x) = a /(a + b)
            ev1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] \
                  / (self.rewardDisMatrix[index_in_dis_matrix, 0] + self.rewardDisMatrix[index_in_dis_matrix, 1]) \
                  - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]

            ev2 = p * self.rewardDisMatrix[index_in_dis_matrix, 2] \
                  / (self.rewardDisMatrix[index_in_dis_matrix, 2] + self.rewardDisMatrix[index_in_dis_matrix, 3]) \
                  - p * self.rewardDisMatrix[index_in_dis_matrix, 5]

            return max(ev1, ev2)

    def calculateLambdaValue(self, i, j):
        p = self.oppActionDis[i, j]
        # sort the agent_no and neighbor, small one at left
        left = min(i, j)
        right = max(i, j)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * self.agent_num + right

        # 1) get the maximum(minimum) expected value in S
        self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
        expected_value_list = self.network.node[i]['expected_value_list']
        minimum_expected_reward = min(expected_value_list)
        sec_minimum_expected_reward = 0
        if len(self.network.neighbors(i)) > 1:
            expected_value_list.remove(minimum_expected_reward)
            sec_minimum_expected_reward = min(expected_value_list)

        if len(self.network.neighbors(i)) == self.network.node[i]['max_conn_num']:
            if self.distribution_type == 0:
                # U(a1,b1) U(a2,b2)
                a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]
                b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1] - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]

                a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2] - p * self.rewardDisMatrix[index_in_dis_matrix, 5]
                b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3] - p * self.rewardDisMatrix[index_in_dis_matrix, 5]



                if len(self.network.neighbors(i)) == 1:
                    z1 = (a1 + b1) / 2 - minimum_expected_reward
                    z2 = (a2 + b2) / 2 - minimum_expected_reward
                else:
                    # z1 = calculateIndex(a1, b1, c, sight)
                    z1 = ut.calculateLambdaIndex(a1, b1, self.rewiring_sight,
                                                 minimum_expected_reward,
                                                 sec_minimum_expected_reward)
                    # z2 = calculateIndex(a2, b2, c, sight)
                    z2 = ut.calculateLambdaIndex(a2, b2, self.rewiring_sight,
                                                 minimum_expected_reward,
                                                 sec_minimum_expected_reward)
                return max(z1, z2)

            if self.distribution_type == 1:
                # beta(a,b)
                a1 = self.rewardDisMatrix[index_in_dis_matrix, 0]
                b1 = self.rewardDisMatrix[index_in_dis_matrix, 1]

                a2 = self.rewardDisMatrix[index_in_dis_matrix, 2]
                b2 = self.rewardDisMatrix[index_in_dis_matrix, 3]

                if len(self.network.neighbors(i)) == 1:
                    z1 = p * a1 / (a1 + b1) - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4] \
                         - minimum_expected_reward
                    z2 = (1 - p) * a2 / (a2 + b2) - p * self.rewardDisMatrix[index_in_dis_matrix, 5] \
                         - minimum_expected_reward
                else:
                    # z1 = calculateIndex(a1, b1, c, sight)
                    z1 = ut.calculateLambdaIndexInBeta(a1, b1, self.rewiring_sight,
                                                    minimum_expected_reward,
                                                    sec_minimum_expected_reward,
                                                    p)
                    # z2 = calculateIndex(a2, b2, c, sight)
                    z2 = ut.calculateLambdaIndexInBeta(a2, b2, self.rewiring_sight,
                                                    minimum_expected_reward,
                                                    sec_minimum_expected_reward,
                                                    1 - p)
                return max(z1, z2)
        else:
            # TO-DO
            # if target has not reached the connection limitation
            # Lambda should be calculated with different equations
            return

    # get the expected reward list for set S
    def calculateExpectedRewardInS(self, i):
        expectedReward_S = []
        S = self.network.neighbors(i)
        # print(S)
        for oppo_no in S:
            # evaluate opponent's possibility to choose action a(for the sight of some agent)
            p = self.oppActionDis[i, oppo_no]
            game = self.network.edge[i][oppo_no]['game']
            expectedReward_S.append(max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1]))

        return expectedReward_S

    def _pickAction(self, i, game, p):
    #     if self.network.node[i]['gaming_strategy'] == 0:
    #         a = ut.calculateBestResponse(game, p)
    #     # elif self.network.node[i]['gaming_strategy'] == 1:
    #     #
    #     elif self.network.node[i]['gaming_strategy'] == 2:
    #         a = ut.calculateJALResponse()
    #
    #
        return

    def _updateReward(self, i, r):
        self.network.node[i]['ar'] += r

    def _updateQ(self, q, i_action, j_action, r, alpha):
        index = i_action * 2 + j_action
        q[0, index] = (1 - alpha) * q[0, index] + alpha * r
        return q

    def _updateAvgPolicy(self, avg_pol, pol, count):
        return avg_pol + 1 / count * (pol - avg_pol)

    def _updatePolicy(self, q, op_pol, pol, avg_pol, delta_w, delta_l):
        q_a = (op_pol * q[0, 0] + (1 - op_pol) * q[0, 1])
        q_b = (op_pol * q[0, 2] + (1 - op_pol) * q[0, 3])
        expect_value_current = pol * q_a + (1 - pol) * q_b
        expect_value_avg = avg_pol * q_a + (1 - avg_pol) * q_b
        delta = delta_w if expect_value_current > expect_value_avg else delta_l

        return min(pol + delta, 1.0) if q_a >= q_b else max(pol - delta, 0.0)

    def _updateOppoPolicy(self, i, j, j_action):
        p_old = self.oppActionDis[i, j]
        self.oppActionDis[i, j] = ut.updatePolcyFromFrequecy(p_old, j_action)

    # initialize the network topology;
    # return a network(graph)
    def _initialNetworkTopology(self):
        N = None
        if self.network_type == 0:
            N = nx.random_graphs.random_regular_graph(self.init_conn_num, self.agent_num)
        elif self.network_type == 1:
            N = nx.random_graphs.watts_strogatz_graph(self.agent_num, self.init_conn_num, 0.3)
        elif self.network_type == 2:
            N = nx.random_graphs.barabasi_albert_graph(self.agent_num, 1)

        initS__Size = self.pot_peer_num
        for i in range(self.agent_num):
            S_ = []
            BL = []
            # links are initial links, the set S
            S = N.neighbors(i)
            # randomly add potential peers, the set S'
            for j in range(initS__Size):
                pot_peer_index = random.randint(0, self.agent_num - 1)
                while pot_peer_index == i or pot_peer_index in S:
                    pot_peer_index = random.randint(0, self.agent_num - 1)
                S_.append(pot_peer_index)
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
            N.node[i]['rewiring_strategy'] = 0
            # ran = random.uniform(0, 1)
            # if ran < 0.33:
            #     N.node[i]['rewiring_strategy'] = 0
            # elif ran < 0.66:
            #     N.node[i]['rewiring_strategy'] = 1
            # else:
            #     N.node[i]['rewiring_strategy'] = 2

            # 0 for BR, 1 for WoLF, 2 for JAL
            N.node[i]['gaming_strategy'] = 2
            # init game_strategy
            # ran = random.uniform(0, 1)
            # if ran <= 1 / 3:
            #     N.node[i]['gaming_strategy'] = 1
            # elif ran <= 2 / 3:
            #     N.node[i]['gaming_strategy'] = 2

        return N

    # initialize the reward distribution for each agent pair( k*(k-1)/2 ).
    # return a numpy matrix
    def _initialDistributingSetting(self):
        distribution = []
        # 0 for uniform distribution, 1 for beta distribution, 2 for mixture
        # for each agent pair, generate a U(a1,b1) and U(a2,b2) for respective reward distribution on different actions.

        size = self.agent_num * self.agent_num

        for i in range(int(size)):
                temp = ut.initSingleGame(self.distribution_type)
                distribution.append(temp)

        distribution = np.matrix(distribution)

        return distribution

    # initial the game in S, binding on the edge between agent(node) pair
    def _initialTheKnownGameInS(self):
        edges = self.network.edges()
        for (i, j) in edges:
            # as i always < j, so directly
            self._revealTheUnknownGameForEdge(i, j)


    # reveal the unknown game by giving agent pair (i,j) (edge)
    # add a new edge in network and set the attribute 'game' to the edge
    def _revealTheUnknownGameForEdge(self, i, j):
        # sort the agent_no and neighbor, small one at left
        left = min(i, j)
        right = max(i, j)
        index = left * self.agent_num + right

        a, b = ut.sampleTrueRewardFromDistribution(self.distribution_type,
                                                   self.rewardDisMatrix[index, 0],
                                                   self.rewardDisMatrix[index, 1],
                                                   self.rewardDisMatrix[index, 2],
                                                   self.rewardDisMatrix[index, 3])
        al = self.rewardDisMatrix[index, 4]
        al2 = self.rewardDisMatrix[index, 5]

        gameMatrix = np.array([[a, al], [al2, b]])
        gameMatrix = np.matrix(gameMatrix)
        self.network.edge[i][j]['game'] = gameMatrix
        self.network.edge[i][j]['alpha'] = [al, al2]

        self.network.edge[i][j]['policy_pair'] = [0.5, 0.5]
        self.network.edge[i][j]['avg_policy_pair'] = [0.5, 0.5]

        # q_table_pair = np.array([[0.0,0.0], [0.0,0.0]])
        q_table_pair = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        q_table_pair = np.matrix(q_table_pair)
        self.network.edge[i][j]['q-table_pair'] = q_table_pair
        self.network.edge[i][j]['e-epsilon'] = 1.0
        self.network.edge[i][j]['alpha'] = 0.1
        self.network.edge[i][j]['count'] = 0

        self.network.edge[i][j]['delta_w'] = 0.0001
        self.network.edge[i][j]['delta_l'] = 0.0002
        # TO-DO params

    # initialize the opponents action distribution for every agent pair
    # return a numpy matrix
    def _initialOpponentsActionDistribution(self):
        oppActionDis = []
        for i in range(self.agent_num):
            subDis = np.random.random(size=self.agent_num)
            oppActionDis.append(subDis)
        return np.matrix(oppActionDis)

    def printAgentInfo(self):
        group_reward = []
        group_rewiring = 0.0

        print('')
        print('Agents info below.')
        print('----------------------------------')

        for agent in self.network.nodes():
            print('Agent: ' + str(agent))
            print('  Neighbors: ' + str(self.network.neighbors(agent)))
            print('  Rewiring time: ' + str(self.network.node[agent]['rewiring_time']))
            print('  AR: ' + str(self.network.node[agent]['ar']))
            print('  AC: ' + str(self.network.node[agent]['ac']))
            print('  Reward: ' + str(self.network.node[agent]['ar'] - self.network.node[agent]['ac']))
            print('----------------------------------')
            group_reward.append(self.network.node[agent]['ar'] - self.network.node[agent]['ac'])
            group_rewiring = group_rewiring + self.network.node[agent]['rewiring_time']

        print('Group average rewiring time: ' + str(group_rewiring / self.agent_num))
        print('Group average reward: ' + str(sum(group_reward) / self.agent_num))
        print('Group highest reward: ' + str(max(group_reward)))
        print('Group lowest reward: ' + str(min(group_reward)))

        return group_reward, group_rewiring
