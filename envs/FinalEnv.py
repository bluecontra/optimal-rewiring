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
                 distribution_type,
                 rewiring_strategy,
                 learning_strategy,
                 K):
        self.foo = 0

        self.agent_num = agent_num
        self.network_type = network_type
        self.init_conn_num = init_conn_num
        self.neighborhood_size = neighborhood_size
        self.pot_peer_num = self.neighborhood_size - self.init_conn_num
        self.rewiring_cost = rewiring_cost
        self.phi = rewiring_probability
        self.distribution_type = distribution_type
        self.rewiring_sight = K / self.phi
        self.rewiring_strategy = rewiring_strategy
        self.learning_strategy = learning_strategy
        self.reward = None

        # initial the network and unknown reward distribution
        self.network = self._initialNetworkTopology()
        self._initPotentialPeers()

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
        elif rewiring_strategy == 2 or rewiring_strategy == 3 or rewiring_strategy == 4:
            self._OptimalRewire(i)
        else:
            return
        # else:
            # self._OptimalRewireMax(i)

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

            if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
                self._unlinkWorstNeighbor(i)

            if len(self.getNeighbors(rewiring_target)) == self.network.node[rewiring_target]['max_conn_num']:
                self._unlinkWorstNeighbor(rewiring_target)

            # rewire target
            # update information
            # print('Rewire agent', rewiring_target)
            if rewiring_target in self.network.node[i]['S_']:
                self.network.node[i]['S_'].remove(rewiring_target)
                # TO-DO
                if i in self.network.node[rewiring_target]['S_']:
                    self.network.node[rewiring_target]['S_'].remove(i)

                self.network.add_edge(i, rewiring_target)
                # reveal the reward matrix of the unknown agent and update
                self._revealTheUnknownGameForEdge(i, rewiring_target)
            else:
                self.network.node[i]['BL'].remove(rewiring_target)
                self.network.node[rewiring_target]['BL'].remove(i)
            # enable the connection
            self.network.edge[i][rewiring_target]['is_connected'] = 0

            # TO-DO
            # the rewiring pair shares the cost
            self.network.node[i]['ac'] += self.rewiring_cost
            # self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
            # self.network.node[rewiring_target]['ar'] += self.rewiring_cost
            self.network.node[i]['rewiring_time'] += 1
        else:
            # do nothing
            # print('Target agent ', rewiring_target, ' refused the rewiring.')
            return


    def _HERewire(self, i):
        # pick HE target
        # calculate the expected value of each neighbor in S' and get the maximum expected value ev
        self.network.node[i]['expected_value_list_S_'] = self.calculateHEValueInS_(i)
        # calculate BL
        self.network.node[i]['expected_value_list_BL'] = self.calculateExpectedRewardInBL(i)

        maximum_expected_reward = self.calculateUpperBound(i)
        minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(i)
        # ev_m = max(self.network.node[i]['expected_value_list_S_']) > max(self.network.node[i]['expected_value_list_BL'])
        max_v_S_ = 0 if self.network.node[i]['expected_value_list_S_'] == [] \
            else max(self.network.node[i]['expected_value_list_S_'])
        max_v_BL = 0 if self.network.node[i]['expected_value_list_BL'] == [] \
            else max(self.network.node[i]['expected_value_list_BL'])
        if max_v_S_ > max_v_BL:
            ev_m = max_v_S_
            ev_max_index = np.array(self.network.node[i]['expected_value_list_S_']).argmax()
            rewiring_target = self.network.node[i]['S_'][ev_max_index]
        else:
            ev_m = max_v_BL
            ev_max_index = np.array(self.network.node[i]['expected_value_list_BL']).argmax()
            rewiring_target = self.network.node[i]['BL'][ev_max_index]

        cohere = self.network.node[i]['max_conn_num'] - len(self.getNeighbors(i)) + 1
        # check if the rewiring target accept the proposal
        if ev_m * self.rewiring_sight > self.rewiring_cost and self._getRewiringResponse(rewiring_target, i) == 1:
            # do rewiring
            # unlink the worst neighbor if reach the limitation

            if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
                self._unlinkWorstNeighbor(i)
            if len(self.getNeighbors(rewiring_target)) == self.network.node[rewiring_target]['max_conn_num']:
                self._unlinkWorstNeighbor(rewiring_target)

            # rewire target
            # update information
            # print('Rewire agent', rewiring_target)
            if rewiring_target in self.network.node[i]['S_']:
                self.network.node[i]['S_'].remove(rewiring_target)
                # TO-DO
                if i in self.network.node[rewiring_target]['S_']:
                    self.network.node[rewiring_target]['S_'].remove(i)

                self.network.add_edge(i, rewiring_target)
                # reveal the reward matrix of the unknown agent and update
                self._revealTheUnknownGameForEdge(i, rewiring_target)
            else:
                self.network.node[i]['BL'].remove(rewiring_target)
                self.network.node[rewiring_target]['BL'].remove(i)
            # enable the connection
            self.network.edge[i][rewiring_target]['is_connected'] = 0

            # TO-DO
            # the rewiring pair shares the cost
            self.network.node[i]['ac'] += self.rewiring_cost
            # self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
            # self.network.node[rewiring_target]['ar'] += self.rewiring_cost
            self.network.node[i]['rewiring_time'] += 1
        else:
            # do nothing
            # print('Target agent ', rewiring_target, ' refused the rewiring.')
            return
        return

    def _OptimalRewire(self, i):
        # pick Optimal target
        rewiring_strategy = self.network.node[i]['rewiring_strategy']
        # calculate BL
        self.network.node[i]['expected_value_list_BL'] = self.calculateExpectedRewardInBL(i)
        # calculate the lambda value of each neighbor in S' and get the maximum expected value ev
        self.network.node[i]['index_z_value_list'] = self.calculateLambdaInS_(i)

        maximum_expected_reward = self.calculateUpperBound(i)
        minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(i)
        mean_expected_value = self.calculateMean(i)
        if rewiring_strategy == 2:
            compare_value = minimum_expected_reward
        elif rewiring_strategy == 3:
            compare_value = maximum_expected_reward
        else:
            compare_value = mean_expected_value

        max_v_S_ = -1 if self.network.node[i]['index_z_value_list'] == [] \
            else max(self.network.node[i]['index_z_value_list'])
        max_v_BL = -1 if self.network.node[i]['expected_value_list_BL'] == [] \
            else max(self.network.node[i]['expected_value_list_BL'])
        # print(max_v_S_, ' ', max_v_BL)
        if max_v_S_ >= max_v_BL - compare_value:
        # if max_v_S_ >= max_v_BL:
            ev_m = max_v_S_
            ev_max_index = np.array(self.network.node[i]['index_z_value_list']).argmax()
            rewiring_target = self.network.node[i]['S_'][ev_max_index]
        else:
            # ev_m = max_v_BL
            ev_m = max_v_BL - compare_value
            ev_max_index = np.array(self.network.node[i]['expected_value_list_BL']).argmax()
            rewiring_target = self.network.node[i]['BL'][ev_max_index]

        # check if the rewiring target accept the proposal
        # NEW
        if rewiring_strategy == 2:
            sight = self.network.node[i]['max_conn_num'] - len(self.getNeighbors(i)) + 1
        elif rewiring_strategy == 3:
            sight = self.network.node[i]['max_conn_num'] * 2 - len(self.getNeighbors(i)) + 1
        else:
            sight = self.network.node[i]['max_conn_num'] * 1.5 - len(self.getNeighbors(i)) + 1
        # print('my sight,', sight)
        # print(ev_m)
        # print(ev_m * sight)
        if ev_m * sight * self.rewiring_sight > self.rewiring_cost:
        # if ev_m * sight * self.rewiring_sight / 2 > self.rewiring_cost:
            # print('propose rewiring...')
            if self._getRewiringResponse(rewiring_target, i) == 1:
                # do rewiring
                # unlink the worst neighbor if reach the limitation

                if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
                    self._unlinkWorstNeighbor(i)
                if len(self.getNeighbors(rewiring_target)) == self.network.node[rewiring_target]['max_conn_num']:
                    self._unlinkWorstNeighbor(rewiring_target)

                # rewire target
                # update information
                # print('Rewire agent', rewiring_target)

                if rewiring_target in self.network.node[i]['S_']:
                    # print('S_', self.network.node[i]['S_'], ', BL', self.network.node[i]['BL'])
                    self.network.node[i]['S_'].remove(rewiring_target)
                    # TO-DO
                    if i in self.network.node[rewiring_target]['S_']:
                        self.network.node[rewiring_target]['S_'].remove(i)

                    self.network.add_edge(i, rewiring_target)
                    # reveal the reward matrix of the unknown agent and update
                    self._revealTheUnknownGameForEdge(i, rewiring_target)
                else:
                    self.network.node[i]['BL'].remove(rewiring_target)
                    self.network.node[rewiring_target]['BL'].remove(i)
                # enable the connection
                self.network.edge[i][rewiring_target]['is_connected'] = 0

                # TO-DO
                # the rewiring pair shares the cost
                self.network.node[i]['ac'] += self.rewiring_cost
                # self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
                # self.network.node[rewiring_target]['ar'] += self.rewiring_cost
                self.network.node[i]['rewiring_time'] += 1
            else:
                # do nothing
                # print('Target agent ', rewiring_target, ' refused the rewiring.')
                return
        else:
            # print('Rewiring not satisfying.')
            return
        return

    # def _OptimalRewireMax(self, i):
    #     # pick Optimal target
    #
    #     # calculate BL
    #     self.network.node[i]['expected_value_list_BL'] = self.calculateExpectedRewardInBL(i)
    #     # calculate the lambda value of each neighbor in S' and get the maximum expected value ev
    #     self.network.node[i]['index_z_value_list'] = self.calculateLambdaInS_(i)
    #
    #     max_v_S_ = -1 if self.network.node[i]['index_z_value_list'] == [] \
    #         else max(self.network.node[i]['index_z_value_list'])
    #     max_v_BL = -1 if self.network.node[i]['expected_value_list_BL'] == [] \
    #         else max(self.network.node[i]['expected_value_list_BL'])
    #     print(max_v_S_, ' ', max_v_BL)
    #     if max_v_S_ >= max_v_BL:
    #         ev_m = max_v_S_
    #         ev_max_index = np.array(self.network.node[i]['index_z_value_list']).argmax()
    #         rewiring_target = self.network.node[i]['S_'][ev_max_index]
    #     else:
    #         ev_m = max_v_BL
    #         ev_max_index = np.array(self.network.node[i]['expected_value_list_BL']).argmax()
    #         rewiring_target = self.network.node[i]['BL'][ev_max_index]
    #
    #     # check if the rewiring target accept the proposal
    #     if ev_m * self.rewiring_sight > self.rewiring_cost / 2 and self._getRewiringResponse(rewiring_target, i) == 1:
    #         # do rewiring
    #         # unlink the worst neighbor if reach the limitation
    #
    #         if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
    #             self._unlinkWorstNeighbor(i)
    #         if len(self.getNeighbors(rewiring_target)) == self.network.node[rewiring_target]['max_conn_num']:
    #             self._unlinkWorstNeighbor(rewiring_target)
    #
    #         # rewire target
    #         # update information
    #         print('Rewire agent', rewiring_target)
    #         if rewiring_target in self.network.node[i]['S_']:
    #             self.network.node[i]['S_'].remove(rewiring_target)
    #             # TO-DO
    #             if i in self.network.node[rewiring_target]['S_']:
    #                 self.network.node[rewiring_target]['S_'].remove(i)
    #
    #             self.network.add_edge(i, rewiring_target)
    #             # reveal the reward matrix of the unknown agent and update
    #             self._revealTheUnknownGameForEdge(i, rewiring_target)
    #         else:
    #             self.network.node[i]['BL'].remove(rewiring_target)
    #             self.network.node[rewiring_target]['BL'].remove(i)
    #         # enable the connection
    #         self.network.edge[i][rewiring_target]['is_connected'] = 0
    #
    #         # TO-DO
    #         # the rewiring pair shares the cost
    #         self.network.node[i]['ac'] += self.rewiring_cost / 2
    #         self.network.node[rewiring_target]['ac'] += self.rewiring_cost / 2
    #         self.network.node[i]['rewiring_time'] += 1
    #     else:
    #         # do nothing
    #         print('Target agent ', rewiring_target, ' refused the rewiring.')
    #     return

    # get rewiring response from target for i
    def _getRewiringResponse(self, target, i):
        # print('asking response...')
        rewiring_strategy = self.network.node[target]['rewiring_strategy']

        maximum_expected_reward = self.calculateUpperBound(target)
        minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(target)
        sight = self.network.node[target]['max_conn_num'] - len(self.getNeighbors(target)) + 1

        # accept at random
        if rewiring_strategy == 0:
            return 1 if random.uniform(0,1) < 0.5 else 0
        # accept with HE
        p = self.oppActionDis[target, i]
        # game = self.network.edge[target][i]['game']
        if rewiring_strategy == 1:
            # if known agent
            if i in self.network.node[target]['BL']:
                game = self.network.edge[target][i]['game']
                expected_value = max(p * game[0,0] + (1 - p) * game[0,1], p * game[1,0] + (1 - p) * game[1,1])
                # return 1 if expected_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
                return 1 if expected_value > 0 else 0
                # return 1 if expected_value * self.rewiring_sight > 0 else 0
                # return 1 if (expected_value - minimum_expected_reward) * self.rewiring_sight > 0 else 0
                # return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            # if i in self.network.node[target]['S_']:
            HE_value = self.calculateHEValue(target, i)
            # return 1 if HE_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
            return 1 if HE_value * self.rewiring_sight > 0 else 0
            # return 1 if HE_value * self.rewiring_sight > self.rewiring_cost / 2 else 0

        if rewiring_strategy == 2:
            # if known agent
            sight = self.network.node[target]['max_conn_num'] - len(self.getNeighbors(target)) + 1
            # print('target sight,', sight)
            if i in self.network.node[target]['BL']:
                game = self.network.edge[target][i]['game']
                expected_value = max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1])
                # return 1 if expected_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
                # return 1 if (expected_value - minimum_expected_reward) * sight * self.rewiring_sight / 2 > 0 else 0
                return 1 if (expected_value - minimum_expected_reward) * sight * self.rewiring_sight > 0 else 0
                # return 1 if (expected_value - minimum_expected_reward) * self.rewiring_sight  > 0 else 0
                # return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            # TO-DO
            # check target connection numbers
            # if i in self.network.node[target]['S_']:
            minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(target)
            lamb_value = self.calculateLambdaValue(target, i, minimum_expected_reward, sec_minimum_expected_reward)
            # return 1 if lamb_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
            # return 1 if lamb_value * sight * self.rewiring_sight / 2 > 0 else 0
            return 1 if lamb_value * sight * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > self.rewiring_cost / 2 else 0

        if rewiring_strategy == 3:
            # if known agent
            sight = self.network.node[target]['max_conn_num'] * 2 - len(self.getNeighbors(target)) + 1
            # print('target sight,', sight)
            if i in self.network.node[target]['BL']:
                game = self.network.edge[target][i]['game']
                expected_value = max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1])
                # return 1 if expected_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
                # return 1 if (expected_value - maximum_expected_reward) * sight * self.rewiring_sight / 2 > 0 else 0
                return 1 if (expected_value - maximum_expected_reward) * sight * self.rewiring_sight > 0 else 0
                # return 1 if (expected_value - maximum_expected_reward) * self.rewiring_sight > 0 else 0
                # return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            # TO-DO
            # check target connection numbers
            # if i in self.network.node[target]['S_']:
                # minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(target)
            maximum_expected_reward = self.calculateUpperBound(target)
            lamb_value = self.calculateLambdaValueMax(target, i, maximum_expected_reward)
            # print('--op lam', lamb_value, 'max', maximum_expected_reward)
            # return 1 if lamb_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
            # return 1 if lamb_value * sight * self.rewiring_sight / 2> 0 else 0
            return 1 if lamb_value * sight * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > self.rewiring_cost / 2 else 0

        if rewiring_strategy == 4:
            sight = self.network.node[target]['max_conn_num'] * 1.5 - len(self.getNeighbors(target)) + 1
            # print('target sight,', sight)
            mean_expected_value = self.calculateMean(target)
            # if known agent
            if i in self.network.node[target]['BL']:
                game = self.network.edge[target][i]['game']
                expected_value = max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1])
                # return 1 if expected_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
                # return 1 if (expected_value - mean_expected_value) * sight * self.rewiring_sight / 2 > 0 else 0
                return 1 if (expected_value - mean_expected_value) * sight * self.rewiring_sight > 0 else 0
                # return 1 if (expected_value - mean_expected_value) * self.rewiring_sight > 0 else 0
                # return 1 if expected_value * self.rewiring_sight > self.rewiring_cost / 2 else 0
            # if unknown agent
            # TO-DO
            # check target connection numbers
            # if i in self.network.node[target]['S_']:
                # minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(target)

            # lamb_value = self.calculateLambdaValueMax(target, i, mean_expected_value)
            minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(target)
            mean_expected_reward = self.calculateMean(target)
            # index_z_list_S_.append(self.calculateLambdaValueMax(i, j, mean_expected_reward))
            if len(self.getNeighbors(target)) == self.network.node[i]['max_conn_num']:
                value = self.calculateHEValue(target, i)
                out = (value - minimum_expected_reward) / len(self.getNeighbors(i))
            else:
                value = self.calculateHEValue(target, i)
                out = (mean_expected_reward * len(self.getNeighbors(i)) + value) / (
                len(self.getNeighbors(i)) + 1) - mean_expected_reward
            lamb_value = out
            # print('--op lam', lamb_value, 'max', maximum_expected_reward)
            # return 1 if lamb_value * self.rewiring_sight + self.rewiring_cost > 0 else 0
            # return 1 if lamb_value * sight * self.rewiring_sight / 2 > 0 else 0
            return 1 if lamb_value * sight * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > 0 else 0
            # return 1 if lamb_value * self.rewiring_sight > self.rewiring_cost / 2 else 0

        print('')
        print('aaaa')
        print('')

        return 0

    # unlink the worst neighbor before rewiring
    def _unlinkWorstNeighbor(self, i):
        rewiring_strategy = self.network.node[i]['rewiring_strategy']

        # unlink a random neighbor
        if rewiring_strategy == 0:
            unlink_agent_index = random.randint(0, len(self.getNeighbors(i)) - 1)
            unlink_agent_no = self.getNeighbors(i)[unlink_agent_index]
            if self.network.edge[i][unlink_agent_no]['is_connected'] == 1:
                print('Error: Target can not be unlink.')
                return
            self.network.node[i]['BL'].append(unlink_agent_no)
            self.network.node[unlink_agent_no]['BL'].append(i)
            # self.network.remove_edge(i, unlink_agent_no)
            self.network.edge[i][unlink_agent_no]['is_connected'] = 1
            # print(i, 'Unlink agent ', unlink_agent_no)
        # unlink the worst neighbor
        else:
            # 1) get the maximum(minimum) expected value in S
            self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
            worst_agent_index = np.array(self.network.node[i]['expected_value_list']).argmin()
            worst_agent_no = self.getNeighbors(i)[worst_agent_index]
            if self.network.edge[i][worst_agent_no]['is_connected'] == 1:
                print('Error: Target can not be unlink.')
                return
            self.network.node[i]['BL'].append(worst_agent_no)
            self.network.node[worst_agent_no]['BL'].append(i)
            self.network.edge[i][worst_agent_no]['is_connected'] = 1
            # self.network.remove_edge(i, worst_agent_no)
            # print(i, 'Unlink agent ', worst_agent_no)
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

        epsilon = max(epsilon - int(count / 1000) * 0.1, 0.0)

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
        p_j = self._updateOppoPolicy(i, j, j_action)
        p_i = self._updateOppoPolicy(j, i, i_action)

        self.oppActionDis[i, j] = p_j
        self.oppActionDis[j, i] = p_i

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

    def calculateLambdaInS_(self, i):
        rewiring_strategy = self.network.node[i]['rewiring_strategy']
        index_z_list_S_ = []
        S_ = self.network.node[i]['S_']
        for j in S_:
            if rewiring_strategy == 2:
                minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(i)
                index_z_list_S_.append(
                    self.calculateLambdaValue(i, j, minimum_expected_reward, sec_minimum_expected_reward))
            elif rewiring_strategy == 3:
                maximum_expected_reward = self.calculateUpperBound(i)
                index_z_list_S_.append(
                    self.calculateLambdaValueMax(i, j, maximum_expected_reward))
            # origin
            elif rewiring_strategy == 4:
                minimum_expected_reward, sec_minimum_expected_reward = self.calculateBaselines(i)
                mean_expected_reward = self.calculateMean(i)
                # index_z_list_S_.append(self.calculateLambdaValueMax(i, j, mean_expected_reward))
                if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
                    value = self.calculateHEValue(i, j)
                    out = (value - minimum_expected_reward) / len(self.getNeighbors(i))
                else:
                    value = self.calculateHEValue(i, j)
                    out = (mean_expected_reward * len(self.getNeighbors(i)) + value) / (len(self.getNeighbors(i)) + 1) - mean_expected_reward
                index_z_list_S_.append(out)
        return index_z_list_S_

    def calculateHEValueInS_(self, i):
        expected_value_S_ = []
        S_ = self.network.node[i]['S_']
        # print(S_)
        for j in S_:
            expected_value_S_.append(self.calculateHEValue(i, j))
        return expected_value_S_

    def calculateExpectedRewardInBL(self, i):
        expectedReward_BL = []
        BL = self.network.node[i]['BL']
        # print(S)
        for j in BL:
            # evaluate opponent's possibility to choose action a(for the sight of some agent)
            p = self.oppActionDis[i, j]
            game = self.network.edge[i][j]['game']
            expectedReward_BL.append(max(p * game[0, 0] + (1 - p) * game[0, 1], p * game[1, 0] + (1 - p) * game[1, 1]))

        return expectedReward_BL

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

    def calculateUpperBound(self, i):
        self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
        expected_value_list = self.network.node[i]['expected_value_list']
        maximum_expected_reward = 0 if expected_value_list == [] else max(expected_value_list)
        return maximum_expected_reward

    def calculateMean(self, i):
        self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
        expected_value_list = self.network.node[i]['expected_value_list']
        mean_expected_reward = 0 if expected_value_list == [] else sum(expected_value_list)/ len(expected_value_list)
        return mean_expected_reward

    def calculateBaselines(self, i):
        # 1) get the maximum(minimum) expected value in S
        self.network.node[i]['expected_value_list'] = self.calculateExpectedRewardInS(i)
        expected_value_list = self.network.node[i]['expected_value_list']
        minimum_expected_reward = 0 if expected_value_list == [] else min(expected_value_list)
        sec_minimum_expected_reward = 0
        if len(self.getNeighbors(i)) > 1:
            expected_value_list.remove(minimum_expected_reward)
            sec_minimum_expected_reward = min(expected_value_list)
        return minimum_expected_reward, sec_minimum_expected_reward

    def calculateLambdaValue(self, i, j, minimum_expected_reward, sec_minimum_expected_reward):
        p = self.oppActionDis[i, j]
        # sort the agent_no and neighbor, small one at left
        left = min(i, j)
        right = max(i, j)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * self.agent_num + right

        # 1) get the maximum(minimum) expected value in S
        # minimum_expected_reward ,sec_minimum_expected_reward = self.calculateBaselines(i, j)

        if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
            if self.distribution_type == 0:
                # U(a1,b1) U(a2,b2)
                a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]
                b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1] - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4]

                a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2] - p * self.rewardDisMatrix[index_in_dis_matrix, 5]
                b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3] - p * self.rewardDisMatrix[index_in_dis_matrix, 5]



                if len(self.getNeighbors(i)) == 1:
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

                if len(self.getNeighbors(i)) == 1:
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
            if self.distribution_type == 0:
                # U(a1,b1) U(a2,b2)
                a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] - (1 - p) * self.rewardDisMatrix[
                    index_in_dis_matrix, 4]
                b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1] - (1 - p) * self.rewardDisMatrix[
                    index_in_dis_matrix, 4]

                a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2] - p * self.rewardDisMatrix[
                    index_in_dis_matrix, 5]
                b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3] - p * self.rewardDisMatrix[
                    index_in_dis_matrix, 5]

                z1 = ut.calculateLambdaIndexNotBreak(a1, b1, self.rewiring_sight,
                                                 minimum_expected_reward)
                # z2 = calculateIndex(a2, b2, c, sight)
                z2 = ut.calculateLambdaIndexNotBreak(a2, b2, self.rewiring_sight,
                                                 minimum_expected_reward)
                return max(z1, z2)

            if self.distribution_type == 1:
                # beta(a,b)
                a1 = self.rewardDisMatrix[index_in_dis_matrix, 0]
                b1 = self.rewardDisMatrix[index_in_dis_matrix, 1]

                a2 = self.rewardDisMatrix[index_in_dis_matrix, 2]
                b2 = self.rewardDisMatrix[index_in_dis_matrix, 3]

                z1 = ut.calculateLambdaIndexInBetaNotBreak(a1, b1, self.rewiring_sight,
                                                       minimum_expected_reward,
                                                       p)
                # z2 = calculateIndex(a2, b2, c, sight)
                z2 = ut.calculateLambdaIndexInBetaNotBreak(a2, b2, self.rewiring_sight,
                                                       minimum_expected_reward,
                                                       1 - p)
                return max(z1, z2)

            return

    def calculateLambdaValueMax(self, i, j, maximum_expected_reward):
        p = self.oppActionDis[i, j]
        # sort the agent_no and neighbor, small one at left
        left = min(i, j)
        right = max(i, j)
        # print(left)
        # print(right)
        index_in_dis_matrix = left * self.agent_num + right

        # 1) get the maximum(minimum) expected value in S
        # minimum_expected_reward ,sec_minimum_expected_reward = self.calculateBaselines(i, j)

        # if len(self.getNeighbors(i)) == self.network.node[i]['max_conn_num']:
        # TO-DO
        if self.distribution_type == 0:
            # U(a1,b1) U(a2,b2)
            a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] - (1 - p) * self.rewardDisMatrix[
                index_in_dis_matrix, 4]
            b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1] - (1 - p) * self.rewardDisMatrix[
                index_in_dis_matrix, 4]

            a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2] - p * self.rewardDisMatrix[
                index_in_dis_matrix, 5]
            b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3] - p * self.rewardDisMatrix[
                index_in_dis_matrix, 5]

            if len(self.getNeighbors(i)) == 1:
                z1 = (a1 + b1) / 2 - maximum_expected_reward
                z2 = (a2 + b2) / 2 - maximum_expected_reward
            else:
                # z1 = calculateIndex(a1, b1, c, sight)
                z1 = ut.calculateLambdaIndexMax(a1, b1, self.rewiring_sight,
                                                maximum_expected_reward)
                # z2 = calculateIndex(a2, b2, c, sight)
                z2 = ut.calculateLambdaIndexMax(a2, b2, self.rewiring_sight,
                                                maximum_expected_reward)
            return max(z1, z2)

        if self.distribution_type == 1:
            # beta(a,b)
            a1 = self.rewardDisMatrix[index_in_dis_matrix, 0]
            b1 = self.rewardDisMatrix[index_in_dis_matrix, 1]

            a2 = self.rewardDisMatrix[index_in_dis_matrix, 2]
            b2 = self.rewardDisMatrix[index_in_dis_matrix, 3]

            if len(self.getNeighbors(i)) == 1:
                z1 = p * a1 / (a1 + b1) - (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 4] \
                     - maximum_expected_reward
                z2 = (1 - p) * a2 / (a2 + b2) - p * self.rewardDisMatrix[index_in_dis_matrix, 5] \
                     - maximum_expected_reward
            else:
                # z1 = calculateIndex(a1, b1, c, sight)
                z1 = ut.calculateLambdaIndexMaxInBeta(a1, b1, self.rewiring_sight,
                                                      maximum_expected_reward,
                                                      p)
                # z2 = calculateIndex(a2, b2, c, sight)
                z2 = ut.calculateLambdaIndexMaxInBeta(a2, b2, self.rewiring_sight,
                                                      maximum_expected_reward,
                                                      1 - p)
            return max(z1, z2)
        # else:
        #     # TO-DO
        #     # if target has not reached the connection limitation
        #     # Lambda should be calculated with different equations
        #     if self.distribution_type == 0:
        #         # U(a1,b1) U(a2,b2)
        #         a1 = p * self.rewardDisMatrix[index_in_dis_matrix, 0] - (1 - p) * self.rewardDisMatrix[
        #             index_in_dis_matrix, 4]
        #         b1 = p * self.rewardDisMatrix[index_in_dis_matrix, 1] - (1 - p) * self.rewardDisMatrix[
        #             index_in_dis_matrix, 4]
        #
        #         a2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 2] - p * self.rewardDisMatrix[
        #             index_in_dis_matrix, 5]
        #         b2 = (1 - p) * self.rewardDisMatrix[index_in_dis_matrix, 3] - p * self.rewardDisMatrix[
        #             index_in_dis_matrix, 5]
        #
        #         z1 = ut.calculateLambdaIndexNotBreak(a1, b1, self.rewiring_sight,
        #                                              minimum_expected_reward)
        #         # z2 = calculateIndex(a2, b2, c, sight)
        #         z2 = ut.calculateLambdaIndexNotBreak(a2, b2, self.rewiring_sight,
        #                                              minimum_expected_reward)
        #         return max(z1, z2)
        #
        #     if self.distribution_type == 1:
        #         # beta(a,b)
        #         a1 = self.rewardDisMatrix[index_in_dis_matrix, 0]
        #         b1 = self.rewardDisMatrix[index_in_dis_matrix, 1]
        #
        #         a2 = self.rewardDisMatrix[index_in_dis_matrix, 2]
        #         b2 = self.rewardDisMatrix[index_in_dis_matrix, 3]
        #
        #         z1 = ut.calculateLambdaIndexInBetaNotBreak(a1, b1, self.rewiring_sight,
        #                                                    minimum_expected_reward,
        #                                                    p)
        #         # z2 = calculateIndex(a2, b2, c, sight)
        #         z2 = ut.calculateLambdaIndexInBetaNotBreak(a2, b2, self.rewiring_sight,
        #                                                    minimum_expected_reward,
        #                                                    1 - p)
        #         return max(z1, z2)


    # get the expected reward list for set S
    def calculateExpectedRewardInS(self, i):
        expectedReward_S = []
        S = self.getNeighbors(i)
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
        return avg_pol * 99 / 100 + pol / 100
        # return avg_pol + 1 / count * (pol - avg_pol)

    def _updatePolicy(self, q, op_pol, pol, avg_pol, delta_w, delta_l):
        q_a = (op_pol * q[0, 0] + (1 - op_pol) * q[0, 1])
        q_b = (op_pol * q[0, 2] + (1 - op_pol) * q[0, 3])
        expect_value_current = pol * q_a + (1 - pol) * q_b
        expect_value_avg = avg_pol * q_a + (1 - avg_pol) * q_b
        delta = delta_w if expect_value_current > expect_value_avg else delta_l

        return min(pol + delta, 1.0) if q_a >= q_b else max(pol - delta, 0.0)

    def _updateOppoPolicy(self, i, j, j_action):
        p_old = self.oppActionDis[i, j]
        return ut.updatePolcyFromFrequecy(p_old, j_action)

    def getNeighbors(self, i):
        S = []
        for n in self.network.neighbors(i):
            if self.network.edge[i][n]['is_connected'] == 0:
                S.append(n)
        return S
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

        # init is_connected for edges
        for (i, j) in N.edges():
            N.edge[i][j]['is_connected'] = 0

        # init nodes
        for i in range(self.agent_num):
            N.node[i]['S_'] = []
            # initial the BL for black-list
            N.node[i]['BL'] = []
            # initial the accumulate reward and cost for agent(node)
            N.node[i]['ar'] = 0
            N.node[i]['ac'] = 0

            N.node[i]['expected_value_list'] = []
            N.node[i]['index_z_value_list'] = []
            N.node[i]['expected_value_list_S_'] = []
            N.node[i]['expected_value_list_BL'] = []

            N.node[i]['rebuild_mark'] = 0
            N.node[i]['rewiring_time'] = 0

            # additionally random initialize the strategy of agents, 10% random, 10% Never, 30% HE, and 50 % Optimal
            N.node[i]['rewiring_strategy'] = self.rewiring_strategy
            # ran = random.uniform(0, 1)
            # if ran < 0.33:
            #     N.node[i]['rewiring_strategy'] = 0
            # elif ran < 0.66:
            #     N.node[i]['rewiring_strategy'] = 1
            # else:
            #     N.node[i]['rewiring_strategy'] = 2

            # 0 for BR, 1 for WoLF, 2 for JAL
            N.node[i]['gaming_strategy'] = self.learning_strategy
            # init game_strategy
            # ran = random.uniform(0, 1)
            # if ran <= 1 / 3:
            #     N.node[i]['gaming_strategy'] = 1
            # elif ran <= 2 / 3:
            #     N.node[i]['gaming_strategy'] = 2

            N.node[i]['max_conn_num'] = self.init_conn_num

        return N

    def _initPotentialPeers(self):
        # adjust neighbors and potential peers
        for i in range(self.agent_num):
            # links are initial links, the set S
            S = self.getNeighbors(i)
            # print(i, ':', S)
            for j in range(self.pot_peer_num):
                pot_peer_no = random.randint(0, self.agent_num - 1)
                while pot_peer_no == i or pot_peer_no in S:
                    pot_peer_no = random.randint(0, self.agent_num - 1)

                self.network.node[i]['S_'].append(pot_peer_no)
            # complex version
            # TO-DO
            # tmp_size = len(S) - self.init_conn_num
            # for j in range(tmp_size):
            #     pot_peer = np.random.choice(S)
            #     while len(self.getNeighbors(pot_peer)) == self.init_conn_num:
            #         pot_peer = np.random.choice(S)
            #     S.remove(pot_peer)
            #     print('unlink',pot_peer)
            #     self.network.edge[i][pot_peer]['is_connected'] = 1
            #     pot_peer_index = random.randint(0, self.agent_num - 1)
            #     while pot_peer_index == i or pot_peer_index in S:
            #         pot_peer_index = random.randint(0, self.agent_num - 1)
            #     self.network.node[i]['S_'].append(pot_peer)
            #     self.network.node[pot_peer]['S_'].append(i)

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
            if self.network.edge[i][j]['is_connected'] == 0:
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
        self.network.edge[i][j]['alpha_value'] = [al, al2]

        self.network.edge[i][j]['policy_pair'] = [0.5, 0.5]
        # self.network.edge[i][j]['avg_policy_pair'] = [0.5, 0.5]
        self.network.edge[i][j]['avg_policy_pair'] = [0.0, 0.0]

        # q_table_pair = np.array([[0.0,0.0], [0.0,0.0]])
        q_table_pair = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        q_table_pair = np.matrix(q_table_pair)
        self.network.edge[i][j]['q-table_pair'] = q_table_pair
        self.network.edge[i][j]['e-epsilon'] = 1.0
        self.network.edge[i][j]['alpha'] = 0.2
        self.network.edge[i][j]['count'] = 0

        self.network.edge[i][j]['delta_w'] = 0.0001
        self.network.edge[i][j]['delta_l'] = 0.0002

        # self.network.edge[i][j]['is_connected'] = 0
        # TO-DO params

    # initialize the opponents action distribution for every agent pair
    # return a numpy matrix
    def _initialOpponentsActionDistribution(self):
        oppActionDis = []
        for i in range(self.agent_num):
            subDis = np.random.random(size=self.agent_num)
            oppActionDis.append(subDis)
        return np.matrix(oppActionDis)

    def printAgentInfo(self, b):
        group_reward = []
        group_rewiring = 0.0

        if b > 0:
            print('')
            print('Agents info below.')
            print('----------------------------------')

        for agent in self.network.nodes():
            if b > 0:
                print('Agent: ' + str(agent))
                print('  Neighbors: ' + str(self.getNeighbors(agent)))
                print('  Potential peers: ' + str(self.network.node[agent]['S_']))
                print('  Unlinked Neighbors: ' + str(self.network.node[agent]['BL']))
                print('  Rewiring time: ' + str(self.network.node[agent]['rewiring_time']))
                print('  AR: ' + str(self.network.node[agent]['ar']))
                print('  AC: ' + str(self.network.node[agent]['ac']))
                print('  Reward: ' + str(self.network.node[agent]['ar'] - self.network.node[agent]['ac']))
                print('----------------------------------')
            group_reward.append(self.network.node[agent]['ar'] - self.network.node[agent]['ac'])
            group_rewiring = group_rewiring + self.network.node[agent]['rewiring_time']

        if b > 0:
            print('Group average rewiring time: ' + str(group_rewiring / self.agent_num))
            print('Group average reward: ' + str(sum(group_reward) / self.agent_num))
            print('Group highest reward: ' + str(max(group_reward)))
            print('Group lowest reward: ' + str(min(group_reward)))

        return group_reward, group_rewiring

    def getInteractionPayoff(self):
        inter_reward = []

        for agent in self.network.nodes():
            inter_reward.append(self.network.node[agent]['ar'])

        return inter_reward
