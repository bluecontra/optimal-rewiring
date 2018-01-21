import envs.BasicEnv as Env
import random
# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 5
NEIGHBORHOOD_SIZE = 10

REWIRING_COST = 20
REWIRING_PROBABILITY = 0.01

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0



if __name__ == '__main__':

    # init Env
    env = Env.BasicEnv(AGENT_NUM,
                       NETWORK_TYPE,
                       BANDWIDTH,
                       NEIGHBORHOOD_SIZE,
                       REWIRING_COST,
                       REWIRING_PROBABILITY,
                       DISTRIBUTION_TYPE)

    phi = env.phi

    for interation in range(INTERACTION_ROUND):
        # to avoid the the problem that decision order may influence the fairness
        # shuffle the order at every iteration
        agents = env.network.nodes()
        random.shuffle(agents)
        print('Env initialized.')

        # rewiring phase
        print('Rewiring phase.')
        for i in agents:
            if random.uniform(0, 1) < phi:
                # do rewire
                env._rewire(i)

        # interaction phase
        print('Interaction phase.')
        for i in env.network.nodes():
            # do interaction
            neighbors_num = len(env.network.neighbors(i))
            if neighbors_num > 0:
                # 1) randomly choose a opponent in S (choose the best opponent)
                oppo_index = random.randint(0, neighbors_num - 1)
                oppo_agent_no = env.network.neighbors(i)[oppo_index]

                # 2) agent i interacts with certain opponent
                env._interact(i,oppo_agent_no)
            else:
                print('agent ', i, ' has no neighbor.')

        # statistic
