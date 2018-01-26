import envs.BasicEnv as Env
import random
import matplotlib.pyplot as plt


# Params
INTERACTION_ROUND = 10000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 40
REWIRING_PROBABILITY = 0.01

K = 2

REWIRING_STRATEGY = 2

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 10

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

# c_list = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
# c_list = [20.0, 40.0, 60.0, 80.0, 100.0]
c_list = [20.0, 40.0, 80.0, 120.0, 160.0, 200.0]
# c_list = [120.0, 140.0, 160.0, 180.0, 200.0]
# c_list = [150.0, 250.0, 300.0, 350.0, 400.0]
sight_list = []
for s in range(1, 21):
    sight_list.append(s*5.0)
# for s in range(1, 21):
#     sight_list.append(s*0.1)

# for m in range(1, 41):
#     sight_list.append(2000 + m*200)

y_list = []

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

    for c in c_list:
        y_axi = []
        print('For cost: ', c)
        REWIRING_COST = c
        for sight in sight_list:
            print('For K: ', sight)
            K = sight

            average_reward = []
            highest_reward = []
            lowest_reward = []
            average_rewiring = 0.0

            for repeat in range(EPISODE):
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
                group_reward, group_rewiring = env.printAgentInfo(0)
                average_reward.append(sum(group_reward) / AGENT_NUM)
                highest_reward.append(max(group_reward))
                lowest_reward.append(min(group_reward))

                average_rewiring = average_rewiring + group_rewiring / AGENT_NUM

                # print(env.oppActionDis)

            # print('--------------------------------------------------------------------')
            # print('--------------------------------------------------------------------')
            # print('Final outputs:')
            # print('Mean average rewiring: ' + str(average_rewiring / EPISODE))
            print('Mean average reward: ' + str(sum(average_reward) / EPISODE))
            # print('Mean highest reward: ' + str(sum(highest_reward) / EPISODE))
            # print('Mean lowest reward: ' + str(sum(lowest_reward) / EPISODE))
            # print('--------------------------------------------------------------------')
            # print('--------------------------------------------------------------------')

            y_axi.append(sum(average_reward) / EPISODE)
        y_list.append(y_axi)

    print('======================================')
    print('Results below.')
    for l in y_list:
        print(l)
    x_axi = sight_list

    # plt.xlim(0.0, 1.0)
    # plt.ylim(-0.1, 1.0)

    # for y in y_list:
    #     plt.plot(x_axi, y)
    # plt.title('(50,3,9) - Optimal - k-sight')
    #
    # plt.show()




############################################################
# Results

# 18/01/24
# Varying sight

# [0.1 - 2.0]
# [1149.9299700903227, 1135.9762871643825, 1171.5733254914564, 1311.6113229201289, 1351.7257808745635, 1378.6945537433439, 1356.7726874214254, 1375.824807503393, 1369.0904659658358, 1375.061890408248, 1379.6721230934315, 1362.751096549795, 1377.4594968147826, 1371.7722746669151, 1372.1105486301772, 1383.0035588886533, 1384.524889072467, 1374.4689017644776, 1382.9795984909608, 1388.0451564906134]
# [1137.2830075369234, 1134.323468940896, 1133.9049820270884, 1140.8610173147997, 1136.3985970315873, 1193.9545921114463, 1245.5138826411801, 1296.5557990176053, 1313.0689993029068, 1314.3633522722785, 1309.3712624242969, 1318.7226877150588, 1323.3756106396297, 1324.6715755061809, 1323.8769720453599, 1322.2436533061098, 1329.27567274282, 1322.4462889752854, 1322.2979396249164, 1318.9130330159724]
# [1153.8831803190674, 1127.2083705683453, 1155.4443322752522, 1143.5376462754748, 1159.6954491091126, 1133.9603214663953, 1147.4132950944374, 1134.3374048201633, 1139.9221553038246, 1149.4545537095321, 1159.4325178225349, 1191.5475244318231, 1195.8748151959685, 1209.8715843427385, 1214.9046114641146, 1238.8293206238131, 1238.9859160125779, 1230.5706989192745, 1238.1400304685253, 1230.2871140805939]
# [1146.7035216281445, 1149.6991368741242, 1137.588304807088, 1151.8828424343587, 1142.7017601519935, 1155.4146113446031, 1154.7073065623358, 1125.4754499719036, 1152.019096230883, 1138.7850232964968, 1159.7186453220318, 1137.8770441238748, 1153.0421854521214, 1146.8248825306389, 1132.6248486052195, 1163.9267052449043, 1157.0212649538948, 1165.3776479041226, 1173.3902343759728, 1163.7311604792874]
# [1147.4518969188516, 1151.192367066046, 1160.5758855285333, 1154.116093408386, 1142.0918942251719, 1132.0287350119238, 1145.6022962928387, 1126.681077422098, 1134.1151287633013, 1156.5947829494364, 1149.7907160112577, 1146.3106426259128, 1146.611288363863, 1148.5924528712599, 1159.239510948001, 1160.4387221928387, 1147.6679867489547, 1146.166346274113, 1148.0633089509063, 1149.9128431611261]
# [1140.7465159534099, 1153.0967902897382, 1151.4453909541194, 1148.444068275566, 1154.7287372696194, 1131.4601274854306, 1139.9287390713064, 1133.4344338582971, 1160.9989053675035, 1136.8378973544925, 1145.0867354536729, 1136.3950640739181, 1143.1906865355081, 1149.7366943207326, 1159.360825874018, 1133.3592924520042, 1146.2544735085662, 1157.923960795348, 1145.9984821719102, 1133.5485125022901]

#
# [13409.589571645283, 13271.982102752163, 13518.698292039155, 13178.10525143646, 13163.44505149744, 13261.568345744006, 13298.223522119177, 13209.121284744586, 13403.026728025308, 13273.263598729158, 13241.264394865895, 13221.472190003729, 13415.824748468372, 13239.4772235034, 13342.271965989081, 13361.534626491635, 13345.349188786731, 13421.150374284733, 13363.946097997756, 13341.796590296421]
# [13120.92225037665, 13347.052825666551, 13432.863319862005, 13215.849566943978, 13349.690045218274, 13152.607790191232, 13241.450565675306, 13335.159739691653, 13259.50536093433, 13264.824397042623, 13411.334208273949, 13278.25191197063, 13267.064359119066, 13257.728957268231, 13308.467477179734, 13429.193518076343, 13370.904487112037, 13095.521117249913, 13436.653889760088, 13503.587552863875]
# [12936.356260260516, 13056.11680060369, 13371.541613367759, 13206.241863825966, 13231.875746411057, 13068.599107057787, 13317.732736378623, 13530.383663390685, 13332.595229890281, 13347.506381474293, 13271.600080949511, 13262.550822738645, 13356.634100245737, 13336.617184913437, 13126.444736814918, 13224.79485304318, 13313.163883766023, 13199.910638282687, 13279.363117561024, 13266.309589357674]
# [12658.077943882898, 13056.977271831716, 13211.615414289636, 13307.733559623521, 13199.983141800936, 13231.336091674017, 13421.168579091804, 13127.885985799978, 13130.025816130767, 13479.47316936941, 13332.246255749782, 13101.210064355455, 13215.718228895585, 13342.832585768596, 13408.422876334342, 13265.369597158939, 13238.84792777929, 13378.385693954982, 13347.531223113168, 13350.621320766502]
# [12235.329364778152, 13070.121055513166, 13279.190464570265, 13037.288082510302, 13189.632333983604, 13293.041681436463, 13250.91898105325, 13192.678886686243, 13400.754511017654, 13322.206209738235, 13217.925539332031, 13192.34208310201, 13265.210669086988, 13287.729379737026, 13251.402521774824, 13248.338717867349, 13299.259198254125, 13290.892955127467, 13312.334931745998, 13242.937405046949]
# [11959.061790734999, 12578.545110463208, 13125.731762475616, 13111.25809150388, 13131.584407192631, 13087.926541025869, 13130.616793262012, 13229.690001479146, 13268.99300804272, 13256.358279836384, 13218.296795035689, 13361.66470209235, 13216.913853035934, 13145.645531684306, 13467.018737851889, 13297.423457786801, 13153.627547805951, 13237.794186546133, 12965.631895596185, 13250.839917211502]
