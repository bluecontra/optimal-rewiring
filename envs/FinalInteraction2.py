import envs.FinalEnv as Env
# import envs.BasicEnv as Env
import random
import matplotlib.pyplot as plt


# Params
INTERACTION_ROUND = 1000

AGENT_NUM = 100
BANDWIDTH = 4
NEIGHBORHOOD_SIZE = 12

REWIRING_COST = 40
REWIRING_PROBABILITY = 0.01

# K for he
K = 4

REWIRING_STRATEGY = 0

NETWORK_TYPE = 0
DISTRIBUTION_TYPE = 0

EPISODE = 15

average_reward = []
highest_reward = []
lowest_reward = []
average_rewiring = 0.0

DEBUG = -1

c_sight_pair = []
for x in range(1,21):
    c_sight_pair.append([x * 0.0025])
print(c_sight_pair)


sg_result = []

if __name__ == '__main__':

    # for sg in [0,1,2,3,4]:
    for sg in [3]:
        result = []
        REWIRING_STRATEGY = sg

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

        print('--K for khe:', K)
        for [c] in c_sight_pair:
            y_axi = []

            average_reward = []
            highest_reward = []
            lowest_reward = []
            average_rewiring = 0.0

            # REWIRING_COST = c
            REWIRING_PROBABILITY = c
            print('--Rewiring cost:', REWIRING_COST)
            print('--Rewiring probability:', REWIRING_PROBABILITY)

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

            result.append(sum(average_reward) / EPISODE)
        print('Interim result:', result)
        sg_result.append(result)

    print('======================================')
    print('Results below.')

    for l in sg_result:
        print(l)

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
# adaptive sight
# 3 optimal rewiring strategy

# fix rewiring cost c = 40
# probability
# [[0.0025], [0.005], [0.0075], [0.01], [0.0125], [0.015], [0.0175], [0.02], [0.0225], [0.025], [0.0275], [0.03], [0.0325], [0.035], [0.0375], [0.04], [0.0425], [0.045], [0.0475], [0.05]]

# ran
# [1079.0808808867955, 1014.9999898697229, 953.85500861146647, 893.69781804112222, 850.24379583296547, 788.69935234470609, 726.58832856576919, 685.90326793425163, 637.3394012733628, 573.07131190588814, 518.56262497702539, 468.20590876864782, 415.8773729382338, 368.04309548919639, 318.78599693992021, 255.29037260368003, 211.34496134778016, 153.38718659097734, 101.05186086935586, 61.605265840060461]

# KHE K= 0.5,1,2,4
# [1280.4675697439513, 1222.8231394530064, 1312.9839574140831, 1197.0841939309366, 1154.7463922944685, 1146.7831533720678, 1155.2038183581269, 1156.0064787210652, 1136.0078893754317, 1133.7296265270131, 1138.7880971939921, 1141.6837912942874, 1146.3687078532535, 1157.6603374570034, 1151.5770824558244, 1140.1731931756526, 1132.8388417102178, 1132.9542968489814, 1139.4361009151262, 1154.1763757078704]
# [1272.2265447029097, 1218.0715204019061, 1138.9360343175254, 1081.5730950767588, 1152.9439497416618, 1306.4041429464457, 1272.6877904029982, 1209.5826472543329, 1140.2722919291505, 1144.0501937843328, 1151.4890532089014, 1141.4484750074309, 1143.5622577142335, 1152.4699865404511, 1149.5443145464592, 1145.9433975642976, 1134.4141509149274, 1150.5166939869634, 1137.7154940854025, 1140.7566089263278]
# [1268.5534530393172, 1229.5201088723159, 1141.1410739720957, 1055.796690051661, 963.37306243262401, 874.47739535542382, 785.68250132213518, 747.51386487220077, 785.79763936375525, 946.0718960808673, 1143.2608914813461, 1300.4867174995213, 1328.0473242617486, 1291.3213313282918, 1235.9285203685072, 1204.9378320116148, 1165.4865166105826, 1152.2629849521998, 1142.2842455363834, 1148.0024362434958]
# [1280.9404408855712, 1215.400306744534, 1142.4385239992509, 1050.9752389934993, 964.49020966620583, 870.38253315169504, 768.60922902874881, 662.90231480486261, 585.59357531796309, 485.99175861092408, 380.09966386692821, 302.80373864027723, 204.93816242702491, 122.97118535906993, 68.223041196095807, 82.401218995570758, 100.29398554538547, 166.74915339363861, 350.04357642990124, 542.51345913677505]

# opt
# [1251.541746483819, 1251.2832082282214, 1258.8470865423221, 1244.3419859165058, 1254.5147586277044, 1264.3609164740842, 1238.8932866343416, 1225.0705460651504, 1222.7886584199464, 1201.8108637034097, 1202.7972826449857, 1185.7668699570734, 1187.0745818620967, 1191.2593735463608, 1165.0142361832941, 1170.0631297274097, 1164.7643575234388, 1154.6847525730691, 1149.5292292662914, 1167.8874282543081]
# [1237.11075650065, 1269.8514817192893, 1257.1875878083185, 1265.5874856456346, 1241.2462526757702, 1252.0558716652897, 1247.7768253602283, 1226.1736328695483, 1220.0366425773823, 1211.5768300023165, 1208.2551783351691, 1197.6980397350051, 1172.8442008604754, 1176.2370575999944, 1156.0755480558994, 1148.3785788115201, 1163.770228146534, 1155.315191375531, 1137.2044721027787, 1152.3762847592695]

# [1251.541746483819, 1251.2832082282214, 1258.8470865423221, 1244.3419859165058, 1254.5147586277044, 1264.3609164740842, 1238.8932866343416, 1225.0705460651504, 1222.7886584199464, 1201.8108637034097, 1202.7972826449857, 1185.7668699570734, 1187.0745818620967, 1191.2593735463608, 1165.0142361832941, 1170.0631297274097, 1164.7643575234388, 1154.6847525730691, 1149.5292292662914, 1167.8874282543081]
# [1226.2790554966077, 1249.4689053129368, 1267.5464378710199, 1278.4141540937687, 1275.9606252383651, 1299.1668779517595, 1298.7074531990731, 1303.1529826847204, 1292.5214900457954, 1299.9645253503231, 1309.0431626940738, 1301.4778668452561, 1305.5323789971053, 1304.0100564838767, 1297.8369978755848, 1304.0198938547676, 1310.4660995592146, 1305.5667440084389, 1315.513494867751, 1314.2765962346234]

# [1309.7957567213894, 1333.4154727304992, 1343.865468983927, 1360.7054910560796, 1355.3802511115134, 1360.6467013416943, 1350.1783736683478, 1354.4362086844251, 1353.9456016709651, 1347.1794372982529, 1333.4533787055896, 1342.3770877060563, 1321.530796423795, 1294.997500332262, 1302.5156233209887, 1267.363125344365, 1249.9808954178022, 1235.1205133823996, 1214.7518236052294, 1195.6302418995961]

# fix rewiring cost c = 80

# ran
# [1037.6466641100367, 922.36306855574344, 811.78067376062768, 711.62772444712084, 585.21233173761686, 488.02051554150444, 375.40748992834443, 281.71060480979094, 178.57392028325594, 83.20149234042681, -15.563456426593193, -120.20653245607564, -231.02947008766452, -337.10639760539249, -428.92611045048915, -536.34070795728053, -647.38719807647112, -746.1854300585627, -834.63020148326314, -944.05584902767407]
#
# HE
# [1180.0483884924674, 1177.8873154470703, 1151.7964292434688, 1149.849242446621, 1146.2565729449686, 1134.3799797019578, 1146.5582841709913, 1140.6083422540673, 1160.4487822484741, 1144.8066584148089, 1146.1931603547639, 1144.4714028272856, 1152.5079577896859, 1142.7472327028702, 1144.2371884696411, 1142.7866109902491, 1148.0343461230993, 1137.9885353344778, 1149.997271714454, 1149.7515384298288]
# [1164.7983011509891, 1032.7199376502044, 1238.9336890397774, 1200.9812449004978, 1148.911869965706, 1143.1370785657384, 1134.6587960671309, 1144.0206062928316, 1133.8283389764531, 1140.0188634339424, 1145.6087212601965, 1145.7328393907858, 1149.5110576561876, 1131.9136331056295, 1143.3841527937179, 1134.2475839789188, 1145.529209780201, 1145.671400076403, 1137.8543781376034, 1149.7089215025042]
# [1165.2657601822534, 1012.1771555862939, 845.37285419784882, 717.90065905428366, 867.80437954682282, 1218.4044343766691, 1276.2638416249022, 1183.4411922972517, 1132.9909970915344, 1140.8220935959073, 1128.5249151586004, 1156.0306834502567, 1132.6575081345297, 1142.4970211247769, 1147.0279503248098, 1132.1766947530298, 1136.5330634252009, 1147.0003270295549, 1149.0488519310888, 1154.089369875471]
# [1181.5665155243328, 1022.979493591805, 845.43815685996913, 655.77441534787738, 458.22230107649204, 271.19800877757888, 102.49762815161284, -1.701767637659215, 93.795049000748577, 434.73690360152403, 886.81412228745307, 1164.2409160451484, 1264.8346736060641, 1255.7791822697739, 1229.2123007560065, 1194.4181317904799, 1161.2386245829853, 1163.7446768836551, 1137.1449992857406, 1146.5751337820429]
#
# opt
# [1193.1380007650284, 1157.372269053739, 1140.1040758972704, 1150.0171288507202, 1140.2701478105894, 1130.1882244978319, 1148.1600857895362, 1143.2618406477252, 1138.1444919601086, 1153.7934446311308, 1143.4383579970624, 1157.3847861172128, 1147.6488206140625, 1149.912254047204, 1152.3272803088582, 1154.792005889801, 1150.6367499162257, 1144.6224193401283, 1146.7582735743197, 1142.4026077277458]
# [1196.0036203928819, 1238.9844802874297, 1243.15242790403, 1249.1266517424592, 1250.8472227990642, 1262.3966758974559, 1256.6428125380419, 1259.7587418619296, 1254.9323159036019, 1265.1357040292098, 1264.0795219621318, 1250.0148062806195, 1252.9507813332093, 1269.8511445694567, 1254.2131884715118, 1243.7312640861858, 1235.2038396956398, 1214.1977743767568, 1203.442895740719, 1175.6435182012549]
# [1265.7065805465872, 1220.0915901879901, 1157.8133925629443, 1142.933140153697, 1133.9575163991447, 1128.4030811183648, 1139.7951637573756, 1134.1510982567177, 1150.843073735404, 1145.0969896372612, 1146.5830773469679, 1140.779472534168, 1150.0916297870181, 1142.7277931147401, 1144.3889455921133, 1153.9181424713163, 1139.6236331442094, 1136.6504740232494, 1138.8734370654149, 1155.8260376636033]
