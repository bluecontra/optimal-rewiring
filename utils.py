import math
import random
from scipy import stats

def calculateIndex(a, b, c, sight):
    z = -c/sight + (a+b)/2
    if z > a:
        z = b - math.sqrt(2*c/sight*(b-a))
    return z

# calculate lambda against baseline
def calculateLambdaIndex(a, b, sight, y, yy):
    lam = 0

    if y >= b:
        lam = (a + b)/2 - y
    if y <= a:
        if yy <= a:
            lam = yy - y
        if yy >= b:
            lam = (a + b) / 2 - y
        if yy > a and yy < b:
            yy_ = (yy*yy - a*a) / (2*(b - a)) + yy*(b-yy) / (b - a)
            lam = yy_ - y
    if y > a and y < b:
        # if yy <= a:
        #     lam = (y*y - a*a)/2 + yy*(b - c) - y - c / sight
        if yy >= b:
            lam = (a + b) / 2 - y
        if yy > a and yy < b:
            yy_ = (yy * yy - a * a) / (2*(b - a)) + yy * (b - yy) / (b - a)
            lam = (y*y - a*a) / (2*(b - a)) - yy_*(b - y) / (b - a) - y

    return lam

# new
# calculate lambda against max value instead of baseline
def calculateLambdaIndexMax(a, b, sight, y):
    lam = 0

    if y >= b:
        lam = 0
        # print('y > b')
    if y <= a:
        lam = (a + b) / 2 - y
    if y > a and y < b:
        lam = (y*(y - a) + (b*b - y*y)/2) / (b-a)
    return lam

# calculate lambda against max value instead of baseline
# in Beta distribution
def calculateLambdaIndexMaxInBeta(a, b, sight, y, p):
    lam = 0
    a = int(a)
    b = int(b)
    B = calculateBetaFunction(a, b)
    part1 = integrate(pow(x, a - 1) * pow(1 - x, b - 1), (x, 0, y))
    part2 = integrate(pow(x, a) * pow(1 - x, b - 1), (x, y, 1))
    lam = p * (part1 * y + part2) / B - y
    return lam

def calculateLambdaIndexNotBreak(a, b, sight, y):
    lam = 0

    if y >= b:
        lam = (a + b)/2 - y
    if y <= a:
        lam = 0
    if y > a and y < b:
        lam = ((y * y - a * a) / 2 + (b - y) * y ) / (b - a) - y

    return lam

def calculateLambdaIndexInBetaNotBreak(a, b, sight, y, p):
    lam = 0
    a = int(a)
    b = int(b)
    B = calculateBetaFunction(a, b)
    part1 = integrate(pow(x, a) * pow(1 - x, b - 1), (x, 0, y))
    part2 = integrate(pow(x, a - 1) * pow(1 - x, b - 1), (x, y, 1))
    lam = p * (part1 + y * part2) / B - y

    return lam

from sympy import integrate
from sympy.abc import x

def calculateBetaFunction(a,b):
    z = integrate(pow(x, a - 1) * pow(1 - x, b - 1), (x, 0, 1))
    # print(z)
    return z


def calculateNewBaselineInBeta(a, b, yy, B, p):
    y_ = 0

    part1 = integrate(pow(x, a) * pow(1 - x, b - 1), (x, 0, yy))
    part2 = integrate(pow(x, a - 1) * pow(1 - x, b - 1), (x, yy, 1))
    y_ = p * (part1 + yy * part2) / B

    return y_

def calculateLambdaIndexInBeta(a, b, sight, y, yy, p):
    lam = 0
    a = int(a)
    b = int(b)
    B = calculateBetaFunction(a, b)
    part1 = integrate(pow(x, a) * pow(1 - x, b - 1), (x, 0, y))
    part2 = integrate(pow(x, a - 1) * pow(1 - x, b - 1), (x, y, 1))
    y_ = calculateNewBaselineInBeta(a, b, yy, B, p)
    lam = p * (part1 + y_ * part2) / B - y

    return lam

def initSingleGame(distribution_type):
    a1, b1, a2, b2 = 0,0,0,0

    if distribution_type == 0:
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

    elif distribution_type == 1:
        a1 = random.randint(1, 10)
        b1 = random.randint(1, 10)
        a2 = random.randint(1, 10)
        b2 = random.randint(1, 10)

    # alpha in (-0.2, 0.0)
    al = -random.uniform(0, 0.2)
    al2 = -random.uniform(0, 0.2)

    return a1, b1, a2, b2, al, al2

def sampleTrueRewardFromDistribution(distribution_type, a1, b1, a2, b2):
    a, b = 0, 0
    if distribution_type == 0:
        a = random.uniform(a1, b1)
        b = random.uniform(a2, b2)
    elif distribution_type == 1:
        a = stats.beta.rvs(a1, b1)
        b = stats.beta.rvs(a2, b2)

    return a, b

# calculate the best response when player1 facing player2, game on edge (i,j)
# p is for the p1's estimation over p2's policy
def calculateBestResponse(g, p):
    if p * g[0, 0] + (1 - p) * g[0, 1] >= p * g[1, 0] + (1 - p) * g[1, 1]:
        br = 0
    else:
        br = 1
    return br

def calculateJALResponse(q, p, epsilon):
    # EV values
    EV_a = q[0, 0] * p + q[0, 1] * (1 - p)
    EV_b = q[0, 2] * p + q[0, 3] * (1 - p)

    # pick action
    ran = random.uniform(0, 1)
    if ran <= epsilon:
        action = random.randint(0, 1)
    else:
        action = 0 if EV_a >= EV_b else 1

    return action

def calculateJAWoLFResponse(epsilon, policy):
    ran = random.uniform(0, 1)
    if ran <= epsilon:
        action = random.randint(0, 1)
    else:
        ran = random.uniform(0, 1)
        action = 0 if ran <= policy else 1

    return action

def updatePolcyFromFrequecy(p_old, a):
    # base_round = 100
    # total_round = base_round + round
    # if a == 0:
    #     p_new = p_old * total_round / (total_round + 1) + 1 / (total_round + 1)
    # else:
    #     p_new = p_old * total_round / (total_round + 1)
    if a == 0:
        p_new = p_old * 99 / 100 + 1 / 100
    else:
        p_new = p_old * 99 / 100
    return p_new

