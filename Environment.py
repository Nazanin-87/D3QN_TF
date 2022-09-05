import numpy as np
from numpy import pi
from random import random, uniform, choices, randint, sample
import math
from scipy import special
from scipy.io import *


class Environ():

    def __init__(self, Num, n_actions, Pmax, Noise, BW, Rmin, negative_cost):
        self.Num = Num
        self.state_dim = self.Num
        self.n_actions = n_actions
        self.Pmax=Pmax
        self.Rmin = Rmin
        self.Noise=Noise
        self.BW=BW
        self.negative_cost=negative_cost

        self.bs = complex((500 / 2), (500 / 2))
        self.QoS = np.zeros(self.Num)
        self.S = (np.zeros(self.Num)).reshape(-1)

    def Location(self):
        rx = np.zeros(self.Num)
        ry = np.zeros(self.Num)
        Loc = np.zeros(self.Num, dtype=complex)
        for i in range(self.Num):
            rx[i] = uniform(0, 500)
            ry[i] = uniform(0, 500)
            Loc[i] = complex(rx[i], ry[i])
        return Loc

    def PathGain(self, Loc):
        d = np.zeros(self.Num)
        x = np.zeros(self.n_actions)
        h = np.zeros((self.Num, self.n_actions))

        for i in range(self.Num):
            d[i] = abs(Loc[i] - self.bs)
            d[i] = d[i] ** (-3)
            for k in range(self.n_actions):
                u = np.random.rand(1, 1)
                sigma = 1
                x[k] = sigma * np.sqrt(-2 * np.log(u))
                h[i, k] = d[i] * x[k]

        return h

    def reset(self):  # Reset the states
        s = np.zeros(self.Num)
        return s.reshape(-1)

    def RecievePower(self, h):
        TotalPower = np.zeros(self.Num, dtype=float)
        P_private = np.zeros((self.Num, self.n_actions), dtype=float)
        P_common = np.zeros((self.Num, self.n_actions), dtype=float)
        UsersRecievePower_private = np.zeros((self.Num, self.n_actions), dtype=float)
        UsersRecievePower_common = np.zeros((self.Num, self.n_actions), dtype=float)

        actionPC=np.random.uniform(size=2*self.n_actions)
        actionPC=actionPC*self.Pmax

        for i in range(self.Num):
            for k in range(self.n_actions):
                for s in range(i, (i + 1)):
                    P_private[i, k] = actionPC[s]
                    P_common[i, k] = actionPC[s + self.n_actions]
                for j in range(self.Num):
                    if j != i :
                        for l in range(j, (j + 1)):
                            P_private[i, k] = actionPC[l]
                            P_common[i, k] = actionPC[l + self.n_actions]
            TotalPower[i] = sum(P_private[i, :]) + sum(P_common[i, :])

        for i in range(self.Num):
            for k in range(self.n_actions):
                UsersRecievePower_private[i, k] = h[i, k] * P_private[i, k]
                UsersRecievePower_common[i, k] = h[i, k] * P_common[i, k]

        return UsersRecievePower_private, UsersRecievePower_common, TotalPower

    def TotalRate(self, actionRB,h):
        interference_common = np.zeros((self.Num, self.n_actions), dtype=float) + self.Noise
        interference_private = np.zeros((self.Num, self.n_actions), dtype=float) + self.Noise
        SINR_common = np.zeros((self.Num, self.n_actions), dtype=float)
        SINR_private = np.zeros((self.Num, self.n_actions), dtype=float)
        Rate_common = np.zeros((self.Num, self.n_actions), dtype=float)
        Rate_private = np.zeros((self.Num, self.n_actions), dtype=float)
        TotalRate = np.zeros(self.Num, dtype=float)

        RecievePower_private, RecievePower_common, TotalPower = self.RecievePower(h)
        RB = np.zeros((self.Num, self.n_actions), dtype=float)

        for i in range(self.Num):
            for k in range(self.n_actions):
                for s in range(i, (i + 1)):
                    if k == int(actionRB[s]):
                        RB[i, k] = 1
                for j in range(self.Num):
                    if j != i :
                        for l in range(j, (j + 1)):
                            if k == int(actionRB[l]):
                                RB[i, k] = 1
        for i in range(self.Num):
            for k in range(self.n_actions):
                for j in range(self.Num):
                    if j != i and h[j, k] > h[i, k]:
                        interference_common[i, k] = interference_common[i, k] + RecievePower_common[j, k]
                        interference_private[i, k] = interference_private[i, k] + RecievePower_private[j, k]
                    else:
                        interference_common[i, k] = interference_common[i, k]
                        interference_private[i, k] = interference_private[i, k]

                SINR_common[i, k] = RecievePower_common[i, k] / interference_common[i, k]
                SINR_private[i, k] = RecievePower_private[i, k] / interference_private[i, k]

                if RB[i, k] == 1:
                    Rate_common[i, k] = self.BW * (np.log2(1 + SINR_common[i, k]))
                    Rate_private[i, k] = self.BW * (np.log2(1 + SINR_private[i, k]))
                else:
                    Rate_common[i, k] = 0
                    Rate_private[i, k] = 0

            TotalRate[i] = sum(Rate_private[i, :]) + sum(Rate_common[i, :])

        return TotalRate, TotalPower

    def computeQoS(self, actionRB, h):
        TotalRate, TotalPower = self.TotalRate(actionRB,h)
        for i in range(self.Num):
            if TotalRate[i] >= self.Rmin :
                self.QoS[i] = (1.0)
            else:
                self.QoS[i] = (0.0)
        return self.QoS

    def ComputeState(self, actionRB, h):
        self.QoS = self.computeQoS(actionRB, h)
        S = np.zeros(self.Num)
        for i in range(self.Num):
            S[i] = self.QoS[i]
        self.S = S
        return self.S.reshape(-1)

    def Reward(self, actionRB, h):
        Rate, Power = self.TotalRate(actionRB, h)
        Satisfied_Users = sum(self.QoS)
        TotalRate = 0.0
        TotalPower = 0.05
        for i in range(self.Num):
            TotalRate = TotalRate + Rate[i]
            TotalPower = TotalPower + Power[i]
        FeMBB_TotalPower_Final = {}

        if Satisfied_Users == self.Num:
            reward = TotalRate / TotalPower
            done = True
        else:
            reward = self.negative_cost
            done = False
        return reward, done

    def step(self, actionRB, h):
        next_s = self.ComputeState(actionRB, h)
        r, d = self.Reward(actionRB, h)
        done = False
        info = None
        if d == True:
            done = True
        return next_s, r, done, info





