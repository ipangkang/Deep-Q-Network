import numpy as np


# 暂时就写三个 看看情况
qTable_1 = {}
qTable_2 = {}
qTable_3 = {}


class agent:

    def __init__(self, agentIndex=0, startLocationIndex=0):
        self.timeNumber = {}
        self.alpha = {}
        self.currentState = ()
        self.nextState = ()
        self.strategy = {}
        self.agentIndex = agentIndex
        self.startLocationIndex = startLocationIndex
        self.qTable = {}
        self.currentReward = 0


    def 


