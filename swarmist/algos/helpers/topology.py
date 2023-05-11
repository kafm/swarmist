import numpy as np
from swarmist.core.dictionary import *

def gbest()->TopologyBuilder:
    # def callback(agents: AgentList)->StaticTopology:
    #     all = [i for i in agents]
    #     return [all for _ in agents]
    # return callback
    return lambda _: None

def lbest(k:int = 2)->TopologyBuilder:
    def callback(agents: AgentList)->StaticTopology:
        n = len(agents)
        topology = []
        for i in range(n):
            neighbors = [i]
            for j in range(1, k):
                neighbors.append((i + j) % n)
                neighbors.append((i - j) % n)
            topology.append(neighbors)
        return topology
    return callback



#def  void set_bidirectional_connection(int i, int j, float weight) {connection[i][j] = connection[j][i] = weight;}
