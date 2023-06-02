from swarmist.core.dictionary import Bounds, FitnessFunction
from swarmist.core.references import SwarmMethods, AgentMethods
from swarmist.strategy import Strategy
from swarmist.initialization import  InitializationMethods, TopologyMethods
from swarmist.recombination import RecombinationMethods
from swarmist.update import *
from swarmist.space import minimize, maximize, le_constraint, lt_constraint, ge_constraint, gt_constraint, eq_constraint, ne_constraint
from swarmist.search import until, using, search
from swarmist.utils import benchmark


strategy = lambda: Strategy()
init = InitializationMethods()
topology = TopologyMethods()
swarm = SwarmMethods()
agent = AgentMethods()
recombination = RecombinationMethods()