from swarmist.core.references import SwarmMethods, AgentMethods
from swarmist.strategy import Strategy
from swarmist.initialization import  InitializationMethods, TopologyMethods
from swarmist.recombination import RecombinationMethods
from swarmist.update import *
from swarmist.space import minimize, maximize
from swarmist.search import until, using, search


strategy = lambda: Strategy()
init = InitializationMethods()
topology = TopologyMethods()
swarm = SwarmMethods()
agent = AgentMethods()
recombination = RecombinationMethods()