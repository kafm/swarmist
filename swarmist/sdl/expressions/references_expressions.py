from lark import v_args
from swarmist.core.references import AgentMethods, SwarmMethods
from .expressions import Expressions

swarm_methods = SwarmMethods()
agent_methods = AgentMethods()

@v_args(inline=True)
class ReferencesExpressions(Expressions):

    def swarm_best(self, size=None):
        return swarm_methods.best(size)

    def swarm_worst(self, size=None):
        return swarm_methods.worse(size)

    def swarm_all(self):
        return swarm_methods.all()

    def swarm_neighborhood(self):
        return swarm_methods.neighborhood()

    def swarm_pick_random(self, unique=None, size=None, replace=None):
        return swarm_methods.pick_random(unique=unique, size=size, replace=replace)

    def swarm_pick_roulette(self, unique=None, size=None, replace=None):
        return swarm_methods.pick_roulette(unique=unique, size=size, replace=replace)

    def swarm_rand_to_best(self, probability=None):
        return swarm_methods.rand_to_best(probability=probability)

    def swarm_current_to_best(self, probability=None):
        return swarm_methods.current_to_best(probability=probability)

    def agent_pos(self):
        return agent_methods.pos()

    def agent_best(self):
        return agent_methods.best()

    def agent_fit(self):
        return agent_methods.fit()

    def agent_index(self):
        return agent_methods.index()

    def agent_ndims(self):
        return agent_methods.ndims()

    def agent_delta(self):
        return agent_methods.delta()

    def agent_trials(self):
        return agent_methods.trials()

    def agent_improved(self):
        return agent_methods.improved()

