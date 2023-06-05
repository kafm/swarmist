from lark import v_args
from typing import cast
from swarmist.core.dictionary import Agent
from swarmist.core.references import AgentMethods, SwarmMethods
from .expressions import Expressions, fetch_value

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
        return lambda ctx: swarm_methods.rand_to_best(f=fetch_value(probability, ctx))(
            ctx
        )

    def swarm_current_to_best(self, probability=None):
        return lambda ctx: swarm_methods.current_to_best(
            f=fetch_value(probability, ctx)
        )(ctx)

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

    def self_agent_pos(self):
        return lambda a: cast(Agent, a).pos

    def self_agent_best(self):
        return lambda a: cast(Agent, a).best

    def self_agent_fit(self):
        return lambda a: cast(Agent, a).fit

    def self_agent_index(self):
        return lambda a: cast(Agent, a).index

    def self_agent_ndims(self):
        return lambda a: cast(Agent, a).ndims

    def self_agent_delta(self):
        return lambda a: cast(Agent, a).delta

    def self_agent_trials(self):
        return lambda a: cast(Agent, a).trials

    def self_agent_improved(self):
        return lambda a: cast(Agent, a).improved
