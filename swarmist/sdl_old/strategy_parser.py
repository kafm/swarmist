from typing import Optional, Dict, Callable, Union
import inspect
from swarmist.core.dictionary import TopologyBuilder, PosGenerationMethod, Bounds, Parameters, Selection, UpdateContext, Agent, Pos, Fit, IReference, IReferences
from swarmist.core.references import SwarmMethods, AgentMethods
from swarmist.initialization import InitializationMethods, TopologyMethods
from swarmist.update import all, quantity, roulette, with_probability, random, filter
from swarmist.recombination import RecombinationMethods
from swarmist.core.random import Random

class StrategyParser:
    def __init__(self):
        self._topology:Optional[TopologyBuilder] = None
        self._init_pos_method:Optional[PosGenerationMethod] = None
        self._population_size:Optional[int] = None
        self._parameters: Parameters = Parameters()
        self._ndims: int = 1
        self.ctx_random_gen:Random = Random(self._ndims)
        
    def binomial_recombination(self, p: float): return RecombinationMethods().binomial(p)
    def exponential_recombination(self, p: float): return RecombinationMethods().exponential(p)
    def with_probability_recombination(self, p: float): return RecombinationMethods().k_with_probability(p)
    def random_recombination(self, size: int ): return RecombinationMethods().k_random(size)
    def init_random_recombination(self): return RecombinationMethods().get_new()
    
    def ndims(self, size: int):
        self.ctx_random: Random = Random(size)
           
    def selection(self, 
        method: Optional[tuple]=None, 
        where: Optional[Callable[[Agent], bool]]=None, 
        order: Optional[Callable[[Agent], bool]]=None, 
        size: Optional[int] = None)->Selection:
        method: self._get_selection_method(method, where, size)
        if order:
            method = lambda agents: order(method(agents))
        return method
     
    def _get_selection_method(self, 
            method:Optional[tuple] = None,
            where: Optional[Callable[[Agent], bool]]=None,
            size: Optional[int] = None)->Callable[[Optional[int]], Selection]:   
        type = None if not method else method[0]
        if not type: 
            if where:
                return lambda: filter(where, size)
            return lambda: all() if not size else quantity(size)
        else: 
            self.assert_no_where(where)
            if type == "roulette": return lambda: roulette(size)
            elif type == "random": return lambda: random(size)
            elif type == "probabilistic":
                if len(method) < 2:
                    raise Exception("Probabilistic selection method must have a probability")
                return lambda: with_probability(p=method[1],size=size)
            else:
                raise TypeError(f"Unknown selection method: {type}")
                
        
    def assert_no_where(self, where: Optional[Callable[[Agent], bool]]):
        if where:
            raise Exception("Where condition is not allowed when using a roulette, random or probabilistic selection")
        
    def assert_probability_selection(self, probability: Optional[float]):
        if not probability:
            raise Exception("Probabilistic selection method must have a probability")
        
    def order_by(self, field, order): return lambda agents: sorted(agents, key=field, reverse=order == "desc")
    
    def ctx_sum(self, expr):
        print(f"SUM: {expr}")
        return lambda _=None:0
    
    def ctx_avg(self, expr):
        print(f"AVG: {expr}")
        return lambda _=None:0
    
    def ctx_min(self, expr):
        print(f"MIN: {expr}")
        return lambda _=None:0
    
    def ctx_max(self, expr):
        print(f"MAX: {expr}")
        return lambda _=None:0
    
    def reduce_op(self, expr):
        print(f"REDUCE: {expr}")
        return lambda _=None:0
    
    def ctx_swarm_best(self, size: Optional[int] = None)->Callable[[UpdateContext], Union[IReference, IReferences]]:
        if not size or size == 1:
            return SwarmMethods().best()
        return SwarmMethods().k_best(size)
    
    def ctx_swarm_worse(self, size: Optional[int] = None)->Callable[[UpdateContext], Union[IReference, IReferences]]:
        if not size or size == 1:
            return SwarmMethods().best()
        return SwarmMethods().k_best(size)
            
    def ctx_swarm_worse(self, size: Optional[int] = None)->Callable[[UpdateContext], Union[IReference, IReferences]]:
        if not size or size == 1:
            return SwarmMethods().worse()
        return SwarmMethods().k_worse(size)
    
    def ctx_swarm_neighborhood(self)->Callable[[UpdateContext], IReferences]:
        return SwarmMethods().neighborhood()
    
    def ctx_pick_random(self, unique: Optional[bool] = False, size: Optional[int]= None, replace:Optional[bool]=False)->Callable[[UpdateContext], Union[IReference, IReferences]]:
        return SwarmMethods().pick_random(size=size, replace=replace, unique=unique)
    
    def ctx_pick_roulette(self, unique: Optional[bool] = False, size: Optional[int]= None, replace:Optional[bool]=False)->Callable[[UpdateContext], Union[IReference, IReferences]]:
        return SwarmMethods().pick_roulette(size=size, replace=replace, unique=unique)
    
    def ctx_rand_to_best(self, p:Optional[float]=None)->Callable[[UpdateContext], Pos]:
        return SwarmMethods().rand_to_best(p)
    
    def ctx_current_to_best(self, p:Optional[float]=None)->Callable[[UpdateContext], Pos]:
        return SwarmMethods().current_to_best(p)
    
    def ctx_random(self):
        return lambda _: self.ctx_random_gen.rand()
    
    def ctx_random_uniform(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.uniform(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_normal(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.normal(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_lognormal(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.lognormal(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_skewnormal(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.skewnormal(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_cauchy(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.cauchy(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_levy(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.levy(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_beta(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.beta(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_exponential(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.exponential(self.exec_distribution_params(*params, ctx))
    
    def ctx_random_rayleigh(self, *args):
        params = self.zip_distribution_params(*args)
        return lambda ctx: self.ctx_random_gen.rayleigh(self.exec_distribution_params(*params, ctx))
    
    def ctx_agent_index(self)->Callable[[UpdateContext], int]: return lambda ctx: ctx.agent.index
    def ctx_agent_trials(self)->Callable[[UpdateContext], int]: return lambda ctx: ctx.agent.trials
    def ctx_agent_best(self)->Callable[[UpdateContext], Pos]: return lambda ctx: ctx.agent.best
    def ctx_agent_pos(self)->Callable[[UpdateContext], Pos]: return lambda ctx: ctx.agent.pos
    def ctx_agent_fit(self)->Callable[[UpdateContext], Fit]: return lambda ctx: ctx.agent.fit
    def ctx_agent_delta(self)->Callable[[UpdateContext], Pos]: return lambda ctx: ctx.agent.delta
    def ctx_agent_improved(self)->Callable[[UpdateContext], bool]: return lambda ctx: ctx.agent.improved
    def agent_index(self)->Callable[[Agent], int]: return lambda a: a.index
    def agent_trials(self)->Callable[[Agent], int]: return lambda a: a.trials
    def agent_best(self)->Callable[[Agent], Pos]: return lambda a: a.best
    def agent_pos(self)->Callable[[Agent], Pos]: return lambda a: a.pos
    def agent_fit(self)->Callable[[Agent], Fit]: return lambda a: a.fit
    def agent_delta(self)->Callable[[Agent], Pos]: return lambda a: a.delta
    def agent_improved(self)->Callable[[Agent], bool]: return lambda a: a.improved
        
    def population_size(self, size: int)->int:
        self._population_size = size
        return size
        
    def init_random(self)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().random()
        )
        
    def init_random_uniform(self)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().uniform()
        )
    
    def init_random_normal(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().normal(self.zip_distribution_params(*args))
        )
    
    def init_random_lognormal(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().lognormal(self.zip_distribution_params(*args))
        )
    
    def init_random_cauchy(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().cauchy(self.zip_distribution_params(*args))
        )
    
    def init_random_skewnormal(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().skewnormal(self.zip_distribution_params(*args))
        )
    
    def init_random_levy(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().levy(self.zip_distribution_params(*args))
        )
    
    def init_random_beta(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().beta(self.zip_distribution_params(*args))
        )
    
    def init_random_exponential(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().exponential(self.zip_distribution_params(*args))
        )
    
    def init_random_rayleigh(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().rayleigh(self.zip_distribution_params(*args))
        )
    
    def init_random_weibull(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().weibull(self.zip_distribution_params(*args))
        )
        
    def zip_distribution_params(self, *args)->Dict[str, float]:
        return {args[i]: args[i+1] for i in range(0, len(args), 2)}
    
    def exec_distribution_params(self, *args, ctx: UpdateContext = None)->Dict[str, float]:        
        def _exec(arg):
            if callable(arg): return arg(ctx) if ctx else arg()
            return arg
        return {key: _exec(value) for key, value in args}

        
    
    def set_init_pos_method(self, method: PosGenerationMethod):
        self._init_pos_method = method
        return self._init_pos_method
    
    def set_parameter(self, name: str, value: float, bounds: tuple = None):
        lbound, ubound = bounds if bounds else (None, None)
        return self._parameters.add(name=name, value=value, min=lbound, max=ubound)
        
    def gbest(self)->TopologyBuilder: return self.set_topology(TopologyMethods().gbest())
    def lbest(self, size: int)->TopologyBuilder: return self.set_topology(TopologyMethods().lbest(size))
    def set_topology(self, topology: TopologyBuilder):
        self._topology = topology
        return self._topology
    
    def bounds(self, lower, upper):
        return Bounds(lower, upper)
    
    def get(self):
        raise NotImplementedError()
    
    def _args_to_params(self, *args)->Dict:
        params:Dict = {arg[0]: arg[1] for arg in args}
        return params
    
    def __repr__(self):
        return f"<StrategyParser: {self._topology}, {self._init_pos_method}, {self._population_size}, {self._parameters}>"
    
    def __str__(self):
        return f"""
            Topology={self._topology}, 
            InitPosMethod={self._init_pos_method}, 
            PopulationSize={self._population_size}, 
            Parameters={self._parameters}
        """
    