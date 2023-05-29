from typing import Optional, Dict
from lark import Token
from swarmist.core.dictionary import TopologyBuilder, PosGenerationMethod, Bounds, Parameters
from swarmist.initialization import InitializationMethods, TopologyMethods

class StrategyParser:
    def __init__(self):
        self._topology:Optional[TopologyBuilder] = None
        self._init_pos_method:Optional[PosGenerationMethod] = None
        self._population_size:Optional[int] = None
        self._parameters: Parameters = Parameters()
        
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
            InitializationMethods().normal(*self._args_to_params(*args))
        )
    
    def init_random_lognormal(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().lognormal(*self._args_to_params(*args))
        )
    
    def init_random_cauchy(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().cauchy(*self._args_to_params(*args))
        )
    
    def init_random_skewnormal(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().skewnormal(*self._args_to_params(*args))
        )
    
    def init_random_levy(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().levy(*self._args_to_params(*args))
        )
    
    def init_random_beta(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().beta(*self._args_to_params(*args))
        )
    
    def init_random_exponential(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().exponential(*self._args_to_params(*args))
        )
    
    def init_random_rayleigh(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().rayleigh(*self._args_to_params(*args))
        )
    
    def init_random_weibull(self, *args)->PosGenerationMethod:
        return self.set_init_pos_method(
            InitializationMethods().weibull(*self._args_to_params(*args))
        )
    
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
        return f"Topology={self._topology}, InitPosMethod={self._init_pos_method}, PopulationSize={self._population_size}, Parameters={self._parameters}"
    