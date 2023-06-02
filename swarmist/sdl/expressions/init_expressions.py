from lark import v_args
from typing import Optional, cast, List, Dict, Any
from .expressions import Expressions
from swarmist.core.dictionary import TopologyBuilder, PosGenerationMethod
from swarmist import InitializationMethods, TopologyMethods

init_methods = InitializationMethods()
@v_args(inline=True)
class InitExpressions(Expressions):

    def init_random(self, props: Dict[str, Any] = {}):
        return init_methods.random(**props)

    def init_random_uniform(self, props: Dict[str, Any] = {}):
        return init_methods.uniform(**props)

    def init_random_normal(self, props: Dict[str, Any] = {}):
        return init_methods.normal(**props)

    def init_random_lognormal(self, props: Dict[str, Any] = {}):
        return init_methods.lognormal(**props)

    def init_random_skewnormal(self, props: Dict[str, Any] = {}):
        return init_methods.skewnormal(**props)

    def init_random_cauchy(self, props: Dict[str, Any] = {}):
        return init_methods.cauchy(**props)

    def init_random_levy(self, props: Dict[str, Any] = {}):
        return init_methods.levy(**props)

    def init_random_beta(self, props: Dict[str, Any] = {}):
        return init_methods.beta(**props)

    def init_random_exponential(self, props: Dict[str, Any] = {}):
        return init_methods.exponential(**props)

    def init_random_rayleigh(self, props: Dict[str, Any] = {}):
        return init_methods.rayleigh(**props)

    def init_random_weibull(self, props: Dict[str, Any] = {}):
        return init_methods.weibull(**props)

    def gbest_topology(self)->TopologyBuilder: 
        return TopologyMethods().gbest()
    
    def lbest_topology(self, size: int)->TopologyBuilder: 
        return TopologyMethods().lbest(size)
       