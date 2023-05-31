from lark import v_args
from collections import OrderedDict
from swarmist.update import PosHelper
from swarmist.recombination import RecombinationMethods
from .expressions import Expressions


@v_args(inline=True)
class UpdateExpressions(Expressions):

    def binomial_recombination(self, p: float):
        return RecombinationMethods().binomial(p)

    def exponential_recombination(self, p: float):
        return RecombinationMethods().exponential(p)

    def with_probability_recombination(self, p: float):
        return RecombinationMethods().k_with_probability(p)

    def random_recombination(self, size: int):
        return RecombinationMethods().k_random(size)

    def init_random_recombination(self):
        return RecombinationMethods().get_new()
    
    def update_pos(self, *args):
        return OrderedDict({arg[0] : arg[1] for arg in args})
        
    def set_update_var(self, key, value):
        return (key, value)
