from typing import Any, Optional
from lark import Lark, Transformer, v_args, Token
from functools import reduce
import math
import numpy as np
from .termination_parser import TerminatonParser
from .strategy_parser import StrategyParser
from .grammar import grammar

@v_args(inline=True) 
class GrammarTransformer(Transformer):

    def __init__(self):
        self.vars = {}
        self.terminator:TerminatonParser = TerminatonParser()
        self.strategy:StrategyParser = StrategyParser()
    
    def update_op(self, *args):
        print(f"Im here in update_op with {str(args)}")
        return f"update {str(args)}"
       
    def ctx_get_param(self, name): 
        return name
    
    def ctx_get_var(self, name):
        return name
    
    def ctx_if_then(self, condition, then, otherwise):
        callback = lambda ctx: then(ctx) if condition(ctx) else otherwise(ctx)
        return callback
    
    def ctx_sum(self, expr):
        return self.strategy.ctx_sum(expr)
    
    def ctx_avg(self, expr):
        return self.strategy.ctx_avg(expr)
    
    def ctx_min(self, expr):
        return self.strategy.ctx_min(expr)
    
    def ctx_max(self, expr):
        return self.strategy.ctx_max(expr)
    
    def reduce_op(self, *args):
        return self.strategy.ctx_reduce(*args)
    
    def ctx_random(self):
        return self.strategy.ctx_random()
    
    def ctx_random_uniform(self, *args):
        return self.strategy.ctx_random_uniform(*args)
    
    def ctx_random_normal(self, *args):
        return self.strategy.ctx_random_normal(*args)
    
    def ctx_random_lognormal(self, *args):
        return self.strategy.ctx_random_lognormal(*args)
    
    def ctx_random_skewnormal(self, *args):
        return self.strategy.ctx_random_skewnormal(*args)
    
    def ctx_random_cauchy(self, *args):
        return self.strategy.ctx_random_cauchy(*args)
    
    def ctx_random_levy(self, *args):
        return self.strategy.ctx.random_levy(*args)
    
    def ctx_random_beta(self, *args):
        return self.strategy.ctx_random_beta(*args)
    
    def ctx_random_exponential(self, *args):
        return self.strategy.ctx_random_exponential(*args)
    
    def ctx_random_rayleigh(self, *args):
        return self.strategy.ctx_random_rayleigh(*args)
    
    def ctx_swarm_best(self, size=None): 
        return self.strategy.ctx_swarm_best(size)
    
    def ctx_swarm_worst(self, size=None): 
        return self.strategy.ctx_swarm_worse(size)
    
    def ctx_swarm_neighborhood(self): 
        return self.strategy.ctx_swarm_neighborhood()
    
    def ctx_pick_random(self, unique=False, n=None, replacement=False): 
        return self.strategy.ctx_pick_random(unique, n, replacement)
    
    def ctx_pick_roulette(self, unique=False, n=None, replacement=False):
        return self.strategy.ctx_pick_roulette(unique, n, replacement)
    
    def ctx_rand_to_best(self, p=None):
        return self.strategy.ctx_rand_to_best(p)
    
    def ctx_current_to_best(self, p=None):
        return self.strategy.ctx_current_to_best(p)

    
    def binomial_recombination(self, p): return self.strategy.binomial_recombination(p)
    def exponential_recombination(self, p): return self.strategy.exponential_recombination(p)
    def with_probability_recombination(self, p): return self.strategy.with_probability_recombination(p)
    def random_recombination(self, size): return self.strategy.random_recombination(size)
    def init_random_recombination(self): return self.strategy.init_random_recombination()
        
    def selection(self, method=None, where=None, order=None, limit=None):
        return self.strategy.selection(
            method=method, 
            where=where, 
            order=order, 
            size=limit
        )
    
    def roulette_selection(self): return ("roulette")
    def random_selection(self): return ("random")
    def probabilistic_selection(self, probability): return ("probabilistic", probability)
        
    def order_by(self, field, order = None): return self.strategy.order_by(field, order)
    def asc(self): return "asc"
    def desc(self): return "desc"
    def limit(self, value): return self.integer(value)
    
    def agent_index(self): return self.strategy.agent_index()
    def agent_trials(self): return self.strategy.agent_trials()
    def agent_best(self): return self.strategy.agent_best()
    def agent_pos(self): return self.strategy.agent_pos()
    def agent_fit(self): return self.strategy.agent_fit()
    def agent_delta(self): return self.strategy.agent_delta()
    def agent_improved(self): return self.strategy.agent_improved()
    def ctx_agent_index(self): return self.strategy.ctx_agent_index()
    def ctx_agent_trials(self): return self.strategy.ctx_agent_trials()
    def ctx_agent_best(self): return self.strategy.ctx_agent_best()
    def ctx_agent_pos(self): return self.strategy.ctx_agent_pos()
    def ctx_agent_fit(self): return self.strategy.ctx_agent_fit()
    def ctx_agent_delta(self): return self.strategy.ctx_agent_delta()
    def ctx_agent_improved(self): return self.strategy.ctx_agent_improved()
        
    def population_size(self, value): return self.strategy.population_size(value)
    def init_random(self): return self.strategy.init_random()
    def init_random_uniform(self): return self.strategy.init_random_uniform()
    def init_random_normal(self, *args): return self.strategy.init_random_normal(*args)
    def init_random_lognormal(self, *args): return self.strategy.init_random_lognormal(*args)
    def init_random_cauchy(self, *args): return self.strategy.init_random_cauchy(*args)
    def init_random_skewnormal(self, *args): return self.strategy.init_random_skewnormal(*args)
    def init_random_levy(self, *args): return self.strategy.init_random_levy(*args)
    def init_random_beta(self, *args): return self.strategy.init_random_beta(*args)
    def init_random_exponential(self, *args): return self.strategy.init_random_exponential(*args)
    def init_random_rayleigh(self, *args): return self.strategy.init_random_rayleigh(*args)
    def init_random_weibull(self, *args): return self.strategy.init_random_weibull(*args)
    
    def gbest_topology(self): return self.strategy.gbest()
    def lbest_topology(self, size): return self.strategy.lbest(size)
    
    def set_parameter(self, name, value, bounds = None): 
        return self.strategy.set_parameter(name, value, bounds)
        
    def set_max_evals(self, value): return self.terminator.max_evals(value)
    def set_max_gen(self, value): return self.terminator.max_gen(value)
    def set_min_fit(self, value): return self.terminator.min_fit(value)
        
    def loc(self): return "loc"
    def scale(self): return "scale"
    def shape(self): "shape"
    def alpha(self): return "alpha"
    def beta(self): return "beta"
    def low(self): return "low"
    def high(self): return "high"

    def bounds(self, lower, upper): return (lower, upper)
    def size(self, value: Any): return self.integer(value)
         
    def and_op(self, left, right): return lambda: left() and right()
    def or_op(self, left, right): return lambda: left() or right()
    def comp_op(self, left, op, right): return lambda ctx: op(left(ctx), right(ctx))  

    def lt(self): return lambda left, right: left < right
    def le(self): return lambda left, right: left <= right
    def gt(self): return lambda left, right: left > right
    def ge(self): return lambda left, right: left >= right
    def eq(self): return lambda left, right: left == right
    def ne(self): return lambda left, right: left != right
    
  
    def add(self, x, y): return lambda: x() + y()
    def sub(self, x, y): return lambda: x() - y()
    def mul(self, x, y): return lambda: x() * y()
    def div(self, x, y): return lambda: x() / y()
    def floordiv(self, x, y): return lambda: x() // y()
    def neg(self, x): return lambda: -x()
    def pow(self, x, y): return lambda: x() ** y()
    def mod(self, x, y): return lambda: x() % y()
    def sin(self, x): return lambda: math.sin(x())
    def cos(self, x): return lambda: math.cos(x())
    def tan(self, x): return lambda: math.tan(x()) 
    def arcsin(self, x): return lambda: math.asin(x())
    def arccos(self, x): return lambda: math.acos(x())
    def arctan(self, x): return lambda: math.atan(x())
    def sqrt(self, x): return lambda: math.sqrt(x())
    def log(self, x): return lambda: math.log(x())
    def exp(self, x): return lambda: math.exp(x())
    def abs(self, x): return lambda: abs(x())
    def norm(self, x): return lambda: np.linalg.norm(x())
    def sum(self, x): return lambda: sum(x())
    def min(self, x): return lambda: min(x())
    def max(self, x): return lambda: max(x())
    def avg(self, x): return lambda: sum(x())/len(x()) if hasattr(x, '__len__') else x
    def number_lambda(self, x): return lambda: self.number(x)
    def pi(self): return lambda: math.pi
    

    def probability(self, value):
        if value < 0 or value > 1:
           raise ValueError(f"Probability must be between 0 and 1")
        return lambda: value
    
    def assign_var(self, name, value):
        self.vars[name] = value
        return value
   
    def var(self, name):
        try:
            return self.vars[name]
        except KeyError:
            raise Exception(f"Variable not found: {name}")
        
    def var_index(self, name, index):
        value = self.var(name)
        if value is list:
            try:
                return value[index]  
            except KeyError:  
                raise IndexError(f"Index {index} out of bounds for variable {name}")
        else:
            raise ValueError(f"Variable {name} is not a list")
    
    def number(self, value: Any): return float(value)
    def integer(self, value: Any): return int(value)  
    def string(self, value: Any): return str(value)
    def true(self): return True
    def false(self): return False
    
    def __repr__(self):
        return f"<Parser: {self.vars}, {self.strategy}>"
    
def parse(expression, grammar: str = grammar):
    transformer = GrammarTransformer()
    lexer = Lark(grammar, parser="lalr", transformer=transformer)
    result = lexer.parse(expression)
    print(f"Expression: {expression}")
    print(transformer)
    print(f"Result: {result}")
    return result
    #return transformer

transformer = GrammarTransformer()
lexer = Lark(grammar, parser="lalr", transformer=transformer)

def evaluate_expression(expression):
    tree = lexer.parse(expression)
    return tree
