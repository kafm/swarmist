from typing import Any, Dict, Union
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
    def lbest_topology(self, size: int): return self.strategy.lbest(size)
    
    def set_parameter(self, name: str, value: float, bounds: tuple = None): 
        return self.strategy.set_parameter(name, value, bounds)
        
    def set_max_evals(self, value: int): return self.terminator.max_evals(value)
    def set_max_gen(self, value: int): return self.terminator.max_gen(value)
    def set_min_fit(self, value: float): return self.terminator.min_fit(value)
        
    def loc(self, value: Any): return ("loc", self.number(value))
    def scale(self, value: Any): return ("scale", self.number(value))
    def shape(self, value: Any): return ("shape", self.number(value))
    def alpha(self, value: Any): return ("alpha", self.number(value))
    def beta(self, value: Any): return ("beta", self.number(value))

    def bounds(self, lower: float, upper: float): return (lower, upper)
    def size(self, value: Any): return self.integer(value)
            
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
