from lark import v_args
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, replace
import numpy as np
from .expressions import Expressions, fetch_value
from swarmist.core.dictionary import Bounds, Pos
from swarmist.space import SpaceBuilder, lt_constraint, le_constraint, gt_constraint, ge_constraint, eq_constraint, ne_constraint


@dataclass(frozen=True)
class Variable:
    name: str
    bounds: Bounds
    begin_index: int = 0
    size: Optional[int] = None


@dataclass(frozen=True)
class Variables:
    kv: Dict[str, Variable]
    bounds: Bounds = None
    ndims: int = None


@dataclass(frozen=True)
class SpaceAssets:
    space: SpaceBuilder
    get_var: Callable[[Pos, str], Any]


@v_args(inline=True)
class SpaceExpressions(Expressions):
    def space(self, variables: Variables, objective_function, constraints=None):
        def get_var(pos, name):
            var: Variable = variables.kv[name]
            return (
                pos[var.begin_index]
                if var.size == 1
                else pos[var.begin_index : var.begin_index + var.size]
            )
        #TODO constraints
        return SpaceAssets(
            space=SpaceBuilder(
                objective_function[0], bounds=variables.bounds, dimensions=variables.ndims, 
                minimize=objective_function[1]
            ).constrained_by(*constraints),
            get_var=get_var,
        )
    
    def build_constraints(self, constraints, coefficient=None):
        print(list(constraints), coefficient)
        return None
    
    def set_constraints(self, *constraints):
        return list(constraints)
    
    def lt_constraint(self, left, right):
        return lt_constraint(left, right)

    def le_constraint(self, left, right):
        return le_constraint(left, right)

    def gt_constraint(self, left, right):
        return gt_constraint(left, right)

    def ge_constraint(self, left, right):
        return ge_constraint(left, right)

    def eq_constraint(self, left, right):
        return eq_constraint(left, right)

    def ne_constraint(self, left, right):
        return ne_constraint(left, right)

    def minimize(self, expr):
        return (self.get_fit_function(expr), False)

    def maximize(self, expr):
        return (self.get_fit_function(expr), True)
    
    def get_fit_function(self, expr):
        return lambda pos: expr(pos)
    
    

    def set_vars(self, *args: Variable):
        variables = {}
        begin_index = 0
        lbound = np.array([])
        ubound = np.array([])
        ndims = 0
        for var in args:
            variables[var.name] = replace(var, begin_index=begin_index)
            lbound = np.concatenate((lbound, np.full(var.size, var.bounds.min)))
            ubound = np.concatenate((ubound, np.full(var.size, var.bounds.max)))
            ndims += var.size
            begin_index += var.size
        return Variables(
            kv=variables,
            bounds=Bounds(lbound, ubound), ndims=ndims
        )

    def var(self, *args):
        return Variable(
            name=args[0],
            size=1 if len(args) == 2 else args[1],
            bounds=args[-1],
        )
