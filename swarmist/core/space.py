from __future__ import annotations
from typing import Callable, List, Optional
from pymonad.either import Either, Right
from .errors import try_catch, assert_not_null, assert_at_least, assert_callable, assert_number
from .dictionary import FitnessFunction, FitnessFunctionDef, Bounds, ConstraintsChecker, SearchSpace

def dimensions(size: int)->Callable[..., int]:
    def callback()->int:
        param = "Dimensions length"
        assert_not_null(size, param)
        assert_at_least(size, 2, param)
        return size
    return callback

def bounded(lbound: float, ubound: float)->Callable[..., Bounds]:
    def callback()->Bounds:
        lparam = "Lower bound"
        uparam = "Upper bound"
        assert_not_null(lbound, lparam)
        assert_not_null(ubound, uparam)
        assert_number(lbound, lparam)
        assert_number(ubound, lparam)
        return Bounds(lbound, ubound)
    return callback

def constrained_by(*cs: ConstraintsChecker)->Callable[..., ConstraintsChecker]:
    def compose_constraints(constraints: List[ConstraintsChecker], f: Optional[ConstraintsChecker] = None)->ConstraintsChecker:
        if len(constraints) > 0:
            c = constraints.pop()
            param = "Constraints checker"
            assert_not_null(c, param)
            assert_callable(c, param)
            if f: 
                c = lambda a: f(a) and c(a)
            return compose_constraints(constraints, c)
    if not cs:
        return lambda: None
    return lambda: compose_constraints([c for c in cs if c])

def fitness_function_def(f: FitnessFunction, minimize: bool = True)-> Callable[..., FitnessFunctionDef]:
    def callback()->FitnessFunctionDef:
        param = "Fitness function"
        assert_not_null(f, param)
        assert_callable(f, param)
        return FitnessFunctionDef(func=f, minimize=minimize)
    return callback

def minimize(f: FitnessFunction)->Callable[..., FitnessFunctionDef]:
    return fitness_function_def(f, True)

def maximize(f: FitnessFunction)->Callable[..., FitnessFunctionDef]:
    return fitness_function_def(f, False)

def space(
    dimensions: Callable[..., int],
    bounds: Callable[..., Bounds],
    fit_function: Callable[..., FitnessFunctionDef],
    constraints: Callable[..., List[ConstraintsChecker]] = None,
)->Either[Callable[..., SearchSpace], Exception]:  
    return lambda : try_catch(
            lambda: SearchSpace(
                fit_func_def=fit_function(),
                ndims=dimensions(),
                bounds=bounds(),
                constraints=constraints() if constraints else None
            )
        )