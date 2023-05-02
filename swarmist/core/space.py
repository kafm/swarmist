from __future__ import annotations
from typing import Callable, List
from pymonad.either import Either
from .errors import try_catch, assert_not_null, assert_at_least, assert_function_signature
from .dictionary import FitnessFunction, FitnessFunctionDef, Bounds, ConstraintsChecker, SearchSpace

def dimensions(size: int)->Callable[..., int]:
    def callback():
        param = "Dimensions length"
        assert_not_null(size, param)
        assert_at_least(size, 2, param)
        return size
    return callback

def assert_fitness_function(f: FitnessFunction):
    param = "Fitness function"
    assert_not_null(f, param)
    assert_function_signature(f, FitnessFunction, param)

def bounded(lbound: float, ubound: float)->Callable[..., Bounds]:
    def callback():
        assert_not_null(lbound, "Lower bound")
        assert_not_null(ubound, "Upper bound")
        assert isinstance(lbound, float), "Lower bound should be float"
        assert isinstance(ubound, float), "Upper bound should be float"
        return Bounds(lbound, ubound)
    return callback

def constrained_by(*cs: ConstraintsChecker)->Callable[..., ConstraintsChecker]:
    def get(c: ConstraintsChecker)->ConstraintsChecker:
        param = "Constraints checker"
        assert_not_null(c, param)
        assert_function_signature(c, ConstraintsChecker, param)
        return c
    return lambda: [get(c) for c in cs]

def minimize(f: FitnessFunction)->Callable[..., FitnessFunctionDef]:
    def callback():
        assert_fitness_function(f)
        return FitnessFunctionDef(f, True)
    return callback

def maximize(f: FitnessFunction)->Callable[..., FitnessFunctionDef]:
    def callback():
        assert_fitness_function(f)
        return FitnessFunctionDef(f, False)
    return callback

def space(
    dimensions: Callable[..., int],
    bounds: Callable[..., Bounds],
    fit_function: Callable[..., FitnessFunctionDef],
    constraints: Callable[..., List[ConstraintsChecker]] = None,
)->Either[Callable[..., SearchSpace], Exception]:
    return lambda : try_catch(
            lambda: SearchSpace(
                fit_func=fit_function(),
                ndims=dimensions(),
                bounds=bounds(),
                constraints=constraints() if constraints else None
            )
      )