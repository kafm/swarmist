from __future__ import annotations
from swarmist.core.dictionary import (
    FitnessFunction,
    Bounds,
    ConstraintsChecker,
    ConstraintValue,
    Pos,
    Fit,
)
from swarmist.core.evaluator import Evaluator
import numpy as np


def minimize(f: FitnessFunction, bounds: Bounds, dimensions: int = 2) -> SpaceBuilder:
    return SpaceBuilder(f, bounds, dimensions, True)


def maximize(f: FitnessFunction, bounds: Bounds, dimensions: int = 2) -> SpaceBuilder:
    return SpaceBuilder(f, bounds, dimensions, False)


def ge_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    return lambda pos: min(0, np.sum(left(pos) - right(pos) ** 2))


def le_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    return ge_constraint(right, left)


def eq_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    return lambda pos: max(0, np.sum((left(pos) - right(pos) ** 2)))


class SpaceBuilder:
    def __init__(
        self,
        fitness_function: FitnessFunction,
        bounds: Bounds,
        dimensions: int,
        minimize: bool = True,
        penalty_coefficient: float = 0.75,
    ):
        self._dimensions: int = dimensions
        self._bounds: Bounds = bounds
        self._constraints: ConstraintsChecker = None
        self._fitness_function = fitness_function
        self._minimize = minimize
        self._penalty_coefficient = penalty_coefficient

    def constrained_by(self, *cs: ConstraintsChecker) -> SpaceBuilder:
        self._constraints = [c for c in cs if c is not None]
        return self

    def get(self) -> Evaluator:
        return Evaluator(
            fit_func=self._fitness_function,
            minimize=self._minimize,
            ndims=self._dimensions,
            bounds=self._bounds,
            constraints=self._constraints,
            penalty_coefficient=self._penalty_coefficient,
        )
