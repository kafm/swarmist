from __future__ import annotations
from swarmist.core.dictionary import FitnessFunction, Bounds, ConstraintsChecker
from swarmist.core.evaluator import Evaluator
import numpy as np


def minimize(f: FitnessFunction, bounds: Bounds, dimensions: int = 2) -> SpaceBuilder:
    return SpaceBuilder(f, bounds, dimensions, True)


def maximize(f: FitnessFunction, bounds: Bounds, dimensions: int = 2) -> SpaceBuilder:
    return SpaceBuilder(f, bounds, dimensions, False)


class SpaceBuilder:
    def __init__(
        self,
        fitness_function: FitnessFunction,
        bounds: Bounds,
        dimensions: int,
        minimize: bool = True,
    ):
        self._dimensions: int = dimensions
        self._bounds: Bounds = bounds
        self._constraints: ConstraintsChecker = None
        self._fitness_function = fitness_function
        self._minimize = minimize

    def constrained_by(self, *cs: ConstraintsChecker) -> SpaceBuilder:
        self._constraints = list(cs)
        return self

    def get(self) -> Evaluator:
        return Evaluator(
            fit_func=self._fitness_function,
            minimize=self._minimize,
            ndims=self._dimensions,
            bounds=self._bounds,
            constraints=self._constraints,
        )
