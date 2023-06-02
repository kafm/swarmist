from __future__ import annotations
from typing import Union
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


def lt_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x < y else x + y

    return callback


def le_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x <= y else x + y

    return callback


def gt_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x > y else x - y

    return callback


def ge_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x >= y else x - y

    return callback


def eq_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x == y else x

    return callback


def ne_constraint(left: ConstraintValue, right: ConstraintValue) -> ConstraintsChecker:
    def callback(pos: Pos) -> Fit:
        x = get_constraint_value(left, pos)
        y = get_constraint_value(right, pos)
        return 0 if x != y else x

    return callback


def get_constraint_value(c: ConstraintValue, pos: Pos) -> Fit:
    x = c(pos)
    if isinstance(x, list):
        raise ValueError(
            "Expected constraint expression to return scalar value, got list"
        )
    return x


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
