from __future__ import annotations
from typing import Tuple
import numpy as np
from swarmist.core.dictionary import FitnessFunction, Bounds


def sphere() -> Tuple(FitnessFunction, Bounds):
    return (lambda x: np.sum(x**2), Bounds(min=-5.12, max=5.12))


def ackley() -> Tuple(FitnessFunction, Bounds):
    def callback(x, a=20, b=0.2, c=2 * np.pi):
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + a + np.exp(1)

    return (callback, Bounds(min=-32.768, max=32.768))


def griewank() -> Tuple(FitnessFunction, Bounds):
    def callback(x, fr=4000):
        n = len(x)
        ii = np.arange(1.0, n + 1)
        return np.sum(x**2) / fr - np.prod(np.cos(x / np.sqrt(ii))) + 1

    return (callback, Bounds(min=-600.0, max=600.0))


def rastrigin() -> Tuple(FitnessFunction, Bounds):
    return (
        lambda x: 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)),
        Bounds(min=-5.12, max=5.12),
    )


def schwefel() -> Tuple(FitnessFunction, Bounds):
    return (
        lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))),
        Bounds(min=-500, max=500),
    )


def rosenbrock() -> Tuple(FitnessFunction, Bounds):
    def callback(x):
        xi = x[:-1]
        xnext = x[1:]
        return np.sum(100 * (xnext - xi**2) ** 2 + (xi - 1) ** 2)

    return (callback, Bounds(min=-2.048, max=2.048))


def michalewicz() -> Tuple(FitnessFunction, Bounds):
    def callback(x, m=10):
        ii = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * (np.sin(ii * x**2 / np.pi)) ** (2 * m))

    return (callback, Bounds(min=0, max=np.pi))


def dejong3() -> Tuple(FitnessFunction, Bounds):
    return (lambda x: np.sum(np.floor(x)), Bounds(min=-5.12, max=5.12))


def dejong5() -> Tuple(FitnessFunction, Bounds):
    def callback(x):
        x1 = x[0]
        x2 = x[1]
        a1 = np.array(
            [
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
                -32,
                -16,
                0,
                16,
                32,
            ]
        )
        a2 = np.array(
            [
                -32,
                -32,
                -32,
                -32,
                -32,
                -16,
                -16,
                -16,
                -16,
                -16,
                0,
                0,
                0,
                0,
                0,
                16,
                16,
                16,
                16,
                16,
                32,
                32,
                32,
                32,
                32,
            ]
        )
        ii = np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
            ]
        )
        return (0.002 + np.sum(1 / (ii + (x1 - a1) ** 6 + (x2 - a2) ** 6))) ** -1

    return (callback, Bounds(min=-65.536, max=65.536))


def easom() -> Tuple(FitnessFunction, Bounds):
    def callback(x):
        x1 = x[0]
        x2 = x[1]
        return (
            -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2)
        )

    return (callback, Bounds(min=-100, max=100))
