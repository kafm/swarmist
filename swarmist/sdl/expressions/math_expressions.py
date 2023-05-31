import math
import numpy as np
import functools as ft
from lark import v_args
from .expressions import Expressions, fetch_value


@v_args(inline=True)
class MathExpressions(Expressions):
    def __init__(self):
        super().__init__()

    def and_(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) and fetch_value(y, ctx)

    def or_(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) or fetch_value(y, ctx)

    def lt(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) < fetch_value(y, ctx)

    def le(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) <= fetch_value(y, ctx)

    def gt(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) > fetch_value(y, ctx)

    def ge(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) >= fetch_value(y, ctx)

    def eq(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) == fetch_value(y, ctx)

    def ne(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) != fetch_value(y, ctx)

    def add(self, x, y, z=None):
        return lambda ctx=None: fetch_value(x, ctx) + fetch_value(y, ctx)

    def sub(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) - fetch_value(y, ctx)

    def mul(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) * fetch_value(y, ctx)

    def div(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) / fetch_value(y, ctx)

    def floordiv(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) // fetch_value(y, ctx)

    def neg(self, x):
        return lambda ctx=None: -fetch_value(x, ctx)

    def pow(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) ** fetch_value(y, ctx)

    def mod(self, x, y):
        return lambda ctx=None: fetch_value(x, ctx) % fetch_value(y, ctx)

    def sin(self, x):
        return lambda ctx=None: math.sin(fetch_value(x, ctx))

    def cos(self, x):
        return lambda ctx=None: math.cos(fetch_value(x, ctx))

    def tan(self, x):
        return lambda ctx=None: math.tan(fetch_value(x, ctx))

    def arcsin(self, x):
        return lambda ctx=None: math.asin(fetch_value(x, ctx))

    def arccos(self, x):
        return lambda ctx=None: math.acos(fetch_value(x, ctx))

    def arctan(self, x):
        return lambda ctx=None: math.atan(fetch_value(x, ctx))

    def sqrt(self, x):
        return lambda ctx=None: math.sqrt(fetch_value(x, ctx))

    def log(self, x):
        return lambda ctx=None: math.log(fetch_value(x, ctx))

    def exp(self, x):
        return lambda ctx=None: math.exp(fetch_value(x, ctx))

    def abs(self, x):
        return lambda ctx=None: abs(fetch_value(x, ctx))

    def norm(self, x):
        return lambda ctx=None: np.linalg.norm(fetch_value(x, ctx))

    def sum(self, x):
        def callback(ctx=None):
            val = fetch_value(x, ctx)
            if hasattr(val, "sum"):
                return val.sum()
            elif hasattr(val, "__len__"):
                return sum(val)
            else:
                return val

        return callback

    def min(self, x):
        def callback(ctx=None):
            val = fetch_value(x, ctx)
            if hasattr(val, "min"):
                return val.min()
            elif hasattr(val, "__len__"):
                return min(val)
            else:
                return val

        return callback

    def max(self, x):
        def callback(ctx=None):
            val = fetch_value(x, ctx)
            if hasattr(val, "max"):
                return val.max()
            elif hasattr(val, "__len__"):
                return max(val)
            else:
                return val

        return callback

    def avg(self, x):
        def callback(ctx=None):
            val = fetch_value(x, ctx)
            if hasattr(val, "avg"):
                return val.avg()
            elif hasattr(val, "__len__"):
                return val / len(val)
            else:
                return val

        return callback

    def reduce(self, acc, x, initial=None):
        def callback(ctx=None):
            val = fetch_value(x, ctx)
            if hasattr(val, "reduce"):
                return val.reduce(acc, initial)
            elif hasattr(val, "__len__"):
                return ft.reduce(acc, val, initial)
            else:
                return val

        return callback

    def pi(self):
        return lambda _=None: math.pi
