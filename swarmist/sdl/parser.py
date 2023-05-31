from typing import Any, Optional, cast
from lark import Lark, Transformer, v_args, Token
from functools import reduce
import swarmist as sw
from swarmist.core.dictionary import UpdateContext
from .grammar import grammar
from .expressions import *


@v_args(inline=True)
class GrammarTransformer(MathExpressions, RandomExpressions, ReferencesExpressions, UpdateExpressions):
    def __init__(self):
        self.strategy = sw.strategy()

    def get_var(self, name):
        return lambda ctx=None: cast(UpdateContext, ctx).get(name)

    def get_parameter(self, name):
        return lambda ctx=None: cast(UpdateContext, ctx).param(name)

    def set_parameter(self, name, value, bounds):
        self.strategy.param(name, value, bounds)
        return None

    def bounds(self, lower, upper):
        return sw.Bounds(lower, upper)

    def assign_var(self, name, value):
        self.vars[name] = value
        return value


class Parser:
    def __init__(self, grammar: str = grammar):
        self.transformer = GrammarTransformer()
        self.lexer = Lark(grammar, parser="lalr", transformer=self.transformer)

    def parse(self, expression):
        self.transformer.__init__()
        result = self.lexer.parse(expression)
        print(f"Expression: {expression}")
        print(f"Result: {result}")
        return result
