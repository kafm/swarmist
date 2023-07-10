from typing import Any, Optional, cast
from lark import Lark, v_args, Tree
from typing import List, Callable, Union
import numpy as np
import swarmist as sw
from swarmist.update import UpdateBuilder
from swarmist.core.dictionary import (
    UpdateContext,
    Selection,
    Bounds,
    Agent,
    AutoFloat,
    AutoInteger,
)
from .grammar import grammar
from .expressions import *


@v_args(inline=True)
class GrammarTransformer(
    MathExpressions,
    RandomExpressions,
    ReferencesExpressions,
    InitExpressions,
    UpdateExpressions,
    SpaceExpressions,
    FunctionExpressions,
):
    def __init__(self):
        self._strategy = sw.strategy()
        self._pipeline: List[UpdateBuilder] = []
        self._get_var = None
        self._tune_until_generation = None
        self._requires_tunning = False

    def search(self, space_assets: SpaceAssets, strategy: sw.Strategy, stop_condition):
        self._get_var = space_assets.get_var
        if not self._requires_tunning:
            if self._tune_until_generation:
                raise "Tune configuration provided but no auto expressions found"
            return lambda: sw.search(
                space_assets.space, sw.until(**stop_condition), sw.using(strategy)
            )
        elif not self._tune_until_generation:
            raise "Tune configuration not provided but auto expressions found"
        return lambda: sw.search(
            space_assets.space,
            sw.until(**stop_condition),
            sw.tune(strategy, max_gen=self._tune_until_generation),
        )

    def build_strategy(self, *_):
        return self._strategy.pipeline(*self._pipeline)

    def set_tunning_config(self, max_gen):
        self._tune_until_generation = max_gen
        return None

    def init(self, pop_size, init_method, topology=None):
        self._strategy.init(init_method, pop_size)
        self._strategy.topology(topology)

    def update(self, selection: Callable[..., Selection], update_tail: UpdateTail):
        self._pipeline.append(
            sw.select(selection)
            .update(**update_tail.update_pos)
            .recombinant(update_tail.recombination)
            .where(update_tail.when)
        )

    def get_var(self, name: str, prop: str = None):
        def callback(ctx=None):
            if ctx is None:
                raise ValueError("Getting var with no context is not allowed")
            elif name.lower() == "population_size":
                return self._strategy.population_size()
            elif isinstance(ctx, FunctionContext):
                return ctx.get_arg(name, prop)
            elif isinstance(ctx, UpdateContext):
                return cast(UpdateContext, ctx).get(name)
            elif isinstance(ctx, Agent):
                return ctx[name.lower()]
            else:
                return self._get_var(ctx, name)

        return callback

    def get_parameter(self, name: str):
        def callback(ctx: UpdateContext = None):
            if ctx is None:
                return self._strategy.get_param(name)
            return ctx.param(name)

        return callback

    def set_parameter(self, name, value, bounds: Optional[Bounds] = None):
        self._strategy.param(name, value, bounds=bounds)
        return None

    def set_auto_parameter(self, name, value: Union[AutoInteger, AutoFloat]):
        self._strategy.param(name, value)
        return None

    def auto_integer(self, bounds: Bounds):
        self._requires_tunning = True
        return AutoInteger(bounds.min, bounds.max)

    def auto_float(self, bounds: Bounds):
        self._requires_tunning = True
        return AutoFloat(bounds.min, bounds.max)

    def stop_condition(self, *args):
        return {arg[0]: arg[1] for arg in args}

    def set_max_evals(self, max_evals):
        return ("max_evals", max_evals)

    def set_max_gen(self, max_gen):
        return ("max_gen", max_gen)

    def set_min_fit(self, min_fit):
        return ("fit", min_fit)

    def bounds(self, lower, upper):
        return sw.Bounds(lower, upper)

    def bound(self, value):
        return value

    def neg_bound(self, value):
        return -value


class Parser:
    def __init__(self, grammar: str = grammar):
        self.transformer = GrammarTransformer()
        self.lexer = Lark(
            grammar,
            parser="lalr",
            transformer=self.transformer,
            start=["start", "strategy_expr"],
        )

    def parse(self, expression, start="start") -> Union[sw.Strategy, sw.SearchResults, sw.TuneResults]:
        self.transformer.__init__()
        result = self.lexer.parse(expression, start=start)
        # print(f"Expression: {expression}")
        # print(f"Result: {result}")
        return result

    # def parse(self, text: str, start: Optional[str]=None, on_error: 'Optional[Callable[[UnexpectedInput], bool]]'=None) -> 'ParseTree':
