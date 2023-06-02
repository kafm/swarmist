from typing import Any, Optional, cast
from lark import Lark, v_args
from typing import List, Callable
import swarmist as sw
from swarmist.update import UpdateBuilder
from swarmist.core.dictionary import UpdateContext, Selection, Bounds, Agent, StopCondition
from .grammar import grammar
from .expressions import *


@v_args(inline=True)
class GrammarTransformer(MathExpressions, RandomExpressions, ReferencesExpressions, InitExpressions, UpdateExpressions):
    def __init__(self):
        self._strategy = sw.strategy()
        self._pipeline: List[UpdateBuilder] = []


    def build_strategy(self, *args):
        return self._strategy.pipeline(*self._pipeline)
        #return self._strategy

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

    def get_var(self, name):
        def callback(ctx=None):
            if not ctx: 
                raise ValueError("Getting var with no context is not allowed")
            elif isinstance(ctx, Agent):
                return ctx[name]
            return cast(UpdateContext, ctx).get(name)
        return callback

    def get_parameter(self, name):
        return lambda ctx=None: cast(UpdateContext, ctx).param(name)

    def set_parameter(self, name, value, bounds: Bounds):
        self._strategy.param(name, value, bounds.min, bounds.max)
        return None
    
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

    def assign_var(self, name, value):
        self.vars[name] = value
        return value


class Parser:
    def __init__(self, grammar: str = grammar):
        self.transformer = GrammarTransformer()
        self.lexer = Lark(grammar, parser="lalr", transformer=self.transformer, start=["start", "strategy_expr", "termination_expr"])

    def parse(self, expression, start="start"):
        self.transformer.__init__()
        result = self.lexer.parse(expression, start=start)
        # print(f"Expression: {expression}")
        # print(f"Result: {result}")
        return result

    #def parse(self, text: str, start: Optional[str]=None, on_error: 'Optional[Callable[[UnexpectedInput], bool]]'=None) -> 'ParseTree':