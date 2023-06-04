from lark import v_args, Transformer
from typing import cast
from swarmist.core.dictionary import UpdateContext, SearchContext, IReference, Agent

@v_args(inline=True)
class Expressions(Transformer):
    def value_to_lambda(self, x):
        return lambda _=None: x

    def number(self, value):
        return float(value)

    def integer(self, value):
        return int(value)

    def string(self, value):
        return str(value)

    def true(self):
        return True

    def false(self):
        return False

def fetch_dimensions(x, ctx=None):
    size = fetch_value(x, ctx)
    if not size: 
        if not ctx:
            return 1
        elif isinstance(ctx, UpdateContext):
            return cast(UpdateContext, ctx).agent.ndims
        elif isinstance(ctx, SearchContext):
            return cast(SearchContext, ctx).ndims
        else: 
            raise ValueError(f"Invalid context type: {type(ctx)}")
    return int(size)


def fetch_value(x, ctx=None):
    if callable(x):
        val = x(ctx)
        if isinstance(val, Agent):
            return val.best
        if isinstance(val, IReference):
            return cast(IReference, val).get()
        return val
    return x
