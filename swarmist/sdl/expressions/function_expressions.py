from __future__ import annotations
from lark import v_args
from typing import Optional, cast, List, Dict, Any, Callable, Union
from dataclasses import dataclass
from functools import reduce
from swarmist.core.dictionary import UpdateContext, IReferences, IReference, Agent, Pos
from .expressions import Expressions, fetch_value


@v_args(inline=True)
class FunctionExpressions(Expressions):
    def reduce_func(self, values: Callable, func: FunctionDef, initial=0):
        args = func.args
        body = func.body
        if len(args) != 2:
            raise ValueError("Reduce function must have 2 arguments")

        def callback(ctx: UpdateContext):
            acc = initial if not callable(initial) else fetch_value(initial, ctx)
            to_reduce = self._get_value_list(values, ctx)
            #print(f"ACC BEFORE: {acc}")
            for value in to_reduce:
                acc = body(FunctionContext.of(ctx, {args[0]: acc, args[1]: value}))
            #print(f"ACC AFTER: {acc}")
            #print(f"##############################################")
            return acc

        return callback

    def filter_func(self, values: Callable, func: FunctionDef):
        args = func.args
        body = func.body
        if len(args) != 1:
            raise ValueError("Reduce function must have 1 argument")

        def callback(ctx: UpdateContext):
            to_filter = self._get_value_list(values, ctx)
            return [
                value
                for value in to_filter
                if body(FunctionContext.of(ctx, {args[0]: value}))
            ]

        return callback

    def func_def(self, *defs):
        args = [defs[i] for i in range(0, len(defs) - 1)]
        body = defs[-1]
        return FunctionDef(body, args)

    def _get_value_list(self, values, ctx: UpdateContext):
        _values = fetch_value(values, ctx)
        if _values is None:
            return []
        elif isinstance(_values, IReferences):
            return cast(IReferences, _values).all()
        elif hasattr(_values, "__len__"):
            return _values
        else:
            return [_values]


@dataclass(frozen=True)
class FunctionContext(UpdateContext):
    args: Dict[str, Union[Agent, IReference, Any]]

    @classmethod
    def of(cls, ctx: UpdateContext, args: Union[Agent, IReference]) -> FunctionContext:
        return cls(
            agent=ctx.agent,
            swarm=ctx.swarm,
            search_context=ctx.search_context,
            random=ctx.random,
            args=args,
            vars=ctx.vars,
        )

    def get_arg(self, name: str, prop: str = None):  
        arg = self.args[name] if name in self.args else None
        if arg is None:
           return self._assert_and_get_ctx_var(name, prop)
        if not prop: 
            return arg
        if isinstance(arg, IReference):
            return arg.get(prop.lower())
        if isinstance(arg, Agent):
            return arg[prop.lower()]
        raise ValueError(f"Argument {arg} not found in function context")
        
    def _assert_and_get_ctx_var(self, name: str, prop: Optional[str]):
        if not prop:
            return self.get(name)
        raise ValueError(f"Argument {name} not found in function context")

@dataclass(frozen=True)
class FunctionDef:
    body: Callable
    args: List[str]
