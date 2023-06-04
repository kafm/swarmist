from typing import Tuple, Dict, Any, Union, Callable, Optional
from lark import v_args
from swarmist.core.random import Random
from swarmist.core.dictionary import Pos
from .expressions import Expressions, fetch_dimensions, fetch_value


@v_args(inline=True)
class RandomExpressions(Expressions):
    def random(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.rand(**props)
        )

    def random_uniform(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.uniform(**props)
        )

    def random_normal(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.normal(**props)
        )

    def random_lognormal(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.lognormal(**props)
        )

    def random_skewnormal(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.skewnormal(**props)
        )

    def random_cauchy(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.cauchy(**props)
        )

    def random_levy(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.levy(**props)
        )

    def random_beta(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.beta(**props)
        )

    def random_exponential(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.exponential(**props)
        )

    def random_rayleigh(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.rayleigh(**props)
        )

    def random_weibull(self, props: Dict[str, Any] = {}):
        return self._get_generator(
            props, lambda generator, props: generator.weibull(**props)
        )

    def _get_generator(
        self, props: Dict[str, Any], callback: Callable[[Random, Dict[str, Any]], Any]
    ) -> Callable[[Optional[Any]], Union[Pos, float]]:
        size = props.pop("size") if "size" in props else None
        return lambda ctx=None: callback(
            Random(fetch_dimensions(size, ctx)), 
            self._exec_props(props, ctx)
        )
    
    def _exec_props(self, props: Dict[str, Any] = {}, ctx=None):
        return {
            key: value(ctx) if callable(value) else value
            for key, value in props.items()
        }

    def random_props(self, *args):
        return {arg[0]: arg[1] for arg in args}

    def random_loc(self, x):
        return ("loc", x)

    def random_scale(self, x):
        return ("scale", x)

    def random_shape(self, x):
        return ("shape", x)

    def random_alpha(self, x):
        return ("alpha", x)

    def random_beta(self, x):
        return ("beta", x)

    def random_low(self, x):
        return ("low", x)

    def random_high(self, x):
        return ("high", x)

    def random_size(self, x):
        return ("size", x)

    def probability(self, x):
        def callback(ctx=None):
            value = fetch_value(x, ctx)
            if value < 0 or value > 1:
                raise ValueError(f"Probability must be between 0 and 1")
            return value
        return callback
