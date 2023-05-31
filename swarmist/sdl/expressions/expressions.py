from lark import v_args, Transformer

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
        ctx_dict = ctx.__dict__ if ctx else {}
        size = ctx_dict["ndims"] if "ndims" in ctx_dict else 1
    return int(size)


def fetch_value(x, ctx=None):
    if callable(x):
        return x(ctx)
    return x
