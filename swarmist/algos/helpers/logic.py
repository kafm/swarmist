from typing import Callable, TypeVar

T = TypeVar("T")
F = TypeVar("F")
Condition = Callable[..., bool] | bool

def is_true(condition: Condition)->bool:
    res = condition if not callable(condition) else condition()
    return res and True

def if_then(condition: Condition, _true: Callable[..., T], _false: Callable[..., F]):
    return _true() if is_true(condition) else _false()

def unless_then(condition: Condition, _false: Callable[..., T], _true: Callable[..., F]):
    return if_then(condition, _true, _false)