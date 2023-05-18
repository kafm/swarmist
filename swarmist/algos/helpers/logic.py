from typing import Callable, TypeVar

T = TypeVar("T")
F = TypeVar("F")
Condition = Callable[..., bool] | bool

def if_then(condition: Condition, _true: Callable[..., T], _false: Callable[..., F]):
    if (callable(condition) and condition()) or condition: 
       return _true()
    return _false()

def unless_then(condition: Condition, _false: Callable[..., T], _true: Callable[..., F]):
    return if_then(condition, _true, _false)