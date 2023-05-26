from __future__ import annotations
from typing import Callable, TypeVar, List, Optional, Any
from pymonad.either import Left, Right, Either
from .dictionary import KeyValue

T = TypeVar("T")

class SearchEnded(Exception):
    "Raised when the search ended gracefully"
    pass     

def try_catch(f: Callable[[], T])->Either[Exception, T]:
    try: 
        return Right(f())
    except Exception as e:
      return Left(e)

def assert_not_null(val: Any, parameter: str):
      if not val: 
            raise ValueError(f"{parameter} is null") 
      
def assert_not_empty(val: List[Any], parameter: str):
      assert_not_null(val, parameter)
      if len(parameter) == 0:
            raise ValueError(f"{parameter} is empty") 

def assert_at_least(val: Optional[int], min_val: int,  parameter: str):
      if val and val < min_val:
            raise ValueError(f"{parameter} must be at lest {min_val}") 
      
def assert_greater_than(val: int, min_val: int,  parameter: str):
      if val > min_val:
            raise ValueError(f"{parameter} must be at lest {min_val}") 
      
def assert_equal_length(val: int, expected: int,  parameter: str):
      if val != expected:
            raise ValueError(f"{parameter} must be equal to {expected}") 
      
def assert_at_least_one_nonnull(kv: KeyValue):
    assert_not_null(kv, "Dictionary is null")
    keys = kv.keys()
    for k in keys: 
        if kv[k]:
            return 
    params_str = ",".join(keys)
    raise ValueError(f"At least one of [{params_str}] should be provided")
      
def assert_callable(f: Any, parameter: str):
      if not callable(f):
            raise ValueError(f"{parameter} should be callable.")
      
def assert_number(num: Any, parameter: str):
      t = type(num)
      if t not in [float, int]:
            raise ValueError(f"{parameter} should be a number.")