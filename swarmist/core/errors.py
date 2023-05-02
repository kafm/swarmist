from typing import Callable, TypeVar, List, Optional, Any
from pymonad.either import Left, Right, Either
import inspect
from .dictionary import KeyValue


T = TypeVar("T")

class SearchEnded(Exception):
    "Raised when the search ended gracefully"
    pass     

def try_catch(f: Callable[[], T])->Either[T,Exception]:
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
    assert_not_null(vars, "Dictionary is null")
    keys = kv.keys()
    for key in keys: 
        if key[kv[key]]:
            return 
    params_str = ",".join(keys)
    raise(f"At least one of [{params_str}] should be provided")
      
def assert_function_signature(f: Callable, expected: Callable, parameter: str):
      func_signature = inspect.signature(f)
      expected_signature = inspect.signature(expected)
      if func_signature != expected_signature:
            raise ValueError(f"{parameter} function signature mismatch. Expected: {expected_signature}, got: {func_signature}")