from typing import Optional
from lark import Token

class TerminatonParser:
    def __init__(self):
        self._max_gen:Optional[int] = None
        self._max_evals:Optional[int] = None
        self._min_fit:Optional[float] = None
        
    def max_gen(self, val:int)->Token:
        self._max_gen = val
        return val
    
    def max_evals(self, val:int)->Token:
        self._max_evals = val
        return val
        
    def min_fit(self, val:float)->Token:
        self._min_fit = val
        return val
    
    def get(self):
        raise NotImplementedError()
    