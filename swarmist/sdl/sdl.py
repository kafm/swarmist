from typing import Union
from swarmist.sdl.parser import Parser
from swarmist.core.dictionary import SearchResults, TuneResults
from swarmist.strategy import Strategy

class Sdl:
    """Swarmist Domain Language (SDL) interface."""

    def execute(self, query: str)->Union[SearchResults, TuneResults]:
        query = Parser().parse(query)
        return query()
    
    def strategy(self, query: str)->Strategy:
        return Parser().parse(query, start="strategy_expr")