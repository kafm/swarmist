from swarmist.sdl.parser import Parser
from swarmist.core.dictionary import SearchResults
from swarmist.strategy import Strategy

class Sdl:
    """Swarmist Domain Language (SDL) interface."""

    def execute(self, query: str)->SearchResults:
        query = Parser().parse(query)
        return query()
    
    def strategy(self, query: str)->Strategy:
        return Parser().parse(query, start="strategy_expr")