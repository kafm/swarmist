from swarmist.sdl.parser import Parser
from swarmist.core.dictionary import SearchResults

class Sdl:
    """Swarmist Domain Language (SDL) interface."""

    def execute(self, query: str)->SearchResults:
        """Execute a swarm search query."""
        query = Parser().parse(query)
        return query()