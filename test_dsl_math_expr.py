from swarmist.sdl.parser import Parser
from swarmist.core.dictionary import SearchContext
#expression = """1+2*4*(sin(90)*pi())"""
search_context = SearchContext(
    evaluate=None,
    parameters=None,
    ndims=20,
    bounds =None,
    curr_gen=None,
    max_gen=None,
    curr_fit=None,
    min_fit=None,
    curr_eval=None,
    max_evals=None
)
# print(search_context.__dict__)
# present = "ndims" in search_context
# print(present)
#expression = """pick_random(unique 3 with replacement)"""
expression = """
PARAMETERS (
    C1 = 2.05 BOUNDED BY (0, 8)
    C2 = 2.05 BOUNDED BY (0, 8) 
    CHI = 0.7298 BOUNDED BY (0, 1)
)
USING BINOMIAL RECOMBINATION WITH PROBABILITY 0.5 
UPDATE (
    VELOCITY = PARAM(CHI) * (
        DELTA + PARAM(C1) * RANDOM() * (BEST-POS) + PARAM(C2) * RANDOM() * (SWARM_BEST()-POS)
    )
    POS = POS + VELOCITY
)
"""
res = Parser().parse(expression)
print(res)
print(res(search_context))