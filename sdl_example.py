from swarmist.sdl.parser import parse
import math
# # Example usage
#expression = "until (evaluations=1000)"
#expression = "population size(30) init random.normal(loc=0,scale=1) with topology lbest size(3)"
#expression = "1 + 2 * 4 * (sin(90) * pi)"
expression = """
    PARAMETERS (
        X=1 + 2 * 4 * (sin(90) * pi) BOUNDED BY (10, 20)
        Y=30
    )
"""
res = parse(expression)
print(res())
print(1 + 2 * 4 * (math.sin(90) * math.pi))


# lexer.parse("""
# SEARCH(
#     VARIABLE X SIZE(20) BOUNDED BY (10, 20) 
#     MINIMIZE X**2
#     CONSTRAINED BY (
#           X[1] < X[2]  PENALIZE 100 OTHERWISE 
#           X[1:2] < X[:2]  PENALIZE 100 OTHERWISE 
#     )
# )
# USING (
#     POPULATION SIZE(30) INIT RAND_UNIFORM() WITH TOPOLOGY LBEST SIZE(3)
#     SET PARAMETERS (
#         C1 = 2.05 BOUNDED BY (0, 8)
#         C2 = 2.05 BOUNDED BY (0, 8) 
#         CHI = 0.7298 BOUNDED BY (0, 1)
#     )
#     SELECT ALL (
#         USING BINOMIAL RECOMBINATION WITH PROBABILITY 0.5 
#         UPDATE (
#             POS = POS + PARAM(CHI) * (
#                 DELTA + PARAM(C1) * RAND() * (BEST-POS) + PARAM(C2) * RAND() * (SWARM_BEST()-POS)
#             )
#         )
#     )
# )  
# UNTIL(
#     EVALUATIONS = 400
# ) 
# """)

# Create the Lark parser using the defined grammar and transformer
# calculator = Lark(grammar, parser="lalr", transformer=CalcTransformer())


# grammar = """
# <SEARCH> ::= SEARCH ( <SPACE> ) USING ( <STRATEGY> ) UNTIL ( <STOP_CONDITION> )
# <SPACE> ::= VARIABLE <VARIABLE> SIZE (<SIZE>) BOUNDED BY (<BOUNDS>) MINIMIZE (<FITNESS_FUNC)> CONSTRAINED BY (<CONSTRAINTS>)



# SEARCH(
#     VARIABLE X SIZE(20) BOUNDED BY (10, 20) 
#     MINIMIZE X**2
#     CONSTRAINED BY (
#         X[1] < X[2]   
#     )
# )
# USING (
#     POPULATION SIZE(30) INIT RANDOM WITH TOPOLOGY LBEST(5)
#     ADD PARAMETERS (
#          SELF_CONFIDANCE: AUTO(), --SUPPORT UNIFORM/NORMAL/... PARAMETER SETTING. INSPIRE FROM PYMC
#          ... 
#     )
    #   SELECT MAX(TRIALS) WHERE TRIALS > 3 (
    #     USING BINOMIAL RECOMBINATION WITH PROBABILITY 0.5 
    #     UPDATE (
    #         VELOCITY = SELF_CONFIDANCE*(BEST-POS)...
    #         POS = POS + VELOCITY  ...                       
    #     ) WHEN IMPROVED
    #   )  
#     PARALLEL UPDATE ALL (  
#        VELOCITY = SELF_CONFIDANCE*(BEST-POS)...
#        POS = POS + VELOCITY  ...
#     ) 
#     PARALLEL UPDATE 3 WITH ROULETTE SELECTION (
#        POS= BEST + POS ...
#     )
#     PARALLEL UPDATE 3 WITH RANDOM SELECTION (
#        POS= BEST + POS ...
#     ) WHERE IMPROVED --TO SPLIT PARAMETERS UPDATE FROM POS UPDATE? TO REPLACE FIT OR ONLY POS . R: POS IS THE AUXILIARY. PBEST IS THE REAL POS
# ) TUNE WITH ACO() --ACO, GE...TO FIND THINGS TO TUNE. OR SHOULD IT BE OUTSITE OF THE DSL. AT PYTHON LEVEL
# UNTIL(
#     EVALUATIONS(400)
# )
# """