from lark import Lark, Transformer, v_args
import numpy as np
# Define the grammar using Lark's EBNF notation
# grammar = """
#     start: search_stmt | create_stmt
#     ?search_stmt: "search"i "(" space_expr ")" "using"i "(" strategy_expr ")" "until"i "(" termination_expr+ ")"
#     ?create_stmt: create_space_stmt | create_param_adaptor_stmt
#     ?create_space_stmt: "create"i "space"i NAME "(" space_expr ")"
#     ?create_param_adaptor_stmt: "create"i "param_adaptor"i NAME "(" param_adaptor_expr ")"
#     ?strategy_expr: "population"i size_expr init_expr parameter_expr? update_expr+
#     ?init_expr: "init"i init_pos_expr topology_expr? 
#     ?init_pos_expr: rand_prefix rand_distribution "(" NUMBER? ("," NUMBER)? ("," NUMBER)? ")"  
#         | rand
#     ?topology_expr: "with"i "topology"i topology
#     ?topology: "gbest"i | "lbest"i size_expr?
#     ?parameter_expr: "set"i "parameters"i "(" parameter+ ")"
#     ?parameter: NAME "=" NUMBER bounds_expr? 
#     ?update_expr: selection_expr "(" ("using"i recombination_expr)? "update"i "(" update_prop_expr+ ")" ")" when_expr?
#     ?selection_expr: "select"i quantity_expr ("with"i selection_method_expr)? ("where"i where_expr)* ("order"i order_expr)?
#     ?selection_method_expr: "roulette"i 
#         | "random"i 
#         | "probability"i probability
#     ?where_expr: where_expr ("and"i | "or"i) where_expr
#         | "(" where_expr ")"
#         | "not"i? "improved"
#         | agent_prop comp_expr NUMBER
#     ?order_expr: "by"i agent_prop ("asc"i | "desc"i)?    
#     ?recombination_expr: "binomial"i "recombination"i ("with"i "probability"i NUMBER)?
#         | "exponential"i "recombination"i ("with"i "probability"i NUMBER)?
#         | quantity_expr ("with"i "probability"i NUMBER)?
#         | "reset"i
#         | quantity_expr "random"i
#     ?quantity_expr: INT | "all"i
#     ?update_prop_expr: NAME "=" update_val_expr
#     ?update_val_expr: update_val_expr op update_val_expr
#         | "(" update_val_expr ")"
#         | "param"i "(" NAME ")"
#         | rand_update_ctx_expr
#         | "if_then"i "(" where_expr "," update_val_expr ","  update_val_expr ")"
#         | references_expr
#         | aggregation_expr
#         | NAME
#         | NUMBER
#     ?aggregation_expr: "sum"i "(" update_val_expr ")"
#         | "avg"i "(" update_val_expr ")"
#         | "min"i "(" update_val_expr ")"
#         | "max"i "(" update_val_expr ")"
#         | "reduce"i "(" update_val_expr ", " update_val_expr ")"
#     ?references_expr: "swarm_best"i "(" INT? ")"
#         | "swarm_worst"i "(" INT? ")"
#         | "neighborhood"i "(" ")"
#         | "pick_random"i "(" "unique"i? INT? ("with"i "replacement"i)? ")"
#         | "pick_roulette"i "(" "unique"i? INT? ("with"i "replacement"i)? ")"
#         | "rand_to_best" "(" ("with"i "probability"i NUMBER)? ")"
#         | "current_to_best"i "(" ("with"i "probability" NUMBER)? ")"
#     ?rand_update_ctx_expr: rand_prefix rand_distribution "(" update_val_expr? ("," update_val_expr)? ("," update_val_expr)? ")" 
#         | rand
#     ?when_expr: "when"i where_expr
#     ?space_expr: variable_expr+ space_objective_expr constraints_expr?
#     ?variable_expr: "variable"i NAME size_expr bounds_expr 
#     ?space_objective_expr: ("minimize"i | "maximize"i) expr
#     ?constraints_expr: "constrained"i "by"i "(" constraint_expr+ ")" 
#     ?constraint_expr: expr comp_expr expr
#     ?param_adaptor_expr: "TODO" NAME "(" param_adaptor_prop+ ")"
#     ?param_adaptor_prop: "TODO"
#     ?comp_expr: "<" | "<=" | ">" | ">=" | "==" | "!="
#     ?rand_prefix: "rand_"i
#     ?rand_distribution: "random"i
#         | "uniform"i
#         | "beta"i 
#         | "exponential"i 
#         | "rayleigh"i 
#         | "normal"i
#         | "lognormal"i 
#         | "skewnormal"i
#         | "weibull"i
#     ?rand: "rand"i "(" ")"
#     ?bounds_expr: "bounded"i "by"i "(" NUMBER "," NUMBER ")"
#     ?size_expr: "size"i "(" INT ")"
#?agent_prop: "index"i | "trials"i | "best"i | "pos"i | "fit"i | "delta"i | "improved"i
#?termination_expr: "evaluations"i "= " INT | "generation"i "=" INT | "fitness"i "=" NUMBER 
#?probability: NUMBER
grammar = """
    ?agent_prop: "index"i 
        | "trials"i 
        | "best"i 
        | "pos"i 
        | "fit"i 
        | "delta"i 
        | "improved"i
    ?termination_expr: "evaluations"i "= " INT 
        | "generation"i "=" INT 
        | "fitness"i "=" NUMBER 
    ?probability: NUMBER -> probability
    ?var: NAME "=" math_expr    -> assign_var
    ?math_expr: math_term
        | math_expr "+" math_term   -> add
        | math_expr "-" math_term   -> sub
    ?math_term: atom
        | math_term "*" atom  -> mul
        | math_term "/" atom  -> div
        | math_term "%" atom  -> mod
        | math_term "**" atom -> pow
        | math_term "//" atom -> floordiv
    ?atom: NUMBER           -> number
        | "-" atom         -> neg
        | NAME             -> var
        | NAME "[" INT "]" -> var_index
        | "(" math_expr ")"
        | "sin"i "(" math_expr ")" -> sin
        | "cos"i "(" math_expr ")" -> cos
        | "tan"i "(" math_expr ")" -> tan
        | "arcsin"i "(" math_expr ")" -> arcsin
        | "arccos"i "(" math_expr ")" -> arccos
        | "arctan"i "(" math_expr ")" -> arctan
        | "sqrt"i "(" math_expr ")" -> sqrt
        | "log"i "(" math_expr ")" -> log
        | "exp"i "(" math_expr ")" -> exp
        | "abs"i "(" math_expr ")" -> abs
        | "norm"i "(" math_expr ")" -> norm
        | "sum"i "(" math_expr ")" -> sum
        | "min"i "(" math_expr ")" -> min
        | "max"i "(" math_expr ")" -> max
        | "avg"i "(" math_expr ")" -> avg

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
"""
# grammar = """
#     ?start: sum
#           | NAME "=" sum    -> assign_var

#     ?sum: product
#         | sum "+" product   -> add
#         | sum "-" product   -> sub

#     ?product: atom
#         | product "*" atom  -> mul
#         | product "/" atom  -> div

#     ?atom: NUMBER           -> number
#          | "-" atom         -> neg
#          | NAME             -> var
#          | "(" sum ")"

#     %import common.CNAME -> NAME
#     %import common.NUMBER
#     %import common.WS_INLINE

#     %ignore WS_INLINE
# """

@v_args(inline=True) 
class GrammarTransformer(Transformer):
    from operator import add, sub, mul, truediv as div, floordiv, neg, pow, mod
    from math import sin, cos, tan, asin, acos, atan, sqrt, log, exp
    number = float
    
    def __init__(self):
        self.vars = {}
        
    def probability(self, value):
        if value < 0 or value > 1:
           raise ValueError(f"Probability must be between 0 and 1")
        return value
             
    def abs(self, value):
        return abs(value)
    
    def norm(self, value):
        return np.linalg.norm(value)
    
    def sum(self, value):
        return sum(value)
   
    def min(self, value):
        return min(value)
    
    def max(self, value):
        return max(value)
    
    def avg(self, value):
        if hasattr(value, '__len__'):
            return sum(value)/len(value)
        return value
   
    def var(self, name):
        try:
            return self.vars[name]
        except KeyError:
            raise Exception(f"Variable not found: {name}")
        
    def var_index(self, name, index):
        value = self.var(name)
        if value is list:
            try:
                return value[index]  
            except KeyError:  
                raise IndexError(f"Index {index} out of bounds for variable {name}")
        else:
            raise ValueError(f"Variable {name} is not a list")
    
    


lexer = Lark(grammar, parser="lalr", transformer=GrammarTransformer())

def evaluate_expression(expression):
    tree = lexer.parse(expression)
    return tree


# # Example usage
expression = "2 * sin(3 + 4)"
result = evaluate_expression(expression)
print(f"Expression: {expression}")
print(f"Result: {result}")


# lexer.parse("""
# SEARCH(
#     VARIABLE X SIZE(20) BOUNDED BY (10, 20) 
#     MINIMIZE X**2
#     CONSTRAINED BY (
#         X[1] < X[2]   
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