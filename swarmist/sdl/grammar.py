from lark import Lark, Transformer
# Define the grammar using Lark's EBNF notation
grammar = """
    start: search_stmt | create_stmt
    ?search_stmt: "search" space_expr "using" strategy_expr "until" "(" termination_expr+ ")"
    ?create_stmt: create_space_stmt | create_param_adaptor_stmt
    ?create_space_stmt: "create" "space" NAME "(" space_expr ")"
    ?create_param_adaptor_stmt: "create" "param_adaptor" NAME "(" param_adaptor_expr ")"
    ?strategy_expr: "population" size_expr init_expr parameter_expr?
    ?init_expr: "init" distribution topology_expr? 
    ?parameter_expr: "set" "parameters" "(" parameter+ ")"
    ?parameter: NAME "=" NUMBER bounds_expr? 
    ?topology_expr: "with" "topology" "(" topology ")"
    ?topology: "gbest" | "lbest" size_expr
    ?space_expr: variable_expr+ space_objective_expr constraints_expr?
    ?variable_expr: variable NAME size_expr bounds_expr 
    ?space_objective_expr: "minimize" expr | "maximize" expr
    ?constraints_expr: "constrained" "by" "(" constraint_expr+ ")" 
    ?constraint_expr: expr comp_expr expr
    ?comp_expr: "<" | "<=" | ">" | ">=" | "==" | "!="
    ?distribution: "random"
        | "uniform" 
        | "beta" 
        | "exponential" 
        | "rayleigh" 
        | "normal" 
        | "lognormal" 
        | "skewnormal"
        | "weibull" 
    ?bounds_expr: "bounded" "by" "(" NUMBER "," NUMBER ")"
    ?size_expr: "size" "(" INT ")"
    ?expr: expr op expr
        | (expr op expr)
        | <pre_op>(expr)
        | var
    ?op: "+" | "-" | "*" | "/" | "%" | "**" | "//" 
    ?pre_op: "sin" 
        | "cos" 
        | "tan"
        | "arcsin"
        | "arccos"
        | "arctan"
        | "sqrt"
        | "log"
        | "exp"
        | "abs"
        | "norm" 
        | "sum"
        | "prod"
        | "min"
        | "max"
        | "avg"
    ?var: NAME "[" INT "]" | NAME | NUMBER
    ?termination_expr: "evaluations" "(" INT ")" | "max_gen" "(" INT ")" | "fitness" "(" NUMBER ")"

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
"""


"""
VARIABLE X SIZE(20) BOUNDED BY (10, 20) 
    MINIMIZE X**2
    CONSTRAINED BY (
        X[1] < X[2]   
    )
"""

# Create a custom Transformer to evaluate the arithmetic expressions
class CalcTransformer(Transformer):
    def add(self, operands):
        return operands[0] + operands[1]

    def sub(self, operands):
        return operands[0] - operands[1]

    def mul(self, operands):
        return operands[0] * operands[1]

    def div(self, operands):
        return operands[0] / operands[1]

    def number(self, value):
        return float(value[0])


# Create the Lark parser using the defined grammar and transformer
calculator = Lark(grammar, parser="lalr", transformer=CalcTransformer())


# Evaluate an arithmetic expression
def evaluate_expression(expression):
    tree = calculator.parse(expression)
    return tree


# Example usage
expression = "2 * (3 + 4)"
result = evaluate_expression(expression)
print(f"Expression: {expression}")
print(f"Result: {result}")


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