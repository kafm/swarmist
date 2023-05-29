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


    # ?strategy_expr: "population"i size_expr init_expr 
    # ?init_expr: "init"i init_pos_expr topology_expr? 
    # ?init_pos_expr: rand_prefix rand_distribution "(" NUMBER? ("," NUMBER)? ("," NUMBER)? ")"  
    #     | rand
    
    
        # | "random.beta"i  ( "(" alpha_beta ( "," alpha_beta? ")" )?    -> init_random_beta
        # | "random.exponential"i ("(" scale ")")?  -> init_random_exponential
        # | "random.rayleigh"i ("(" scale ")")?  -> init_random_rayleigh
        # | "random.weibull"i ( "(" shape_scale ( "," shape_scale? ")" )?  -> init_random_weibull
grammar = """
    ?start: parameters_expr
    ?start_bck: init_population_expr parameters_expr?
    ?parameters_expr: "parameters"i "(" parameter+ ")"
    ?parameter: key "=" math_expr bounds_expr? -> set_parameter
    ?init_population_expr: population_size_expr init_expr
    ?population_size_expr: "population"i size_expr  -> population_size 
    ?init_expr: "init"i init_pos_expr topology_expr? 
    ?init_pos_expr: "random"i   -> init_random
        | "random.uniform"i -> init_random_uniform
        | "random.normal"i ( "(" loc_scale ("," loc_scale)? ")" )?  -> init_random_normal
        | "random.lognormal"i ( "(" loc_scale ("," loc_scale)? ")" )?   -> init_random_lognormal
        | "random.skewnormal"i ( "(" loc_scale_shape ("," loc_scale_shape)? ")" )?  -> init_random_skewnormal
        | "random.cauchy"i ( "(" loc_scale ("," loc_scale)? ")" )?  -> init_random_cauchy
        | "random.levy"i ( "(" loc_scale ("," loc_scale)? ")" )?    -> init_random_levy
        | "random.beta"i ( "(" alpha_beta ("," alpha_beta)? ")" )?  -> init_random_beta
        | "random.exponential"i ( "(" scale ")" )?  -> init_random_exponential
        | "random.rayleigh"i ( "(" scale ")" )? -> init_random_rayleigh
        | "random.weibull"i ( "(" shape_scale ("," shape_scale)? ")" )?  -> init_random_weibull
    ?topology_expr: "with"i "topology"i topology
    ?topology: "gbest"i -> gbest_topology
        | "lbest"i size_expr? -> lbest_topology
    ?agent_prop: "index"i 
        | "trials"i 
        | "best"i 
        | "pos"i 
        | "fit"i 
        | "delta"i 
        | "improved"i
    ?loc_scale_shape: loc | scale | shape
    ?loc_scale:  loc | scale
    ?shape_scale:  shape | scale
    ?alpha_beta: alpha | beta
    ?loc: "loc"i "=" NUMBER -> loc
    ?scale: "scale"i "=" NUMBER -> scale
    ?shape: "shape"i "=" NUMBER -> shape
    ?alpha: "alpha"i "=" NUMBER -> alpha
    ?beta: "beta"i "=" NUMBER -> beta
    ?bounds_expr: "bounded"i "by"i "(" value "," value ")" -> bounds
    ?size_expr: "size"i "(" INT ")" -> size
    ?termination_expr: "until"i "(" termination_condition ("," termination_condition)* ")"
    ?termination_condition: "evaluations"i "=" atom -> set_max_evals
        | "generation"i "=" atom -> set_max_gen
        | "fitness"i "=" atom -> set_min_fit
    ?probability: atom -> probability
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
    ?atom: (NUMBER | INT)   -> number_lambda
        | "-" atom         -> neg
        | NAME -> string
        | NAME "TODO" "[" INT "]" -> var_index
        | "pi"               -> pi
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
    ?key: NAME -> string
    ?value: NUMBER  -> number
        | INT   -> int
   
    
    
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
"""