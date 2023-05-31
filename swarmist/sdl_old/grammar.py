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
#     ?selection_expr: "select"i quantity_expr ("with"i selection_method_expr)?("where"i where_expr)* ("order"i order_expr)?
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
    ?start: strategy_expr
    ?strategy_expr: parameters_expr? init_population_expr update_expr+
    ?update_expr: selection_expr "(" ("using"i recombination_expr)? ")"
    ?recombination_expr: "binomial"i "recombination"i ("with"i "probability"i probability) -> binomial_recombination
        | "exponential"i "recombination"i ("with"i "probability"i probability) -> exponential_recombination
        | "random"i "recombination"i size_expr  -> random_recombination
        | "random"i "recombination"i "with"i "probability"i probability    -> with_probability_recombination
        | "restart"i "random"i    -> init_random_recombination 
    ?do_update_expr: "update"i "(" (key "=" update_val_expr)+ ")" ")" "when_expr"?
    ?update_val_expr: update_val_term
        | update_val_expr "+" update_val_term -> update_val_add
        | update_val_expr "-" update_val_term -> update_val_sub
    ?update_val_term: update_val_atom
        | update_val_term "*" update_val_atom -> update_val_mul
        | update_val_term "/" update_val_atom -> update_val_div
        | update_val_term "%" update_val_atom  -> update_val_mod
        | update_val_term "**" update_val_atom -> update_val_pow
        | update_val_term "//" update_val_atom -> update_val_floordiv
    ?update_val_atom: "(" update_val_expr ")"
        | references_expr  
        | reference_expr
        | aggregation_expr
        | key  -> ctx_get_var
        | "param"i "(" key ")" -> ctx_get_param
        | rand_update_ctx_expr
        | "if_then"i "(" if_condition_expr "," update_val_expr ","  update_val_expr ")"    -> ctx_if_then
        | value
    ?if_condition_expr: if_condition_term
        | if_condition_expr "and"i if_condition_expr -> and_op
        | if_condition_expr "or"i if_condition_expr -> or_op
        | "(" if_condition_expr ")"
    ?if_condition_term: update_val_expr comp_expr (value | bool) -> comp_op
    ?op: "+" | "-" | "*" | "/" | "^"
    ?aggregation_expr: "sum"i "(" update_val_expr ")"   -> ctx_sum
        | "avg"i "(" update_val_expr ")"    -> ctx_avg
        | "min"i "(" update_val_expr ")"    -> ctx_min
        | "max"i "(" update_val_expr ")"    -> ctx_max
        | "reduce"i "(" update_val_expr ", " update_val_expr ")"    -> ctx_reduce
    ?references_expr: "swarm_best"i ("(" INT")")?    -> ctx_swarm_best
        | "swarm_worst"i ("(" INT ")")?   -> ctx_swarm_worst
        | "neighborhood"i   -> ctx_swarm_neighborhood
        | "pick_random"i ("(" "unique"i? INT? ("with"i "replacement"i)? ")")?  -> ctx_pick_random
        | "pick_roulette"i ("(" "unique"i? INT? ("with"i "replacement"i)? ")")?    -> ctx_pick_roulette
        | "rand_to_best" "(" ("with"i "probability"i NUMBER)? ")"   ->ctx_rand_to_bes
        | "current_to_best"i "(" ("with"i "probability" NUMBER)? ")"    -> ctx_current_to_best
    ?reference_expr: "index"i -> ctx_agent_index
        | "trials"i -> ctx_agent_trials
        | "best"i   -> ctx_agent_best
        | "pos"i    -> ctx_agent_pos
        | "fit"i    -> ctx_agent_fit
        | "delta"i  -> ctx_agent_delta
        | "improved"i   -> ctx_agent_improved
    ?rand_update_ctx_expr: "random"i   -> ctx_random
        | "random.uniform"i ( "(" low_high "=" update_val_expr ("," low_high "=" update_val_expr)? ")" )? -> ctx_random_uniform
        | "random.normal"i ( "(" loc_scale "=" update_val_expr ("," loc_scale "=" update_val_expr)? ")" )?  -> ctx_random_normal
        | "random.lognormal"i ( "(" loc_scale "=" update_val_expr ("," loc_scale "=" update_val_expr)? ")" )?   -> ctx_random_lognormal
        | "random.skewnormal"i ( "(" loc_scale_shape "=" update_val_expr ("," loc_scale_shape "=" update_val_expr)? ")" )?  -> ctx_random_skewnormal
        | "random.cauchy"i ( "(" loc_scale "=" update_val_expr ("," loc_scale "=" update_val_expr)? ")" )?  -> ctx_random_cauchy
        | "random.levy"i ( "(" loc_scale "=" update_val_expr ("," loc_scale "=" update_val_expr)? ")" )?    -> ctx_random_levy
        | "random.beta"i ( "(" alpha_beta "=" update_val_expr ("," alpha_beta "=" update_val_expr)? ")" )?  -> ctx_random_beta
        | "random.exponential"i ( "(" scale "=" update_val_expr ")" )?  -> ctx_random_exponential
        | "random.rayleigh"i ( "(" scale "=" update_val_expr ")" )? -> ctx_random_rayleigh
        | "random.weibull"i ( "(" shape_scale "=" update_val_expr ("," shape_scale "=" update_val_expr)? ")" )?  -> ctx_random_weibull
    ?low_high: "low"i -> low
        | "high"i   -> high
    ?selection_expr: ("with"i selection_method_expr)? "select"i ("where"i condition_expr)? order_expr? limit? -> selection  
    ?selection_method_expr: "roulette"i -> roulette_selection
        | "random"i -> random_selection
        | "probability"i probability  -> probabilistic_selection
    ?probability: value -> probability
    ?condition_expr: condition_term
        | condition_expr "and"i condition_expr -> and_op
        | condition_expr "or"i condition_expr -> or_op
        | "(" condition_expr ")"
    ?condition_term: agent_prop comp_expr (value | bool) -> comp_op
    ?comp_expr: "<" -> lt
        | "<="  -> le
        | ">"   -> gt
        | ">="  -> ge 
        | "="   -> eq
        | "!="  -> ne
    ?order_expr: "order"i "by"i agent_prop order? -> order_by 
    ?order: "asc"i  -> asc   
        | "desc"i    -> desc  
    ?limit: "limit"i INT -> limit
    ?agent_prop: "index"i -> agent_index
        | "trials"i -> agent_trials
        | "best"i   -> agent_best
        | "pos"i    -> agent_pos
        | "fit"i    -> agent_fit
        | "delta"i  -> agent_delta
        | "improved"i   -> agent_improved
    ?parameters_expr: "parameters"i "(" parameter+ ")"
    ?parameter: key "=" math_expr bounds_expr? -> set_parameter
    ?init_population_expr: population_size_expr init_expr
    ?population_size_expr: "population"i size_expr  -> population_size 
    ?init_expr: "init"i init_pos_expr topology_expr? 
    ?init_pos_expr: "random"i   -> init_random
        | "random.uniform"i -> init_random_uniform
        | "random.normal"i ( "(" loc_scale "=" value ("," loc_scale "=" value)? ")" )?  -> init_random_normal
        | "random.lognormal"i ( "(" loc_scale "=" value ("," loc_scale "=" value)? ")" )?   -> init_random_lognormal
        | "random.skewnormal"i ( "(" loc_scale_shape "=" value ("," loc_scale_shape "=" value)? ")" )?  -> init_random_skewnormal
        | "random.cauchy"i ( "(" loc_scale "=" value ("," loc_scale "=" value)? ")" )?  -> init_random_cauchy
        | "random.levy"i ( "(" loc_scale "=" value ("," loc_scale "=" value)? ")" )?    -> init_random_levy
        | "random.beta"i ( "(" alpha_beta "=" value ("," alpha_beta "=" value)? ")" )?  -> init_random_beta
        | "random.exponential"i ( "(" scale "=" value ")" )?  -> init_random_exponential
        | "random.rayleigh"i ( "(" scale "=" value ")" )? -> init_random_rayleigh
        | "random.weibull"i ( "(" shape_scale "=" value ("," shape_scale)? ")" )?  -> init_random_weibull
    ?loc_scale: loc | scale
    ?shape_scale: shape | scale
    ?loc_scale_shape: loc_scale | shape
    ?alpha_beta: alpha | beta
    ?loc: "loc"i -> loc
    ?scale: "scale"i -> scale
    ?shape: "shape"i -> shape
    ?alpha: "alpha"i -> alpha
    ?beta: "beta"i -> beta
    ?topology_expr: "with"i "topology"i topology
    ?topology: "gbest"i -> gbest_topology
        | "lbest"i size_expr? -> lbest_topology
    ?bounds_expr: "bounded"i "by"i "(" value "," value ")" -> bounds
    ?size_expr: "size"i "(" INT ")" -> size
    ?termination_expr: "until"i "(" termination_condition ("," termination_condition)* ")"
    ?termination_condition: "evaluations"i "=" atom -> set_max_evals
        | "generation"i "=" atom -> set_max_gen
        | "fitness"i "=" atom -> set_min_fit
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
        | INT   -> integer
    ?bool: "true"i -> true
        | "false"i -> false
   
    
    
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
"""