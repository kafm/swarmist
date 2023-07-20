grammar = """
    ?start: "search"i "(" space_expr ")" "using"i "(" strategy_expr ")" "until"i "(" termination_expr ")" -> search
    ?space_expr: variables objective_function_expr constraints_expr? -> space
    ?objective_function_expr: "minimize"i math_expr -> minimize
        | "maximize"i math_expr -> maximize
    ?variables: variable+ -> set_vars
    ?variable: "var"i key size_expr? bounds_expr  -> var
    ?constraints_expr: "subject"i "to"i "(" constraints ")" ("with"i "coefficient"i float)?   -> build_constraints
    ?constraints:  constraint+  -> set_constraints
    ?constraint: math_expr "<=" math_expr -> le_constraint
        | math_expr ">=" math_expr -> ge_constraint
        | math_expr "=" math_expr -> eq_constraint
    ?strategy_expr: parameters_expr? init_population_expr update_expr+ tune_expr? -> build_strategy
    ?tune_expr: "tune"i "auto"i "until"i "(" "generation"i "=" integer ")"  -> set_tunning_config
    ?parameters_expr: parameter*
    ?parameter: "param"i key "=" math_expr bounds_expr? -> set_parameter
        | "param"i key "=" (auto_int | auto_float) -> set_auto_parameter
    ?init_population_expr: "population"i (size_expr | auto_size_expr) "init"i init_pos_expr topology_expr? -> init
    ?init_pos_expr: "random"i  "(" random_props_without_size? ")"  -> init_random
        | "random_uniform"i "(" random_props_without_size? ")" -> init_random_uniform
        | "random_normal"i "(" random_props_without_size? ")"  -> init_random_uniform
        | "random_lognormal"i "(" random_props_without_size? ")"  -> init_random_lognormal
        | "random_skewnormal"i "(" random_props_without_size? ")"  -> init_random_skewnormal
        | "random_cauchy"i "(" random_props_without_size? ")"  -> init_random_cauchy
        | "random_levy"i "(" random_props_without_size? ")"    -> init_random_levy
        | "random_beta"i "(" random_props_without_size? ")"  -> init_random_beta
        | "random_exponential"i "(" random_props_without_size? ")"  -> init_random_exponential
        | "random_rayleigh"i "(" random_props_without_size? ")" -> init_random_rayleigh
        | "random_weibull"i "(" random_props_without_size? ")"  -> init_random_weibull
    ?topology_expr: "with"i "topology"i topology
    ?topology: "gbest"i -> gbest_topology
        | "lbest"i size_expr? -> lbest_topology
    ?auto_int: "auto"i "int"i bounds_expr -> auto_integer
    ?auto_float: "auto"i "float"i bounds_expr -> auto_float
    ?bounds_expr: "bounded"i "by"i "(" bound "," bound ")" -> bounds
    ?bound: value -> bound 
        | "-" value -> neg_bound
    ?update_expr: selection_expr "(" update_method ")" -> update
    ?update_method: "using"i recombination_method "update"i "(" update_vars ")" ("when"i where)? -> recombine_pos
        | "update"i "(" update_vars ")" ("when"i where)? -> replace_all_pos
        | "init"i init_pos_expr -> reset_pos
    ?update_vars: update_var+   -> update_pos
    ?update_var: "pos"i "=" math_expr -> set_pos_var
        | key "=" math_expr -> set_update_var
    ?selection_expr: "select"i selection_size order_by? -> all_selection
        | "select"i selection_size "where"i where order_by? -> filter_selection
        | "with"i "roulette"i "select"i selection_size -> roulette_selection
        | "with"i "random"i "select"i selection_size -> random_selection
        | "with"i "probability"i probability "select"i selection_size -> probabilistic_selection 
    ?selection_size: size_expr -> selection_size
        | "all"i -> selection_size
    ?where: where_condition
        | where "and"i where_condition -> and_
        | where "or"i where_condition -> or_ 
    ?where_condition: sortable_agent_prop "<" math_expr -> lt
        | sortable_agent_prop "<=" math_expr -> le
        | sortable_agent_prop ">" math_expr -> gt
        | sortable_agent_prop ">=" math_expr -> ge
        | sortable_agent_prop "=" math_expr -> eq
        | sortable_agent_prop "!=" math_expr -> ne
        | "(" where ")"
    ?order_by: "order"i "by"i sortable_agent_prop asc_desc? -> order_by
    ?asc_desc: "asc"i
        | "desc"i  -> reverse_order 
    ?recombination_method: "binomial"i "recombination"i "with"i "probability"i probability  -> binomial_recombination
        | "exponential"i "recombination"i "with"i "probability"i probability    -> exponential_recombination
        | "recombination"i "with"i "probability"i probability   -> with_probability_recombination
        | "random"i "recombination"i (size_expr | auto_size_expr)  -> random_recombination
    ?references_expr: "swarm_best"i "(" integer? ")"    -> swarm_best
        | "swarm_worst"i "(" integer? ")"   -> swarm_worst
        | "all"i "(" ")"  -> swarm_all
        | "neighborhood"i "(" ")"  -> swarm_neighborhood
        | "pick_random"i "(" pick_args ")" -> swarm_pick_random
        | "pick_roulette"i "(" pick_args ")"   -> swarm_pick_roulette
        | "random_pos"i "(" ")" -> swarm_random_pos
        | "rand_to_best"i "(" "with"i "probability"i probability ")" -> swarm_rand_to_best
        | "current_to_best"i "(" "with"i "probability"i probability ")" -> swarm_current_to_best
        | param
        | agent_prop
    ?pick_args: pick_unique_prop? pick_size_prop? pick_replace_prop? -> swarm_pick_args
    ?pick_unique_prop: "unique"i -> swarm_pick_unique_prop
    ?pick_size_prop: integer -> swarm_pick_size_prop
    ?pick_replace_prop: "with"i "replacement"i -> swarm_pick_replacement_prop
    ?func_expr: "apply"i "(" math_expr "," apply_def ")" -> apply_func
        | "map"i "(" math_expr "," apply_def ")" -> map_func
        | "reduce"i "(" math_expr "," reduce_def ("," math_expr)? ")" -> reduce_func
        | "filter"i "(" math_expr "," filter_def ")" -> filter_func
    ?apply_def: "(" key ")" "=>" math_expr -> func_def
    ?reduce_def: "(" key "," key ")" "=>" math_expr -> func_def
    ?filter_def: "(" key ")" "=>" conditions_expr -> func_def
    ?agent_prop: sortable_agent_prop
        | "best"i   -> agent_best
        | "pos"i    -> agent_pos
        | "delta"i  -> agent_delta
    ?sortable_agent_prop: "trials"i -> agent_trials
        | "fit"i    -> agent_fit
        | "improved"i   -> agent_improved
    ?probability: (value | param) -> probability
    ?size_expr: "size"i "(" (integer | param) ")"
    ?auto_size_expr: "size"i "(" auto_int ")"
    ?termination_expr: termination_condition+ -> stop_condition
    ?termination_condition: "evaluations"i "=" integer -> set_max_evals
        | "generation"i "=" integer -> set_max_gen
        | "fitness"i "=" float -> set_min_fit
    ?random_expr: "random"i  "(" random_props? ")"  -> random
        | "random_uniform"i "(" random_props? ")" -> random_uniform
        | "random_normal"i "(" random_props? ")"  -> random_normal
        | "random_lognormal"i "(" random_props? ")"  -> random_lognormal
        | "random_skewnormal"i "(" random_props? ")"  -> random_skewnormal
        | "random_cauchy"i "(" random_props? ")"  -> random_cauchy
        | "random_levy"i "(" random_props? ")"    -> random_levy
        | "random_beta"i "(" random_props? ")"  -> random_beta
        | "random_exponential"i "(" random_props? ")"  -> random_exponential
        | "random_rayleigh"i "(" random_props? ")" -> random_rayleigh
        | "random_weibull"i "(" random_props? ")"  -> random_weibull
    ?random_props: random_prop ("," random_prop)* -> random_props
    ?random_props_without_size: random_prop_without_size ("," random_prop_without_size)* -> random_props
    ?random_prop: random_prop_without_size
        | "size"i "=" math_expr -> random_size
    ?random_prop_without_size: "loc"i "=" math_expr -> random_loc
        | "scale"i "=" math_expr -> random_scale
        | "shape"i "=" math_expr -> random_shape
        | "alpha"i "=" math_expr -> random_alpha
        | "beta"i "=" math_expr -> random_beta
        | "low"i "=" math_expr -> random_low
        | "high"i "=" math_expr -> random_high
    ?conditions_expr: condition_expr 
        | conditions_expr "and"i condition_expr -> and_
        | conditions_expr "or"i condition_expr -> or_
    ?condition_expr: math_expr "<" math_expr -> lt
        | math_expr "<=" math_expr -> le
        | math_expr ">" math_expr -> gt
        | math_expr ">=" math_expr -> ge
        | math_expr "=" math_expr -> eq
        | math_expr "!=" math_expr -> ne
        | "(" conditions_expr ")"
    ?math_expr: math_term
        | math_expr "+" math_term   -> add
        | math_expr "-" math_term   -> sub
    ?math_term: math_factor
        | math_term "*" math_factor  -> mul
        | math_term "/" math_factor  -> div
        | math_term "//" math_factor -> floordiv
    ?math_factor: math_factor "**" atom -> pow
        | math_factor "%" atom  -> mod
        | "-" atom         -> neg
        | "(" math_expr ")"
        |  atom
    ?atom: key -> get_var
        | key "." key -> get_var
        | "pi"i "(" ")"               -> pi
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
        | "avg"i "(" math_expr ("," math_expr)? ")" -> avg
        | "diff"i "(" math_expr "," math_expr ")" -> diff
        | "repeat"i "(" math_expr "," math_expr ")" -> repeat
        | func_expr
        | "count"i "(" math_expr ")" -> count
        | "if_then"i "(" conditions_expr "," math_expr ","  math_expr ")"    -> if_then
        | references_expr
        | random_expr
        | value -> value_to_lambda
        | bool -> value_to_lambda
    ?param: "param"i "(" key ")" -> get_parameter
    ?key: NAME -> string
    ?value: float 
        | integer
    ?integer: INT -> integer
    ?float: NUMBER -> number
    ?bool: "true"i -> true
        | "false"i -> false 
    
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.INT
    %import common.WS
    %ignore WS
"""