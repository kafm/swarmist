#?var: NAME "=" math_expr    -> assign_var
#parameters_expr? init_population_expr

#TODO CONSTRAINTS
# constraints_expr?
# ?constraints_expr: "subject"i "to"i constraint+
# "var"i key size_expr? bounds_expr  -> set_var
grammar = """
    ?start: "search"i "(" space_expr ")" "using"i "(" strategy_expr ")" "until"i "(" termination_expr ")" -> search
    ?space_expr: variables objective_function_expr constraints_expr? -> space
    ?objective_function_expr: "minimize"i math_expr -> minimize
        | "maximize"i math_expr -> maximize
    ?variables: variable+ -> set_vars
    ?variable: "var"i key size_expr? bounds_expr  -> var
    ?constraints_expr: "subject"i "to"i "(" constraint+ ")" ("with"i "coefficient"i float)?   -> build_constraints
    ?constraints:  constraint+  -> set_constraints
    ?constraint: math_expr "<" math_expr -> lt_constraint
        | math_expr "<=" math_expr -> le_constraint
        | math_expr ">" math_expr -> gt_constraint
        | math_expr ">=" math_expr -> ge_constraint
        | math_expr "=" math_expr -> eq_constraint
        | math_expr "!=" math_expr -> ne_constraint
    ?strategy_expr: parameters_expr? init_population_expr update_expr+ -> build_strategy
    ?parameters_expr: "parameters"i "(" parameter+ ")"
    ?parameter: key "=" math_expr bounds_expr? -> set_parameter
    ?init_population_expr: "population"i size_expr "init"i init_pos_expr topology_expr? -> init
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
    ?bounds_expr: "bounded"i "by"i "(" bound "," bound ")" -> bounds
    ?bound: value -> bound 
        | "-" value -> neg_bound
    ?update_expr: selection_expr "(" update_method ")" -> update
    ?update_method: recombination_expr "update"i "(" update_vars ")" ("when"i where)? -> recombine_pos
        | "update"i "(" update_vars ")" ("when"i where)? -> replace_all_pos
    ?update_vars:  update_var+ -> update_pos
    ?update_var: key "=" math_expr -> set_update_var
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
    ?recombination_expr: "using"i recombination_method
        | "reset"i "agent"i -> init_random_recombination
    ?recombination_method: "binomial"i "recombination"i "with"i "probability"i probability  -> binomial_recombination
        | "exponential"i "recombination"i "with"i "probability"i probability    -> exponential_recombination
        | "recombination"i "with"i "probability"i probability   -> with_probability_recombination
        | "random"i "recombination"i size_expr  -> random_recombination
    ?references_expr: "swarm_best"i "(" integer? ")"    -> swarm_best
        | "swarm_worst"i "(" integer? ")"   -> swarm_worst
        | "all"i "(" ")"  -> swarm_all
        | "neighborhood"i "(" ")"  -> swarm_neighborhood
        | "pick_random"i "(" reference_unique_prop? integer? reference_replace_prop? ")" -> swarm_pick_random
        | "pick_roulette"i "(" reference_unique_prop? integer? reference_replace_prop? ")"   -> swarm_pick_roulette
        | "rand_to_best" "(" "with"i "probability"i probability ")" -> swarm_rand_to_best
        | "current_to_best"i "(" "with"i "probability"i probability ")" -> swarm_current_to_best
        | "param"i "(" key ")" -> get_parameter
        | agent_prop
    ?agent_prop: sortable_agent_prop
        | "best"i   -> agent_best
        | "pos"i    -> agent_pos
        | "delta"i  -> agent_delta
    ?sortable_agent_prop: "trials"i -> agent_trials
        | "fit"i    -> agent_fit
        | "improved"i   -> agent_improved
    ?reference_unique_prop: "unique"i -> true
    ?reference_replace_prop: "with"i "replacement"i -> true
    ?probability: value -> probability
    ?size_expr: "size"i "(" integer ")"
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
    ?math_term: atom
        | math_term "*" atom  -> mul
        | math_term "/" atom  -> div
        | math_term "%" atom  -> mod
        | math_term "**" atom -> pow
        | math_term "//" atom -> floordiv
    ?atom: "-" atom         -> neg
        | key -> get_var
        | "pi" "(" ")"               -> pi
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
        | "if_then"i "(" condition_expr "," math_expr ","  math_expr ")"    -> if_then
        | references_expr
        | random_expr
        | value -> value_to_lambda
        | bool -> value_to_lambda
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