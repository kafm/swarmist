#?var: NAME "=" math_expr    -> assign_var
grammar = """
    ?start: strategy_expr
    ?strategy_expr: parameters_expr? update_expr
    ?parameters_expr: "parameters"i "(" parameter+ ")"
    ?parameter: key "=" math_expr bounds_expr? -> set_parameter
    ?bounds_expr: "bounded"i "by"i "(" value "," value ")" -> bounds
    ?update_expr: recombination_expr? "update"i "(" update_vars ")"
    ?update_vars:  update_var+ -> update_pos
    ?update_var: key "=" math_expr -> set_update_var
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
        | "trials"i -> agent_trials
        | "best"i   -> agent_best
        | "pos"i    -> agent_pos
        | "fit"i    -> agent_fit
        | "delta"i  -> agent_delta
        | "improved"i   -> agent_improved
        | "param"i "(" key ")" -> get_parameter
    ?reference_unique_prop: "unique"i -> true
    ?reference_replace_prop: "with"i "replacement"i -> true
    ?probability: value -> probability
    ?size_expr: "size"i "(" integer ")"
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
    ?random_prop: "loc"i "=" math_expr -> random_loc
        | "scale"i "=" math_expr -> random_scale
        | "shape"i "=" math_expr -> random_shape
        | "alpha"i "=" math_expr -> random_alpha
        | "beta"i "=" math_expr -> random_beta
        | "low"i "=" math_expr -> random_low
        | "high"i "=" math_expr -> random_high
        | "size"i "=" math_expr -> random_size
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
        | NAME -> get_var
        | NAME "TODO" "[" INT "]" -> var_index
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
        | references_expr
        | random_expr
        | value -> value_to_lambda
    ?key: NAME -> string
    ?value: float | integer
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