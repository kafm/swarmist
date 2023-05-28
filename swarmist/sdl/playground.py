from lark import Lark


# Define the grammar using Lark's EBNF notation
grammar = """
    ?start: expr_with_optional_commas

    expr_with_optional_commas: expr? ("," expr)?

    expr: CNAME

    %import common.CNAME -> CNAME
    %import common.WS
    %ignore WS
"""


# Create the Lark parser using the defined grammar
parser = Lark(grammar, parser="lalr")


# Parse the input
def parse_input(input_str):
    parsed_tree = parser.parse(input_str)
    return parsed_tree


# Example usage
input_str = "a"
parsed_input = parse_input(input_str)
print(parsed_input)