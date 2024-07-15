import enum
from collections.abc import Generator


class ParserException(Exception):
    pass


class OperatorNode(enum.StrEnum):
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"
    EQ = "EQ"


class FieldNode:
    def __init__(self, value: str) -> None:
        self.value = value


class ConstantNode:
    def __init__(self, value: int | float) -> None:
        self.value = value


type DefinitionNode = OperatorNode | FieldNode | ConstantNode


class ParserState(enum.Enum):
    START = 1
    FIELD = 2
    OPERATOR = 3
    CONSTANT = 4


class RPNParser:
    def __init__(self, raw_definition: str) -> None:
        self.raw_definition = raw_definition
        self.parsed_definition = list(self.parse())

    def parse(self) -> Generator[DefinitionNode]:
        current = ""
        state = ParserState.START
        for char in self.raw_definition + "\0":
            match state:
                case ParserState.START:
                    match char:
                        case "`":
                            state = ParserState.OPERATOR
                        case '"':
                            state = ParserState.FIELD
                        case num if num.isdigit():
                            state = ParserState.CONSTANT
                            current += char
                        case " ":
                            pass
                        case "\0":
                            return
                        case _:
                            raise ParserException(
                                f"Unknown char '{char}' in '{self.raw_definition}', START state"
                            )
                case ParserState.FIELD:
                    match char:
                        case '"':
                            yield FieldNode(current)
                            state = ParserState.START
                            current = ""
                        case c if c.isalnum():
                            current += char
                        case _:
                            raise ParserException(
                                f"Unknown char '{char}' in '{self.raw_definition}', FIELD state"
                            )
                case ParserState.OPERATOR:
                    match char:
                        case "`":
                            try:
                                operator = OperatorNode(current)
                            except ValueError:
                                raise ParserException(
                                    f"No matching OperatorNode found for '{current}'"
                                )
                            yield operator
                            state = ParserState.START
                            current = ""
                        case c if c.isalpha():
                            current += char
                        case _:
                            raise ParserException(
                                f"Unknown char '{char}' in '{self.raw_definition}', OPERATOR state"
                            )
                case ParserState.CONSTANT:
                    match char:
                        case num if num.isdigit():
                            current += char
                        case " " | "\0":
                            value = float(current) if "." in current else int(current)
                            yield ConstantNode(value)
                            state = ParserState.START
                            current = ""
                        case _:
                            raise ParserException(
                                f"Unknown char '{char}' in '{self.raw_definition}', CONSTANT state"
                            )
