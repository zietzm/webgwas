import enum
from collections.abc import Generator


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
    def __init__(self, definition: str) -> None:
        self.definition = definition
        self.state = ParserState.START
        self.stack = list()

    def parse(self) -> Generator[DefinitionNode]:
        current = ""
        for char in self.definition + "\0":
            match self.state:
                case ParserState.START:
                    match char:
                        case "`":
                            self.state = ParserState.OPERATOR
                        case '"':
                            self.state = ParserState.FIELD
                        case num if num.isdigit():
                            self.state = ParserState.CONSTANT
                            current += char
                        case " ":
                            pass
                        case "\0":
                            return
                        case _:
                            raise ValueError(
                                f"Unknown char '{char}' in '{self.definition}', START state"
                            )
                case ParserState.FIELD:
                    match char:
                        case '"':
                            yield FieldNode(current)
                            self.state = ParserState.START
                            current = ""
                        case c if c.isalnum():
                            current += char
                        case _:
                            raise ValueError(
                                f"Unknown char '{char}' in '{self.definition}', FIELD state"
                            )
                case ParserState.OPERATOR:
                    match char:
                        case "`":
                            try:
                                operator = OperatorNode(current)
                            except ValueError:
                                raise ValueError(
                                    f"No matching OperatorNode found for '{current}'"
                                )
                            yield operator
                            self.state = ParserState.START
                            current = ""
                        case c if c.isalpha():
                            current += char
                        case _:
                            raise ValueError(
                                f"Unknown char '{char}' in '{self.definition}', OPERATOR state"
                            )
                case ParserState.CONSTANT:
                    match char:
                        case num if num.isdigit():
                            current += char
                        case " " | "\0":
                            value = float(current) if "." in current else int(current)
                            yield ConstantNode(value)
                            self.state = ParserState.START
                            current = ""
                        case _:
                            raise ValueError(
                                f"Unknown char '{char}' in '{self.definition}', CONSTANT state"
                            )
