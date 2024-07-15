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

    def apply_definition(
        self, record: dict[str, int | float | bool]
    ) -> int | float | bool:
        stack = list()
        for node in self.parsed_definition:
            match node:
                case FieldNode(value=value):
                    stack.append(record[value])
                case ConstantNode(value=value):
                    stack.append(value)
                case OperatorNode.ADD:
                    stack.append(stack.pop() + stack.pop())
                case OperatorNode.SUB:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x - y)
                case OperatorNode.MUL:
                    stack.append(stack.pop() * stack.pop())
                case OperatorNode.DIV:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x / y)
                case OperatorNode.AND:
                    y = stack.pop()
                    x = stack.pop()
                    verify_boolean(x)
                    verify_boolean(y)
                    stack.append(x and y)
                case OperatorNode.OR:
                    y = stack.pop()
                    x = stack.pop()
                    verify_boolean(x)
                    verify_boolean(y)
                    stack.append(x or y)
                case OperatorNode.NOT:
                    x = stack.pop()
                    verify_boolean(x)
                    stack.append(not x)
                case OperatorNode.GT:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x > y)
                case OperatorNode.GE:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x >= y)
                case OperatorNode.LT:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x < y)
                case OperatorNode.LE:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x <= y)
                case OperatorNode.EQ:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(x == y)
                case _:
                    raise ParserException(
                        f"Unknown node type {type(node)} in '{self.raw_definition}'"
                    )
        return stack.pop()


def verify_boolean(value: int | float | bool) -> None:
    is_bool = isinstance(value, bool) or value in (0, 1)
    if not is_bool:
        raise ValueError(f"{value} is not a bool")
