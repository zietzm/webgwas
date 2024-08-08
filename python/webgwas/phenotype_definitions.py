from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeInt


class NodeType(StrEnum):
    BOOL = "BOOL"
    REAL = "REAL"
    ANY = "ANY"
    # Categorical, etc. intentionally omitted here. May implement later.


class Operator(BaseModel):
    id: int
    name: Literal[
        "ROOT",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "AND",
        "OR",
        "NOT",
        "GT",
        "GE",
        "LT",
        "LE",
        "EQ",
    ]
    arity: NonNegativeInt
    input_type: NodeType
    output_type: NodeType


class OperatorType(Enum):
    ROOT = Operator(
        id=0, name="ROOT", arity=1, input_type=NodeType.ANY, output_type=NodeType.ANY
    )
    ADD = Operator(
        id=1, name="ADD", arity=2, input_type=NodeType.REAL, output_type=NodeType.REAL
    )
    SUB = Operator(
        id=2, name="SUB", arity=2, input_type=NodeType.REAL, output_type=NodeType.REAL
    )
    MUL = Operator(
        id=3, name="MUL", arity=2, input_type=NodeType.REAL, output_type=NodeType.REAL
    )
    DIV = Operator(
        id=4, name="DIV", arity=2, input_type=NodeType.REAL, output_type=NodeType.REAL
    )
    AND = Operator(
        id=5, name="AND", arity=2, input_type=NodeType.BOOL, output_type=NodeType.BOOL
    )
    OR = Operator(
        id=6, name="OR", arity=2, input_type=NodeType.BOOL, output_type=NodeType.BOOL
    )
    NOT = Operator(
        id=7, name="NOT", arity=1, input_type=NodeType.BOOL, output_type=NodeType.BOOL
    )
    GT = Operator(
        id=8, name="GT", arity=2, input_type=NodeType.REAL, output_type=NodeType.BOOL
    )
    GE = Operator(
        id=9, name="GE", arity=2, input_type=NodeType.REAL, output_type=NodeType.BOOL
    )
    LT = Operator(
        id=10, name="LT", arity=2, input_type=NodeType.REAL, output_type=NodeType.BOOL
    )
    LE = Operator(
        id=11, name="LE", arity=2, input_type=NodeType.REAL, output_type=NodeType.BOOL
    )
    EQ = Operator(
        id=12, name="EQ", arity=2, input_type=NodeType.REAL, output_type=NodeType.BOOL
    )


class Field(BaseModel):
    id: int | None = None
    code: str
    name: str | None = None
    type: NodeType | None = None


class Constant(BaseModel):
    value: float | bool
    type: NodeType

    @classmethod
    def from_str(cls, string: str) -> Constant:
        type, value_str = string.split(":")
        match type:
            case "BOOL":
                value = value_str == "True"
            case "REAL":
                value = float(value_str)
            case _:
                raise ValueError(f"Unknown constant type {type}")
        return Constant(value=value, type=NodeType(type))


class KnowledgeBase(BaseModel):
    operators: list[Operator]
    fields: list[Field]

    @classmethod
    def default(cls, fields: list[Field]) -> KnowledgeBase:
        operators = [
            OperatorType.__members__[name].value for name in OperatorType.__members__
        ]
        return cls(operators=operators, fields=fields)

    def query_field(self, code: str) -> Field:
        results = [field for field in self.fields if field.code == code]
        if len(results) == 0:
            raise ValueError(f"Field {code} not found")
        elif len(results) > 1:
            raise ValueError(f"Field {code} is ambiguous")
        return results[0]


Node = Field | Operator | Constant


def parse_string_definition(definition: str) -> list[Node]:
    items = definition.split(" ")
    results = list()
    for item in items:
        match item[0]:
            case '"':
                if item[-1] != '"':
                    raise ValueError(f"Invalid field name {item}")
                node = Field(code=item[1:-1])  # check whether valid later
                results.append(node)
            case "`":
                if item[-1] != "`":
                    raise ValueError(f"Invalid operator name {item}")
                node = OperatorType[item[1:-1]].value
                results.append(node)
            case "<":
                if item[-1] != ">":
                    raise ValueError(f"Invalid constant name {item}")
                node = Constant.from_str(item[1:-1])
                results.append(node)
            case _:
                raise ValueError(f"Unknown item {item}")
    return results


def validate_nodes(nodes: list[Node], knowledge_base: KnowledgeBase) -> list[Node]:
    results = list()
    for node in nodes:
        match node:
            case Field(code=code):
                node = knowledge_base.query_field(code)
                results.append(node)
            case Operator() | Constant():
                results.append(node)
    return results


def type_check_nodes(nodes: list[Node]) -> None:
    """Check that validated nodes make a valid definition."""
    stack = list()
    for node in nodes:
        match node:
            case Field() | Constant():
                stack.append(node)
            case Operator(arity=arity, input_type=input_type, output_type=output_type):
                for _ in range(arity):
                    try:
                        item = stack.pop()
                    except IndexError:
                        raise ValueError(
                            f"Operator {node.name} expects {arity} operands"
                        )
                    if not (isinstance(item, Constant) or isinstance(item, Field)):
                        raise ValueError(
                            f"Operator {node.name} expects {arity} operands"
                        )
                    if item.type != input_type:
                        raise ValueError(
                            f"Operator {node.name} expects {input_type} operand"
                        )
                result_node = Constant(value=0, type=output_type)
                stack.append(result_node)
    if len(stack) != 1:
        raise ValueError(f"Invalid definition Stack: {[s for s in stack]}")
    return


def validate_item(item: Any, expected_type: NodeType, arity: int, name: str) -> None:
    if not isinstance(item, pd.Series):
        raise ValueError(f"Operator {name} expects {arity} operands")
    if item.dtype != expected_type:
        raise ValueError(f"Operator {name} expects {expected_type} operand")


def apply_definition_pandas(nodes: list[Node], df: pd.DataFrame) -> pd.Series:
    """Apply a definition to a pandas DataFrame."""
    stack = list()
    for node in nodes:
        match node:
            case Field(code=code):
                stack.append(df[code])
            case Operator(name=name, narity=arity, input_type=input_type):
                match arity:
                    case 1:
                        item = stack.pop()
                        validate_item(item, input_type, arity, node.name)
                        match name:
                            case "ROOT":
                                stack.append(item)
                            case "ADD":
                                stack.append(~item)
                            case _:
                                raise ValueError(
                                    f"Unknown operator {name} with arity 1"
                                )
                    case 2:
                        item2 = stack.pop()
                        validate_item(item2, input_type, arity, node.name)
                        item1 = stack.pop()
                        validate_item(item1, input_type, arity, node.name)
                        match name:
                            case "ADD":
                                stack.append(item1 + item2)
                            case "SUB":
                                stack.append(item1 - item2)
                            case "MUL":
                                stack.append(item1 * item2)
                            case "DIV":
                                stack.append(item1 / item2)
                            case "AND":
                                stack.append(np.minimum(item1, item2))
                            case "OR":
                                stack.append(np.maximum(item1, item2))
                            case "NOT":
                                stack.append(1 - item1)
                            case "GT":
                                stack.append(item1 > item2)
                            case "GE":
                                stack.append(item1 >= item2)
                            case "LT":
                                stack.append(item1 < item2)
                            case "LE":
                                stack.append(item1 <= item2)
                            case "EQ":
                                stack.append(item1 == item2)
                            case _:
                                raise ValueError(
                                    f"Unknown operator {name} with arity 2"
                                )
                    case _:
                        raise ValueError(f"Unknown operator {name} with arity {arity}")
            case Constant(value=value, type=type):
                match type:
                    case NodeType.BOOL:
                        stack.append(bool(value))
                    case NodeType.REAL:
                        stack.append(float(value))
                    case _:
                        raise ValueError(f"Unknown constant type {type}")
    if len(stack) != 1:
        raise ValueError(f"Invalid definition Stack: {[s for s in stack]}")
    return stack[0]
