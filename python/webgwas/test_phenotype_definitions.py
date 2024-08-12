import pytest

from webgwas.phenotype_definitions import (
    Constant,
    Field,
    NodeType,
    OperatorType,
    parse_string_definition,
    type_check_nodes,
)


@pytest.mark.parametrize(
    "definition,expected",
    [
        (
            '"X" "Y" `ADD`',
            [
                Field(code="X"),
                Field(code="Y"),
                OperatorType.ADD.value,
            ],
        ),
        (
            '"X" "Y" `ADD` "Z" `ADD`',
            [
                Field(code="X"),
                Field(code="Y"),
                OperatorType.ADD.value,
                Field(code="Z"),
                OperatorType.ADD.value,
            ],
        ),
        (
            '<REAL:0.5> <REAL:1.0> `ADD` "X" `ADD`',
            [
                Constant(value=0.5, type=NodeType.REAL),
                Constant(value=1.0, type=NodeType.REAL),
                OperatorType.ADD.value,
                Field(code="X"),
                OperatorType.ADD.value,
            ],
        ),
        (
            '<BOOL:True> <BOOL:False> `AND` "X" `AND`',
            [
                Constant(value=True, type=NodeType.BOOL),
                Constant(value=False, type=NodeType.BOOL),
                OperatorType.AND.value,
                Field(code="X"),
                OperatorType.AND.value,
            ],
        ),
    ],
)
def test_parse_string_definition(definition, expected):
    nodes = parse_string_definition(definition)
    assert nodes == expected


@pytest.mark.parametrize(
    "nodes",
    [
        [
            Field(code="X", type=NodeType.REAL),
            Field(code="Y", type=NodeType.REAL),
            OperatorType.ADD.value,
        ],
        [
            Field(code="X", type=NodeType.REAL),
            Field(code="Y", type=NodeType.REAL),
            OperatorType.ADD.value,
            Field(code="Z", type=NodeType.REAL),
            OperatorType.ADD.value,
        ],
        [
            Field(code="A", type=NodeType.REAL),
            Field(code="B", type=NodeType.REAL),
            OperatorType.ADD.value,
            Field(code="C", type=NodeType.REAL),
            Field(code="D", type=NodeType.REAL),
            OperatorType.ADD.value,
            OperatorType.DIV.value,
        ],
        [
            Constant(value=True, type=NodeType.BOOL),
            Constant(value=False, type=NodeType.BOOL),
            OperatorType.OR.value,
        ],
        [
            Constant(value=1.5, type=NodeType.REAL),
            Constant(value=2.0, type=NodeType.REAL),
            OperatorType.GT.value,
            Constant(value=True, type=NodeType.BOOL),
            OperatorType.AND.value,
        ],
    ],
)
def test_type_check_nodes(nodes):
    type_check_nodes(nodes)
