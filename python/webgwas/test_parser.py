import pytest

from webgwas.parser import (
    ConstantNode,
    FieldNode,
    OperatorNode,
    ParserException,
    RPNParser,
)


@pytest.mark.parametrize("operator", ["AND", "OR", "EQ"])
def test_operator(operator):
    definition = f'"X" "Y" `{operator}`'
    parser = RPNParser(definition)
    expected_result = [FieldNode("X"), FieldNode("Y"), OperatorNode(operator)]
    for found, expected in zip(parser.parse(), expected_result):
        assert type(found) == type(expected)
        if isinstance(found, FieldNode):
            assert found.value == expected.value


@pytest.mark.parametrize("value", ["0", "01", "10"])
def test_constant(value):
    parsed = list(RPNParser(value).parse())
    assert len(parsed) == 1
    parsed_value = parsed[0]
    assert isinstance(parsed_value, ConstantNode)
    assert parsed_value.value == int(value)


@pytest.mark.parametrize("value", ["X", "XYZ", "XZY123"])
def test_field(value):
    parsed = list(RPNParser(f'"{value}"').parse())
    assert len(parsed) == 1
    parsed_value = parsed[0]
    assert isinstance(parsed_value, FieldNode)
    assert parsed_value.value == value


@pytest.mark.parametrize(
    "definition,value",
    [
        # Numeric
        ('"X" "Y" `ADD`', 3),
        ('"X" "Y" `SUB`', -1),
        ('"X" "Y" `MUL`', 2),
        ('"X" "Y" `DIV`', 0.5),
        ('"X" "Y" `GT`', False),
        ('"X" "Y" `GE`', False),
        ('"X" "Y" `LT`', True),
        ('"X" "Y" `LE`', True),
        ('"X" "Y" `EQ`', False),
        # Boolean
        ('"Z" `NOT`', False),
        ('"W" `NOT`', True),
        ('"Z" "W" `AND`', False),
        ('"Z" "W" `OR`', True),
        # Mixed
        ('"X" "Y" `LT` `NOT` "Z" `AND`', False),
        ('"X" "Y" `LT` `NOT` "Z" `OR`', True),
    ],
)
def test_apply_definition(definition, value):
    parser = RPNParser(definition)
    record = {"X": 1, "Y": 2, "Z": True, "W": False}
    assert parser.apply_definition(record) == value


@pytest.mark.parametrize("definition", ["X", "X `ADD`", "`1`", "`ABC`"])
def test_bad_definition(definition):
    with pytest.raises(ParserException):
        RPNParser(definition)
