import pytest

from webgwas.parser import ConstantNode, RPNParser, FieldNode, OperatorNode


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
