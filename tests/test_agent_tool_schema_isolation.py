"""Provider tool schemas must match local validation and remain immutable externally."""

import pytest

from evalforge.agent_tools import (
    TOOL_REGISTRY,
    anthropic_tool_schemas,
    openai_tool_schemas,
    valid_tool_arguments,
)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_provider_schemas_reject_extra_arguments_and_match_registry(provider):
    exported = anthropic_tool_schemas() if provider == "anthropic" else openai_tool_schemas()
    assert len(exported) == len(TOOL_REGISTRY)
    for entry in exported:
        name = entry["name"] if provider == "anthropic" else entry["function"]["name"]
        schema = entry["input_schema"] if provider == "anthropic" else entry["function"]["parameters"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == TOOL_REGISTRY[name]["parameters"]["required"]
        assert schema["properties"] == TOOL_REGISTRY[name]["parameters"]["properties"]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_mutating_exported_schema_does_not_change_grading_contract(provider):
    exported = anthropic_tool_schemas() if provider == "anthropic" else openai_tool_schemas()
    weather = next(
        entry for entry in exported
        if (entry["name"] if provider == "anthropic" else entry["function"]["name"]) == "get_weather"
    )
    schema = weather["input_schema"] if provider == "anthropic" else weather["function"]["parameters"]
    schema["required"].clear()
    schema["properties"]["city"]["type"] = "integer"
    schema["additionalProperties"] = True

    assert valid_tool_arguments("get_weather", {"city": "Boston"})
    assert not valid_tool_arguments("get_weather", {})
    assert not valid_tool_arguments("get_weather", {"city": 42})
    assert not valid_tool_arguments("get_weather", {"city": "Boston", "unexpected": "x"})
    assert TOOL_REGISTRY["get_weather"]["parameters"]["additionalProperties"] is False

    fresh_anthropic = next(
        item for item in anthropic_tool_schemas() if item["name"] == "get_weather"
    )["input_schema"]
    fresh_openai = next(
        item for item in openai_tool_schemas() if item["function"]["name"] == "get_weather"
    )["function"]["parameters"]
    for fresh in (fresh_anthropic, fresh_openai):
        assert fresh["required"] == ["city"]
        assert fresh["properties"]["city"]["type"] == "string"
        assert fresh["additionalProperties"] is False
