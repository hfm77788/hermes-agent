from tools import mcp_tool as core
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_registration as registration
from tools.mcp_tool_schema import mcp_prefixed_tool_name
from tools.mcp_tool_scope import _server_key


def _reset():
    core._parallel_safe_servers.clear()
    core._parallel_explicit_readonly_tools.clear()
    core._mcp_tool_server_names.clear()
    core._tool_read_only_hints.clear()


def test_explicit_readonly_tool_is_parallel_safe_without_hint():
    _reset()
    try:
        name = mcp_prefixed_tool_name("docs", "search")
        registration._track_mcp_tool_server(name, "docs")
        key = _server_key("docs")
        core._parallel_explicit_readonly_tools[key] = {"search"}
        assert discovery.is_mcp_tool_parallel_safe(name) is True
    finally:
        _reset()


def test_explicit_allowlist_does_not_admit_write_sibling():
    _reset()
    try:
        read_name = mcp_prefixed_tool_name("docs", "search")
        write_name = mcp_prefixed_tool_name("docs", "write")
        registration._track_mcp_tool_server(read_name, "docs")
        registration._track_mcp_tool_server(write_name, "docs")
        key = _server_key("docs")
        core._parallel_explicit_readonly_tools[key] = {"search"}
        assert discovery.is_mcp_tool_parallel_safe(read_name) is True
        assert discovery.is_mcp_tool_parallel_safe(write_name) is False
    finally:
        _reset()
