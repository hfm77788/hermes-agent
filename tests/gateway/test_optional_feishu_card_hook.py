"""Tests for optional hermes-feishu-card hook logging."""

import logging

from gateway.run import _log_hermes_feishu_card_failure


def test_absent_optional_feishu_card_package_is_silent(caplog):
    caplog.set_level(logging.WARNING)
    caplog.clear()
    exc = ModuleNotFoundError(
        "No module named 'hermes_feishu_card'",
        name="hermes_feishu_card",
    )

    _log_hermes_feishu_card_failure(exc)

    assert not caplog.records


def test_feishu_card_internal_dependency_failure_is_logged(caplog):
    caplog.set_level(logging.WARNING)
    caplog.clear()
    exc = ModuleNotFoundError(
        "No module named 'unexpected_dependency'",
        name="unexpected_dependency",
    )

    _log_hermes_feishu_card_failure(exc)

    assert any("hermes-feishu-card" in record.message for record in caplog.records)
