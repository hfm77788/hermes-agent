"""Regression tests for Feishu channel_skill_bindings YAML bridging."""

from unittest.mock import patch

from gateway.config import Platform


def test_feishu_channel_skill_bindings_bridged_from_gateway_yaml(tmp_path):
    """gateway.platforms.feishu.extra.channel_skill_bindings reaches PlatformConfig.extra."""
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """
gateway:
  platforms:
    feishu:
      extra:
        channel_skill_bindings:
          - id: oc_9841d208db7edafcd9c61da0420b0059
            skills:
              - colleague-hou-fangming
              - feishu-group-response-rules
""".lstrip(),
        encoding="utf-8",
    )

    with patch("gateway.config.get_hermes_home", return_value=tmp_path):
        from gateway.config import load_gateway_config

        with patch.dict(
            "os.environ",
            {
                "FEISHU_APP_ID": "cli_test",
                "FEISHU_APP_SECRET": "secret_test",
            },
            clear=False,
        ):
            config = load_gateway_config()

    feishu_config = config.platforms.get(Platform.FEISHU)
    assert feishu_config is not None
    assert feishu_config.extra["channel_skill_bindings"] == [
        {
            "id": "oc_9841d208db7edafcd9c61da0420b0059",
            "skills": ["colleague-hou-fangming", "feishu-group-response-rules"],
        }
    ]
