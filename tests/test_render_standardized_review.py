from analysis.render_standardized_review import resolve_target_size


def test_review_target_size_inherits_config_unless_cli_overrides() -> None:
    config: dict[str, object] = {"target_size": 1536}
    assert resolve_target_size(None, config) == 1536
    assert resolve_target_size(1024, config) == 1024
    assert resolve_target_size(None, {}) == 768
