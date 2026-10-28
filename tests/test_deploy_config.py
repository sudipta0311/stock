"""Deployment configuration checks remain runnable and reject bad cron values."""

from scripts.check_deploy_config import _secret_like_yaml, _valid_cron, validate_deployment


def test_current_render_and_github_files_pass_the_deployment_checker() -> None:
    assert validate_deployment() == []


def test_cron_checker_accepts_the_scheduled_job_expressions() -> None:
    assert _valid_cron("30 13 * * 1-5")
    assert _valid_cron("0 2-20/2 * * 6")
    assert _valid_cron("0 * * * 0")


def test_cron_checker_rejects_invalid_fields() -> None:
    assert not _valid_cron("60 13 * * 1-5")
    assert not _valid_cron("0 25 * * 1-5")
    assert not _valid_cron("0 2-20/0 * * 6")
    assert not _valid_cron("0 0 * *")


def test_yaml_secret_scan_rejects_secret_shaped_values_without_echoing_them() -> None:
    assert _secret_like_yaml({"value": "-----BEGIN PRIVATE KEY-----"})
    assert _secret_like_yaml({"database": "postgresql://user:password@host/db"})
    assert _secret_like_yaml({"value": "A" * 40})
    assert not _secret_like_yaml({"value": "3.12.7", "mode": "oidc"})
