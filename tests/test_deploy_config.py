"""Deployment configuration checks remain runnable and reject bad cron values."""

from pathlib import Path

import yaml
from scripts.check_deploy_config import (
    _non_editable_project_install,
    _secret_like_yaml,
    _valid_cron,
    _workflow_issues,
    validate_deployment,
)


def test_current_render_and_github_files_pass_the_deployment_checker() -> None:
    assert validate_deployment() == []


def test_cron_checker_accepts_the_scheduled_job_expressions() -> None:
    assert _valid_cron("30 13 * * 1-5")
    assert _valid_cron("0 2-20/2 * * 6")
    assert _valid_cron("0 * * * 0")


def test_url_only_workflows_need_no_python_but_python_jobs_still_do(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    common = (
        "on:\n  schedule:\n    - cron: '4-59/10 * * * *'\n"
        "permissions:\n  contents: read\nconcurrency:\n  group: test\n"
    )
    ping = workflows / "ping.yml"
    ping.write_text(
        common + "jobs:\n  ping:\n    timeout-minutes: 4\n    steps:\n"
        "      - run: curl --fail https://api.example.test/healthz\n",
        encoding="utf-8",
    )
    job = workflows / "job.yml"
    job.write_text(
        common + "jobs:\n  job:\n    timeout-minutes: 4\n    steps:\n"
        "      - run: python -m wealth.alerts.daily\n",
        encoding="utf-8",
    )

    assert _workflow_issues(tmp_path, ping, yaml.safe_load(ping.read_text("utf-8"))) == []
    assert "Python 3.12 setup is required" in " ".join(
        _workflow_issues(tmp_path, job, yaml.safe_load(job.read_text("utf-8")))
    )


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


def test_project_installs_must_be_editable_so_rule_files_resolve() -> None:
    # A plain install copies the code into site-packages, where rules/*.yaml can't be found.
    assert _non_editable_project_install('pip install --no-cache-dir ".[api,importer]"')
    assert not _non_editable_project_install('pip install --no-cache-dir -e ".[api,importer]"')
    assert not _non_editable_project_install("python -m pip install --upgrade pip")
