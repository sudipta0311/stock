"""Validate Render and GitHub deployment files without printing secret values."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INLINE_SECRET_PATTERNS = (
    re.compile(r"\bpostgres(?:ql)?://[^:\s/]+:[^@\s/]+@", re.IGNORECASE),
    re.compile(r"sk-", re.IGNORECASE),
    re.compile(r"api-key", re.IGNORECASE),
    re.compile(r"-----BEGIN", re.IGNORECASE),
)
_LONG_SECRET_VALUE = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")
_MODULE_REFERENCE = re.compile(r"\bwealth(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_MODULE_GUARD = re.compile(r"find_spec\(\s*['\"](wealth(?:\.[A-Za-z_][A-Za-z0-9_]*)+)['\"]\s*\)")
_E2E_DEV_AUTH_REFERENCE = re.compile(r"\bVITE_E2E_DEV_AUTH\b")
_REQUIRED_WORKFLOWS = {
    "alerts-weekly.yml",
    "ci.yml",
    "db-migrate.yml",
    "marketdata-abroad.yml",
    "marketdata-daily.yml",
    "marketdata-monthly.yml",
    "policy-weekly.yml",
    "research-weekly.yml",
    "sentinel-daily.yml",
}


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError):
        return None


def _walk_values(value: Any, parent_key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_values(child, str(key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child, parent_key)
    else:
        yield parent_key, value


def _secret_like_yaml(document: Any) -> bool:
    for key, value in _walk_values(document):
        if not isinstance(value, str):
            continue
        if any(pattern.search(value) for pattern in _INLINE_SECRET_PATTERNS):
            return True
        if key.casefold() == "value" and _LONG_SECRET_VALUE.fullmatch(value.strip()):
            return True
    return False


def _valid_cron(expression: str) -> bool:
    parts = expression.split()
    if len(parts) != 5:
        return False
    bounds = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
    for part, (minimum, maximum) in zip(parts, bounds, strict=True):
        for item in part.split(","):
            base, separator, step_text = item.partition("/")
            if separator and (not step_text.isdigit() or int(step_text) < 1):
                return False
            if base == "*":
                continue
            if "-" in base:
                endpoints = base.split("-")
                if len(endpoints) != 2 or not all(endpoint.isdigit() for endpoint in endpoints):
                    return False
                lower, upper = (int(endpoint) for endpoint in endpoints)
                if lower < minimum or upper > maximum or lower > upper:
                    return False
                continue
            if not base.isdigit() or not minimum <= int(base) <= maximum:
                return False
    return True


def _module_exists(root: Path, module: str) -> bool:
    relative = Path(*module.split("."))
    return (root / "src" / f"{relative}.py").is_file() or (
        (root / "src" / relative / "__init__.py").is_file()
    )


def _non_editable_project_install(command: str) -> bool:
    """Rules and copy are read from the checkout (rules/, copy/), not from site-packages."""
    return bool(re.search(r"pip install\b.*\.\[", command)) and not re.search(
        r"pip install\b.*\s-e\s", command
    )


def _e2e_dev_auth_issue(raw: str, relative: str) -> str | None:
    if _E2E_DEV_AUTH_REFERENCE.search(raw):
        return f"{relative}: the e2e dev-token auth switch must not appear in deploy config"
    return None


def _workflow_issues(root: Path, workflow: Path, document: Any) -> list[str]:
    relative = workflow.relative_to(root).as_posix()
    issues: list[str] = []
    raw = workflow.read_text(encoding="utf-8")
    e2e_auth_issue = _e2e_dev_auth_issue(raw, relative)
    if e2e_auth_issue:
        issues.append(e2e_auth_issue)
    if not isinstance(document, dict):
        return [f"{relative}: YAML could not be parsed"]

    permissions = document.get("permissions", {})
    if not isinstance(permissions, dict) or permissions.get("contents") != "read":
        issues.append(f"{relative}: permissions.contents must be read")
    concurrency = document.get("concurrency")
    if not isinstance(concurrency, dict) or not concurrency.get("group"):
        issues.append(f"{relative}: a concurrency group is required")

    jobs = document.get("jobs", {})
    if not isinstance(jobs, dict) or not jobs:
        issues.append(f"{relative}: at least one job is required")
        jobs = {}
    for name, job in jobs.items():
        if not isinstance(job, dict) or not str(job.get("timeout-minutes", "")).isdigit():
            issues.append(f"{relative}: job {name} needs timeout-minutes")

    # Workflows that only call a URL (the keep-awake ping) need no Python.
    if re.search(r"\bpython\b(?!-version)", raw):
        if not re.search(r"python-version:\s*['\"]?3\.12['\"]?", raw):
            issues.append(f"{relative}: Python 3.12 setup is required")
        if not re.search(r"cache:\s*pip\b", raw):
            issues.append(f"{relative}: setup-python must cache pip")
        if not re.search(r"pip install", raw):
            issues.append(f"{relative}: a pip install step is required")

    triggers = document.get("on", {})
    if isinstance(triggers, dict):
        schedules = triggers.get("schedule", [])
        if isinstance(schedules, list):
            for entry in schedules:
                cron = entry.get("cron") if isinstance(entry, dict) else None
                if not isinstance(cron, str) or not _valid_cron(cron):
                    issues.append(f"{relative}: invalid cron schedule")

    for line in raw.splitlines():
        if re.search(r"\b(?:echo|printf)\b.*\$\{\{\s*secrets\.", line, re.IGNORECASE):
            issues.append(f"{relative}: a command appears to print a GitHub secret")
            break

    if any(_non_editable_project_install(line) for line in raw.splitlines()):
        issues.append(f"{relative}: install the project with pip install -e")

    guarded = set(_MODULE_GUARD.findall(raw))
    modules = set(_MODULE_REFERENCE.findall(raw))
    for module in sorted(modules):
        if not _module_exists(root, module) and module not in guarded:
            issues.append(f"{relative}: missing module {module} needs an exit-78 guard")
    return issues


def _render_issues(root: Path) -> list[str]:
    path = root / "render.yaml"
    document = _load_yaml(path)
    relative = path.relative_to(root).as_posix()
    if not isinstance(document, dict):
        return [f"{relative}: YAML could not be parsed"]
    issues: list[str] = []
    e2e_auth_issue = _e2e_dev_auth_issue(path.read_text(encoding="utf-8"), relative)
    if e2e_auth_issue:
        issues.append(e2e_auth_issue)
    if _secret_like_yaml(document):
        issues.append(f"{relative}: a value matches a secret pattern")

    services = document.get("services", [])
    if not isinstance(services, list):
        return [*issues, f"{relative}: services must be a list"]
    by_name = {
        service.get("name"): service
        for service in services
        if isinstance(service, dict) and isinstance(service.get("name"), str)
    }
    static = by_name.get("wealth-pwa")
    api = by_name.get("wealth-api")
    if not isinstance(static, dict):
        issues.append(f"{relative}: wealth-pwa service is required")
    if not isinstance(api, dict):
        issues.append(f"{relative}: wealth-api service is required")
    if not isinstance(static, dict) or not isinstance(api, dict):
        return issues

    # Render declares static sites as `type: web` + `runtime: static`; they have no plan or
    # region (global CDN), and staticPublishPath is relative to the repo root.
    if static.get("type") != "web" or static.get("runtime") != "static":
        issues.append(f"{relative}: wealth-pwa must be a static site (type web, runtime static)")
    if "region" in static or "plan" in static:
        issues.append(f"{relative}: static sites must not define a region or plan")
    build_command = str(static.get("buildCommand", ""))
    if (
        static.get("staticPublishPath") != "apps/web/dist"
        or "cd apps/web" not in build_command
        or "npm ci" not in build_command
    ):
        issues.append(f"{relative}: wealth-pwa must build apps/web and publish apps/web/dist")
    routes = static.get("routes", [])
    if not any(
        isinstance(route, dict)
        and route.get("type") == "rewrite"
        and route.get("source") == "/*"
        and route.get("destination") == "/index.html"
        for route in routes
    ):
        issues.append(f"{relative}: the PWA fallback rewrite is required")
    rendered_headers = {
        (item.get("path"), item.get("name")): item.get("value")
        for item in static.get("headers", [])
        if isinstance(item, dict)
    }
    required_headers = {
        ("/sw.js", "Cache-Control"): "no-cache",
        ("/manifest.webmanifest", "Cache-Control"): "no-cache",
        ("/*", "X-Frame-Options"): "DENY",
        ("/*", "X-Content-Type-Options"): "nosniff",
        ("/*", "Referrer-Policy"): "no-referrer",
        ("/*", "Permissions-Policy"): "camera=(), microphone=(), geolocation=()",
    }
    if any(rendered_headers.get(key) != value for key, value in required_headers.items()):
        issues.append(f"{relative}: required PWA response headers are missing")

    static_env = {
        item.get("key"): item for item in static.get("envVars", []) if isinstance(item, dict)
    }
    required_static = {
        "VITE_API_URL",
        "VITE_AUTH_MODE",
        "VITE_OIDC_AUTHORITY",
        "VITE_OIDC_CLIENT_ID",
        "VITE_OIDC_AUDIENCE",
        "VITE_VAPID_PUBLIC_KEY",
    }
    if not required_static <= set(static_env):
        issues.append(f"{relative}: required PWA environment fields are missing")
    for key in required_static - {"VITE_AUTH_MODE"}:
        if key in static_env and static_env[key].get("sync") != "false":
            issues.append(f"{relative}: {key} must be entered in Render")
    if static_env.get("VITE_AUTH_MODE", {}).get("value") != "oidc":
        issues.append(f"{relative}: VITE_AUTH_MODE must be oidc")
    if "GOOGLE_CLIENT_SECRET" in static_env:
        issues.append(f"{relative}: the Google secret must not reach the PWA")
    if "APP_SESSION_SECRET" in static_env:
        issues.append(f"{relative}: the app-session signing key must not reach the PWA")

    if (
        api.get("type") != "web"
        or api.get("runtime") != "python"
        or api.get("plan") != "free"
        or api.get("region") != "frankfurt"
    ):
        issues.append(f"{relative}: wealth-api must be a free Python service in frankfurt")
    api_env = {item.get("key"): item for item in api.get("envVars", []) if isinstance(item, dict)}
    api_build_command = str(api.get("buildCommand", ""))
    if re.search(r"\bresearch\b", api_build_command, re.IGNORECASE):
        issues.append(f"{relative}: wealth-api must not install the research extra")
    if _non_editable_project_install(api_build_command):
        issues.append(f"{relative}: wealth-api must install the project with pip install -e")
    required_api = {
        "PYTHON_VERSION",
        "ENV",
        "AUTH_MODE",
        "MAX_UPLOAD_MB",
        "CDSL_PARSER_BETA",
        "DATABASE_URL",
        "AUTH_JWKS_URL",
        "AUTH_ISSUER",
        "AUTH_AUDIENCE",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "APP_SESSION_SECRET",
        "STATEMENT_PASSWORD_KEY",
        "ALLOWED_EMAILS",
        "ALLOWED_ORIGINS",
        "EMAIL_LINK_SECRET",
    }
    if not required_api <= set(api_env):
        issues.append(f"{relative}: required API environment fields are missing")
    manual_api = required_api - {
        "PYTHON_VERSION",
        "ENV",
        "AUTH_MODE",
        "MAX_UPLOAD_MB",
        "CDSL_PARSER_BETA",
    }
    for key in manual_api:
        if key in api_env and api_env[key].get("sync") != "false":
            issues.append(f"{relative}: {key} must be entered in Render")
    if api_env.get("PYTHON_VERSION", {}).get("value", "").split(".")[:2] != ["3", "12"]:
        issues.append(f"{relative}: Python 3.12 is required on Render")
    if any(key.startswith("FOUNDRY_") or key == "OWNER_DATABASE_URL" for key in api_env):
        issues.append(f"{relative}: LLM keys and OWNER_DATABASE_URL must stay off the API")
    return issues


def validate_deployment(root: Path = _REPOSITORY_ROOT) -> list[str]:
    """Return configuration problems without including any configured values."""
    issues = _render_issues(root)
    workflows_dir = root / ".github" / "workflows"
    workflows = sorted((*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")))
    if not workflows:
        issues.append(".github/workflows: at least one workflow is required")
    present_workflows = {workflow.name for workflow in workflows}
    for missing in sorted(_REQUIRED_WORKFLOWS - present_workflows):
        issues.append(f".github/workflows/{missing}: required workflow is missing")
    for workflow in workflows:
        document = _load_yaml(workflow)
        if _secret_like_yaml(document):
            issues.append(
                f"{workflow.relative_to(root).as_posix()}: a value matches a secret pattern"
            )
        issues.extend(_workflow_issues(root, workflow, document))
        if workflow.name == "ci.yml":
            raw = workflow.read_text(encoding="utf-8")
            required_ci = (
                "image: postgres:18",
                "CREATE ROLE wealth_owner LOGIN CREATEDB CREATEROLE BYPASSRLS",
                'pip install -e ".[api,importer,research,dev]"',
                "alembic upgrade head",
                "pytest",
                "node-version: 24",
                "npm run typecheck",
                "npm run lint",
                "npm test",
                "npm run build",
                "npm run budget",
                "wealth.api.export_openapi",
                "git diff --exit-code",
            )
            if any(text not in raw for text in required_ci):
                issues.append(
                    ".github/workflows/ci.yml: required Python, PWA, or schema checks are missing"
                )
    return issues


def main() -> int:
    issues = validate_deployment()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print("Deployment configuration OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
