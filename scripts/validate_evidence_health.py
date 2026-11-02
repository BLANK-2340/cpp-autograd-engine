#!/usr/bin/env python3
"""Validate the recorded portfolio evidence-health snapshot offline."""

from __future__ import annotations

import argparse
import copy
from datetime import date, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEALTH = ROOT / "docs" / "evidence_health.json"
DEFAULT_PORTFOLIO = ROOT / "docs" / "portfolio.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "EVIDENCE_HEALTH.md"
TOP_KEYS = {"checked_on", "projects", "schema_version", "source"}
PROJECT_KEYS = {
    "ci", "commit_url", "evidence_revision", "evidence_tree", "project_id",
    "repository", "showcase_blob", "showcase_url",
}
CI_KEYS = {
    "completed_at", "conclusion", "event", "head_revision", "run_id",
    "run_url", "status", "workflow_name",
}


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{context} must be one non-empty line")
    return value


def _sha(value: Any, context: str) -> str:
    text = _text(value, context)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{context} must be a full lowercase Git SHA")
    return text


def _load(path: Path) -> tuple[Any, str]:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}"))), raw
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def validate_document(health: Any, portfolio: Any, markdown: str) -> None:
    if not isinstance(health, dict) or not isinstance(portfolio, dict):
        raise ValueError("health and portfolio documents must be objects")
    _keys(health, TOP_KEYS, "health")
    if isinstance(health["schema_version"], bool) or health["schema_version"] != 1:
        raise ValueError("unsupported health schema_version")
    if health["source"] != "docs/portfolio.json":
        raise ValueError("unexpected health source")
    checked_on = _text(health["checked_on"], "checked_on")
    try:
        date.fromisoformat(checked_on)
    except ValueError as error:
        raise ValueError("checked_on must be an ISO calendar date") from error

    portfolio_projects = portfolio.get("projects")
    health_projects = health["projects"]
    if not isinstance(portfolio_projects, list) or not isinstance(health_projects, list):
        raise ValueError("project collections must be arrays")
    expected = {project["id"]: project for project in portfolio_projects
                if isinstance(project, dict) and "id" in project}
    if len(expected) != len(portfolio_projects) or len(health_projects) != len(expected):
        raise ValueError("health must cover each portfolio project exactly once")

    seen: set[str] = set()
    for index, project in enumerate(health_projects):
        if not isinstance(project, dict):
            raise ValueError(f"projects[{index}] must be an object")
        _keys(project, PROJECT_KEYS, f"projects[{index}]")
        project_id = _text(project["project_id"], f"projects[{index}].project_id")
        if project_id not in expected or project_id in seen:
            raise ValueError(f"unexpected or duplicate project id: {project_id}")
        source = expected[project_id]
        repository = _text(project["repository"], f"{project_id}.repository")
        revision = _sha(project["evidence_revision"], f"{project_id}.evidence_revision")
        _sha(project["evidence_tree"], f"{project_id}.evidence_tree")
        _sha(project["showcase_blob"], f"{project_id}.showcase_blob")
        if repository != source["repository"] or revision != source["evidence_revision"]:
            raise ValueError(f"portfolio identity differs for {project_id}")
        base = f"https://github.com/{repository}"
        expected_urls = {
            "commit_url": f"{base}/commit/{revision}",
            "showcase_url": f"{base}/blob/main/docs/SHOWCASE.md",
        }
        for key, expected_url in expected_urls.items():
            if project[key] != expected_url or expected_url not in markdown:
                raise ValueError(f"{key} differs or is undocumented for {project_id}")

        ci = project["ci"]
        if not isinstance(ci, dict):
            raise ValueError(f"{project_id}.ci must be an object")
        _keys(ci, CI_KEYS, f"{project_id}.ci")
        if ci["head_revision"] != revision:
            raise ValueError(f"CI head differs for {project_id}")
        if ci["event"] != "push" or ci["status"] != "completed":
            raise ValueError(f"CI event/status differs for {project_id}")
        if ci["conclusion"] not in {"success", "failure", "cancelled", "timed_out"}:
            raise ValueError(f"unsupported CI conclusion for {project_id}")
        run_id = ci["run_id"]
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            raise ValueError(f"invalid CI run id for {project_id}")
        expected_run_url = f"{base}/actions/runs/{run_id}"
        if ci["run_url"] != expected_run_url or expected_run_url not in markdown:
            raise ValueError(f"CI URL differs or is undocumented for {project_id}")
        _text(ci["workflow_name"], f"{project_id}.ci.workflow_name")
        try:
            completed_at = datetime.fromisoformat(
                _text(ci["completed_at"], f"{project_id}.ci.completed_at").replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"invalid CI completion time for {project_id}") from error
        if completed_at.tzinfo is None:
            raise ValueError(f"CI completion time lacks a timezone for {project_id}")
        seen.add(project_id)
    if seen != set(expected):
        raise ValueError("health project set is incomplete")


def validate_files(health_path: Path = DEFAULT_HEALTH,
                   portfolio_path: Path = DEFAULT_PORTFOLIO,
                   markdown_path: Path = DEFAULT_MARKDOWN) -> None:
    health, raw = _load(health_path)
    portfolio, _ = _load(portfolio_path)
    validate_document(health, portfolio, markdown_path.read_text(encoding="utf-8"))
    canonical = json.dumps(health, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if raw != canonical:
        raise ValueError("health snapshot is not canonical JSON")


def run_self_test() -> None:
    health, _ = _load(DEFAULT_HEALTH)
    portfolio, _ = _load(DEFAULT_PORTFOLIO)
    markdown = DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    mutations = []
    extra_key = copy.deepcopy(health)
    extra_key["unexpected"] = True
    mutations.append(extra_key)
    wrong_revision = copy.deepcopy(health)
    wrong_revision["projects"][0]["evidence_revision"] = "0" * 40
    mutations.append(wrong_revision)
    wrong_tree = copy.deepcopy(health)
    wrong_tree["projects"][1]["evidence_tree"] = "not-a-tree"
    mutations.append(wrong_tree)
    wrong_ci_head = copy.deepcopy(health)
    wrong_ci_head["projects"][2]["ci"]["head_revision"] = "f" * 40
    mutations.append(wrong_ci_head)
    wrong_run = copy.deepcopy(health)
    wrong_run["projects"][3]["ci"]["run_id"] += 1
    mutations.append(wrong_run)
    for mutation in mutations:
        try:
            validate_document(mutation, portfolio, markdown)
        except ValueError:
            continue
        raise AssertionError("corrupted health snapshot was accepted")
    print(f"PASS: rejected {len(mutations)} evidence-health corruptions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        else:
            validate_files(args.health, args.portfolio, args.markdown)
            print(f"validated {args.health} against {args.portfolio}")
    except (OSError, ValueError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
