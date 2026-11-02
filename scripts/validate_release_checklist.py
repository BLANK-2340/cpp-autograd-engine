#!/usr/bin/env python3
"""Strict offline validation for the portfolio release checklist."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = ROOT / "docs" / "release_checklist.json"
DEFAULT_PORTFOLIO = ROOT / "docs" / "portfolio.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "RELEASE_CHECKLIST.md"
FACETS = ["setup", "demo", "checks", "evidence", "limitations"]
PROJECT_KEYS = {"id", *FACETS}


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def _line(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{context} must be one non-empty line")
    return value


def _normalized(value: str) -> str:
    return " ".join(value.replace("`", "").split())


def validate_document(checklist: Any, portfolio: Any, markdown: str) -> None:
    if not isinstance(checklist, dict) or not isinstance(portfolio, dict):
        raise ValueError("documents must be objects")
    _keys(checklist, {"schema_version", "generated_from", "required_facets", "projects"}, "checklist")
    if checklist["schema_version"] != 1 or isinstance(checklist["schema_version"], bool):
        raise ValueError("unsupported schema_version")
    if checklist["generated_from"] != "docs/portfolio.json":
        raise ValueError("unexpected generated_from path")
    if checklist["required_facets"] != FACETS:
        raise ValueError("required facets differ")
    source_projects = portfolio.get("projects")
    projects = checklist["projects"]
    if not isinstance(source_projects, list) or not isinstance(projects, list):
        raise ValueError("projects must be arrays")
    source_by_id = {project.get("id"): project for project in source_projects if isinstance(project, dict)}
    if len(source_by_id) != len(source_projects) or len(projects) != len(source_projects):
        raise ValueError("project coverage differs")
    seen: set[str] = set()
    normalized_markdown = _normalized(markdown)
    for index, project in enumerate(projects):
        if not isinstance(project, dict):
            raise ValueError(f"projects[{index}] must be an object")
        _keys(project, PROJECT_KEYS, f"projects[{index}]")
        values = {key: _line(project[key], f"projects[{index}].{key}") for key in PROJECT_KEYS}
        project_id = values["id"]
        if project_id in seen or project_id not in source_by_id:
            raise ValueError(f"unexpected or duplicate project: {project_id}")
        source = source_by_id[project_id]
        expected = {"setup": source["showcase_url"], "demo": source["showcase_url"],
                    "checks": source["verify_command"], "evidence": source["evidence_url"],
                    "limitations": source["limitation"]}
        for facet in FACETS:
            if values[facet] != expected[facet]:
                raise ValueError(f"{facet} differs for {project_id}")
            if _normalized(values[facet]) not in normalized_markdown:
                raise ValueError(f"{facet} is absent from Markdown for {project_id}")
        seen.add(project_id)
    if seen != set(source_by_id):
        raise ValueError("project set is incomplete")


def _load(path: Path) -> tuple[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        return raw, json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def validate_files() -> None:
    raw, checklist = _load(DEFAULT_CHECKLIST)
    _, portfolio = _load(DEFAULT_PORTFOLIO)
    validate_document(checklist, portfolio, DEFAULT_MARKDOWN.read_text(encoding="utf-8"))
    if raw != json.dumps(checklist, indent=2, sort_keys=True, allow_nan=False) + "\n":
        raise ValueError("release checklist is not canonical JSON")


def run_self_test() -> None:
    _, checklist = _load(DEFAULT_CHECKLIST)
    _, portfolio = _load(DEFAULT_PORTFOLIO)
    markdown = DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    mutations = []
    missing = copy.deepcopy(checklist); del missing["projects"][0]["demo"]; mutations.append(missing)
    command = copy.deepcopy(checklist); command["projects"][1]["checks"] = "echo pass"; mutations.append(command)
    evidence = copy.deepcopy(checklist); evidence["projects"][2]["evidence"] = "https://example.invalid"; mutations.append(evidence)
    duplicate = copy.deepcopy(checklist); duplicate["projects"][3] = copy.deepcopy(duplicate["projects"][0]); mutations.append(duplicate)
    limit = copy.deepcopy(checklist); limit["projects"][3]["limitations"] = "No limitations."; mutations.append(limit)
    for mutation in mutations:
        try:
            validate_document(mutation, portfolio, markdown)
        except ValueError:
            continue
        raise AssertionError("corrupted release checklist was accepted")
    print(f"PASS: rejected {len(mutations)} release-checklist corruptions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_self_test() if args.self_test else validate_files()
        if not args.self_test:
            print(f"validated {DEFAULT_CHECKLIST} against portfolio and Markdown")
    except (OSError, KeyError, ValueError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
