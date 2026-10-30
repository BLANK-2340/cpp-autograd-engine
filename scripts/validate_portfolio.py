#!/usr/bin/env python3
"""Strict offline validation for the machine-readable portfolio index."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs" / "portfolio.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "PORTFOLIO_INDEX.md"
PROJECT_KEYS = {
    "id", "repository", "repository_url", "showcase_url", "evidence_revision",
    "evidence_url", "verify_command", "recorded_result", "limitation",
}
EXPECTED = {
    "cpp-autograd-engine": ("BLANK-2340/cpp-autograd-engine",
                            "bash scripts/verify_clean_checkout.sh"),
    "visiontrack-nn": ("BLANK-2340/VisionTrack-NN", "./scripts/verify_showcase.sh"),
    "cuda-kernel-lab": ("BLANK-2340/cuda-kernel-lab", "./scripts/verify_showcase.sh"),
    "pendulum-wave-simulation": ("BLANK-2340/Pendulum-Wave-simulation",
                                 "./scripts/verify_showcase.sh"),
}


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{context} must be one non-empty line")
    return value


def _normalized(value: str) -> str:
    return " ".join(value.replace("`", "").split())


def validate_document(document: Any, markdown: str) -> None:
    if not isinstance(document, dict):
        raise ValueError("portfolio must be an object")
    _keys(document, {"schema_version", "generated_from", "projects"}, "portfolio")
    if isinstance(document["schema_version"], bool) or document["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if document["generated_from"] != "docs/PORTFOLIO_INDEX.md":
        raise ValueError("unexpected generated_from path")
    projects = document["projects"]
    if not isinstance(projects, list) or len(projects) != len(EXPECTED):
        raise ValueError("projects must contain the four prescribed entries")

    ids: set[str] = set()
    repositories: set[str] = set()
    revisions: set[str] = set()
    normalized_markdown = _normalized(markdown)
    for index, project in enumerate(projects):
        if not isinstance(project, dict):
            raise ValueError(f"projects[{index}] must be an object")
        _keys(project, PROJECT_KEYS, f"projects[{index}]")
        values = {key: _text(project[key], f"projects[{index}].{key}")
                  for key in PROJECT_KEYS}
        project_id = values["id"]
        if project_id not in EXPECTED:
            raise ValueError(f"unexpected project id: {project_id}")
        repository, command = EXPECTED[project_id]
        if values["repository"] != repository or values["verify_command"] != command:
            raise ValueError(f"identity or command differs for {project_id}")
        revision = values["evidence_revision"]
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"invalid evidence revision for {project_id}")
        base = f"https://github.com/{repository}"
        expected_urls = {
            "repository_url": base,
            "showcase_url": f"{base}/blob/main/docs/SHOWCASE.md",
            "evidence_url": f"{base}/commit/{revision}",
        }
        for key, expected in expected_urls.items():
            if values[key] != expected:
                raise ValueError(f"{key} differs for {project_id}")
            if expected not in markdown:
                raise ValueError(f"{key} is absent from Markdown for {project_id}")
        for key in ("verify_command", "recorded_result", "limitation"):
            if _normalized(values[key]) not in normalized_markdown:
                raise ValueError(f"{key} is out of sync for {project_id}")
        ids.add(project_id)
        repositories.add(repository)
        revisions.add(revision)
    if len(ids) != len(projects) or len(repositories) != len(projects):
        raise ValueError("project ids and repositories must be unique")
    if len(revisions) != len(projects):
        raise ValueError("evidence revisions must be unique")
    if ids != set(EXPECTED):
        raise ValueError("project set is incomplete")


def validate_files(json_path: Path = DEFAULT_JSON,
                   markdown_path: Path = DEFAULT_MARKDOWN) -> None:
    raw = json_path.read_text(encoding="utf-8")
    try:
        document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON: {error}") from error
    validate_document(document, markdown_path.read_text(encoding="utf-8"))
    canonical = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if raw != canonical:
        raise ValueError("portfolio is not canonical JSON")


def run_self_test() -> None:
    document = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    markdown = DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    mutations = []
    extra_key = copy.deepcopy(document)
    extra_key["unexpected"] = True
    mutations.append(extra_key)
    bad_revision = copy.deepcopy(document)
    bad_revision["projects"][0]["evidence_revision"] = "not-a-commit"
    mutations.append(bad_revision)
    duplicate_repository = copy.deepcopy(document)
    duplicate_repository["projects"][1]["repository"] = document["projects"][0]["repository"]
    mutations.append(duplicate_repository)
    stale_result = copy.deepcopy(document)
    stale_result["projects"][2]["recorded_result"] = "invented speedup"
    mutations.append(stale_result)
    for mutation in mutations:
        try:
            validate_document(mutation, markdown)
        except ValueError:
            continue
        raise AssertionError("corrupted portfolio was accepted")
    print(f"PASS: rejected {len(mutations)} portfolio corruptions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        else:
            validate_files(args.json, args.markdown)
            print(f"validated {args.json} against {args.markdown}")
    except (OSError, ValueError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
