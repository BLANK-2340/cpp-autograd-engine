#!/usr/bin/env python3
"""Verify the hash-bound portfolio release summary and its source validators."""

from __future__ import annotations

import argparse
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "docs" / "release_summary.json"
TOP_KEYS = {"artifacts", "generated_on", "schema_version", "verification_command"}
ARTIFACT_KEYS = {"path", "sha256", "validator"}
EXPECTED = [
    ("docs/portfolio.json", "scripts/validate_portfolio.py"),
    ("docs/evidence_health.json", "scripts/validate_evidence_health.py"),
    ("docs/release_checklist.json", "scripts/validate_release_checklist.py"),
]


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def _line(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{context} must be one non-empty line")
    return value


def _inside_root(relative: str, context: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context} must be a repository-relative path")
    resolved = (ROOT / candidate).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"{context} escapes the repository")
    return resolved


def validate_document(summary: Any, *, check_files: bool) -> None:
    if not isinstance(summary, dict):
        raise ValueError("release summary must be an object")
    _keys(summary, TOP_KEYS, "release summary")
    if summary["schema_version"] != 1 or isinstance(summary["schema_version"], bool):
        raise ValueError("unsupported schema_version")
    if summary["verification_command"] != "python3 scripts/verify_portfolio_release.py":
        raise ValueError("verification command differs")
    generated_on = _line(summary["generated_on"], "generated_on")
    try:
        date.fromisoformat(generated_on)
    except ValueError as error:
        raise ValueError("generated_on must be an ISO calendar date") from error

    artifacts = summary["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED):
        raise ValueError("artifacts must contain the three prescribed entries")
    identities: list[tuple[str, str]] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"artifacts[{index}] must be an object")
        _keys(artifact, ARTIFACT_KEYS, f"artifacts[{index}]")
        path = _line(artifact["path"], f"artifacts[{index}].path")
        validator = _line(artifact["validator"], f"artifacts[{index}].validator")
        digest = _line(artifact["sha256"], f"artifacts[{index}].sha256")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"artifacts[{index}].sha256 is invalid")
        identities.append((path, validator))
        artifact_path = _inside_root(path, f"artifacts[{index}].path")
        validator_path = _inside_root(validator, f"artifacts[{index}].validator")
        if check_files:
            actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"hash differs for {path}")
            if not validator_path.is_file():
                raise ValueError(f"validator is absent: {validator}")
    if identities != EXPECTED:
        raise ValueError("artifact identities or order differ")


def load_summary(path: Path = DEFAULT_SUMMARY) -> tuple[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid release summary JSON: {error}") from error
    return raw, document


def validate_summary(path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    raw, document = load_summary(path)
    validate_document(document, check_files=True)
    canonical = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if raw != canonical:
        raise ValueError("release summary is not canonical JSON")
    return document


def run_validators(summary: dict[str, Any]) -> None:
    for artifact in summary["artifacts"]:
        validator = ROOT / artifact["validator"]
        for arguments in ([], ["--self-test"]):
            command = [sys.executable, str(validator), *arguments]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                suffix = " self-test" if arguments else ""
                raise RuntimeError(f"{artifact['validator']}{suffix} failed")


def run_self_test() -> None:
    _, document = load_summary()
    mutations = []
    extra = copy.deepcopy(document); extra["unexpected"] = True; mutations.append(extra)
    bad_hash = copy.deepcopy(document); bad_hash["artifacts"][0]["sha256"] = "0" * 64; mutations.append(bad_hash)
    bad_path = copy.deepcopy(document); bad_path["artifacts"][1]["path"] = "../private.json"; mutations.append(bad_path)
    duplicate = copy.deepcopy(document); duplicate["artifacts"][2] = copy.deepcopy(duplicate["artifacts"][0]); mutations.append(duplicate)
    bad_validator = copy.deepcopy(document); bad_validator["artifacts"][2]["validator"] = "scripts/check.sh"; mutations.append(bad_validator)
    for mutation in mutations:
        try:
            validate_document(mutation, check_files=True)
        except ValueError:
            continue
        raise AssertionError("corrupted release summary was accepted")
    print(f"PASS: rejected {len(mutations)} release-summary corruptions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
        else:
            summary = validate_summary()
            run_validators(summary)
            print(f"PASS: verified {len(summary['artifacts'])} release artifacts and validators")
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
