#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
revision=${1:-HEAD}
commit=$(git -C "$repository_root" rev-parse --verify "${revision}^{commit}")
clean_root=$(mktemp -d "${TMPDIR:-/tmp}/autograd-clean.XXXXXX")
trap 'rm -rf "$clean_root"' EXIT

git -C "$repository_root" archive --format=tar "$commit" | tar -xf - -C "$clean_root"
printf 'Verifying clean archive at %s\n' "$commit"
(cd "$clean_root" && bash scripts/check.sh)
printf 'PASS: clean archive contains everything required by the direct check suite\n'
