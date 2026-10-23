#!/usr/bin/env bash
# Regenerate tests/parity/fixtures from a pi checkout (default: ../pi).
#
# Requirements: Node.js, `npm ci && npm run build` in the pi checkout, and
# rg/fd on PATH (pi would otherwise download them into ~/.pi/agent/bin).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
pi_root="$(cd "${1:-$repo/../pi}" && pwd)"
fixtures="$repo/tests/parity/fixtures"
mkdir -p "$fixtures"

version="$(node -p "require('$pi_root/packages/coding-agent/package.json').version")"
commit="$(git -C "$pi_root" rev-parse HEAD)"

node "$here/dump_declarations.ts" "$pi_root" /work > "$fixtures/pi_declarations.json"
node "$here/run_tool_cases.ts" "$pi_root" "$repo/tests/parity/tool_cases.json" > "$fixtures/pi_tool_results.json"
node "$here/run_transform_cases.ts" "$pi_root" "$repo/tests/parity/transform_cases.json" > "$fixtures/pi_transform_results.json"
node "$here/run_json_cases.ts" "$pi_root" "$repo/tests/parity/json_cases.json" > "$fixtures/pi_json_results.json"
node "$here/run_provider_cases.ts" "$pi_root" "$repo/tests/parity/provider_cases.json" > "$fixtures/pi_provider_results.json"
node "$here/run_compaction_cases.ts" "$pi_root" "$repo/tests/parity/compaction_cases.json" > "$fixtures/pi_compaction_results.json"
printf '{\n  "version": "%s",\n  "commit": "%s"\n}\n' "$version" "$commit" > "$fixtures/pi_version.json"
echo "Fixtures regenerated from pi $version ($commit)"

# The model catalog the port ships (pi's generated provider data).
for provider in anthropic deepseek; do
	cp "$pi_root/packages/ai/src/providers/data/$provider.json" "$repo/src/pi_agent/data/models/$provider.json"
done
echo "Model catalog copied for: anthropic deepseek"
