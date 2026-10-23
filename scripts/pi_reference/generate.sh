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
node "$here/run_streaming_json_cases.ts" "$pi_root" "$repo/tests/parity/streaming_json_cases.json" > "$fixtures/pi_streaming_json_results.json"
node "$here/run_provider_cases.ts" "$pi_root" "$repo/tests/parity/provider_cases.json" > "$fixtures/pi_provider_results.json"
node "$here/run_compaction_cases.ts" "$pi_root" "$repo/tests/parity/compaction_cases.json" > "$fixtures/pi_compaction_results.json"
node "$here/run_collation_cases.ts" "$pi_root" "$repo/tests/parity/collation_cases.json" > "$fixtures/pi_collation_results.json"
node "$here/run_image_cases.ts" "$pi_root" "$repo/tests/parity/image_cases.json" > "$fixtures/pi_image_results.json"
printf '{\n  "version": "%s",\n  "commit": "%s"\n}\n' "$version" "$commit" > "$fixtures/pi_version.json"
echo "Fixtures regenerated from pi $version ($commit)"

# The model catalog the port ships (pi's generated provider data).
for provider in anthropic deepseek; do
	cp "$pi_root/packages/ai/src/providers/data/$provider.json" "$repo/src/pi_agent/data/models/$provider.json"
done
echo "Model catalog copied for: anthropic deepseek"

# Byte-identical image processing: ship Pi's exact Photon binary and its license.
photon="$pi_root/node_modules/@silvia-odwyer/photon-node"
photon_version="$(node -p "require('$photon/package.json').version")"
if [[ "$photon_version" != "0.3.4" ]]; then
    echo "Photon changed to $photon_version: update the Python bindings and image fixtures first" >&2
    exit 1
fi
cp "$photon/photon_rs_bg.wasm" "$repo/src/pi_agent/data/wasm/photon-0.3.4.wasm"
cp "$photon/LICENSE.md" "$repo/src/pi_agent/data/licenses/photon.txt"
