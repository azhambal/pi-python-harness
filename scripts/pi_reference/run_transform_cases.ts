// Run pi's transformMessages (packages/ai/src/api/transform-messages.ts) over the
// cases in tests/parity/transform_cases.json and print the results as JSON.
// Run: node run_transform_cases.ts <pi-root> <cases.json>
//
// The tool call ID normalizers are copies of the provider-private ones in
// anthropic-messages.ts and openai-completions.ts.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [piRoot, casesPath] = process.argv.slice(2);
const { transformMessages } = await import(
	pathToFileURL(join(piRoot, "packages/ai/src/api/transform-messages.ts")).href
);

const { shortHash } = await import(pathToFileURL(join(piRoot, "packages/ai/src/utils/hash.ts")).href);

const normalizers: Record<string, ((id: string, model: any) => string) | undefined> = {
	anthropic: (id) => id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64),
	openai: (id, model) => {
		if (id.includes("|")) {
			const separatorIndex = id.indexOf("|");
			const callId = id.slice(0, separatorIndex).replace(/[^a-zA-Z0-9_-]/g, "_");
			const itemId = id.slice(separatorIndex + 1).replace(/[^a-zA-Z0-9_-]/g, "_");
			const combinedId = itemId.length > 0 ? `${callId}_${itemId}` : callId;
			if (combinedId.length <= 40) return combinedId;
			const hash = shortHash(id).slice(0, 8);
			const prefix = callId.slice(0, Math.max(1, 40 - hash.length - 1));
			return `${prefix}_${hash}`;
		}
		if (model.provider === "openai") return id.length > 40 ? id.slice(0, 40) : id;
		return id;
	},
	none: undefined,
};

const cases = JSON.parse(readFileSync(casesPath, "utf-8"));
const results = cases.map((testCase: any) => ({
	name: testCase.name,
	messages: transformMessages(testCase.messages, testCase.model, normalizers[testCase.normalizer ?? "none"]).map(
		(message: any) => {
			const { timestamp: _timestamp, ...rest } = message;
			return rest;
		},
	),
}));
console.log(JSON.stringify(results, null, 2));
