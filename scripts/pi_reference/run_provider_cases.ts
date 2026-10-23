// Run provider cases through pi's own streamSimple with an injected fetch, and print
// what pi sent (URL, selected headers, JSON body) and what it made of the response.
// Run: node run_provider_cases.ts <pi-root> <cases.json>
//
// A case without "sse" stops after capturing the request; with "sse", the fetch answers
// with that text/event-stream body and the final assistant message is recorded.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [piRoot, casesPath] = process.argv.slice(2);
const src = (rel: string) => pathToFileURL(join(piRoot, "packages/ai/src", rel)).href;
const apis: Record<string, any> = {
	"anthropic-messages": await import(src("api/anthropic-messages.ts")),
	"openai-completions": await import(src("api/openai-completions.ts")),
};

const HEADERS = [
	"accept",
	"anthropic-beta",
	"anthropic-dangerous-direct-browser-access",
	"anthropic-version",
	"authorization",
	"content-type",
	"x-api-key",
	"x-session-affinity",
	"x-session-id",
];

function catalogModel(provider: string, id: string): any {
	const groups = JSON.parse(readFileSync(join(piRoot, `packages/ai/src/providers/data/${provider}.json`), "utf-8"));
	for (const entries of Object.values(groups) as any[]) {
		const entry = entries[`chat:${id}`];
		if (entry) return entry;
	}
	throw new Error(`No catalog model ${provider}/${id}`);
}

class Captured extends Error {}

const cases = JSON.parse(readFileSync(casesPath, "utf-8"));
const results = [];
for (const testCase of cases) {
	const model = testCase.model.catalog
		? { ...catalogModel(testCase.model.provider, testCase.model.catalog), ...(testCase.model.overrides ?? {}) }
		: testCase.model;
	let request: any;
	const fetch = async (url: any, init: any) => {
		const headers = new Headers(init?.headers);
		request = {
			url: String(url),
			headers: Object.fromEntries(HEADERS.filter((name) => headers.has(name)).map((name) => [name, headers.get(name)])),
			body: JSON.parse(init.body),
		};
		if (testCase.sse === undefined) throw new Captured("captured");
		return new Response(testCase.sse, { status: 200, headers: { "content-type": "text/event-stream" } });
	};
	const stream = apis[model.api].streamSimple(model, { messages: testCase.messages }, {
		apiKey: testCase.apiKey ?? "sk-ant-api03-test",
		fetch,
		maxRetries: 0,
		...(testCase.options ?? {}),
	});
	const toolDeltas = [];
	if (testCase.sse !== undefined) {
		for await (const event of stream) {
			if (event.type === "toolcall_delta") {
				toolDeltas.push(JSON.parse(JSON.stringify({contentIndex:event.contentIndex,delta:event.delta,
					arguments:event.partial.content[event.contentIndex].arguments})));
			}
		}
	}
	const message = await stream.result();
	const { timestamp: _timestamp, durationMs: _duration, ...result } = message;
	results.push({ name: testCase.name, request, ...(testCase.sse === undefined ? {} : { result, toolDeltas }) });
}
console.log(JSON.stringify(results, null, 1));
