// Run pi's prepareCompaction() and compact() (packages/coding-agent/src/core/compaction)
// over the cases in tests/parity/compaction_cases.json and print the results as JSON.
// Run: node run_compaction_cases.ts <pi-root> <cases.json>
//
// Each case lists session entries in order: {"message": AgentMessage},
// {"compaction": {summary, firstKept (entry index), tokensBefore, details?}} or
// {"omit": <entry index>} (a context edit that omits the target).
// Entry ids are "e<index>". A recording streamFn answers every summarization
// request with "summary <n>" and captures what pi would send.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [piRoot, casesPath] = process.argv.slice(2);
const { prepareCompaction, compact } = await import(
	pathToFileURL(join(piRoot, "packages/coding-agent/src/core/compaction/index.ts")).href
);

const zeroUsage = () => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

function buildEntries(testCase: any): any[] {
	return testCase.entries.map((item: any, index: number) => {
		const base = {
			id: `e${index}`,
			parentId: index === 0 ? null : `e${index - 1}`,
			timestamp: new Date(1_700_000_000_000 + index * 1000).toISOString(),
		};
		if (item.message) return { ...base, type: "message", message: item.message };
		if (item.omit !== undefined) return { ...base, type: "context_edit", targetId: `e${item.omit}`, replacement: null };
		const { summary, firstKept, tokensBefore, details } = item.compaction;
		return { ...base, type: "compaction", summary, firstKeptEntryId: `e${firstKept}`, tokensBefore, details };
	});
}

const results = [];
for (const testCase of JSON.parse(readFileSync(casesPath, "utf-8"))) {
	const entries = buildEntries(testCase);
	const settings = { enabled: true, reserveTokens: 16384, keepRecentTokens: testCase.keepRecentTokens };
	const preparation = prepareCompaction(entries, settings);
	if (!preparation) {
		results.push({ name: testCase.name, preparation: null });
		continue;
	}
	const requests: any[] = [];
	const streamFn = async (_model: any, context: any, options: any) => {
		const [system, user] = context.messages;
		requests.push({
			system: system.content,
			roles: context.messages.map((message: any) => message.role),
			prompt: user.content.map((block: any) => block.text).join(""),
			reasoning: options.reasoning ?? null,
			cacheRetention: options.cacheRetention ?? null,
			maxTokens: options.maxTokens ?? null,
			hasSessionId: typeof options.sessionId === "string" && options.sessionId.length > 0,
		});
		const message = {
			role: "assistant",
			content: [{ type: "text", text: `summary ${requests.length}` }],
			api: "anthropic-messages",
			provider: "anthropic",
			model: "test",
			usage: { ...zeroUsage(), input: requests.length, totalTokens: requests.length },
			stopReason: "stop",
			timestamp: 0,
		};
		return { result: async () => message };
	};
	const model = {
		id: "test",
		name: "test",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://example.invalid",
		reasoning: testCase.reasoning ?? false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 8192,
	};
	const result = await compact(
		preparation,
		model,
		"synthetic",
		undefined,
		testCase.customInstructions,
		undefined,
		testCase.thinkingLevel ?? "off",
		streamFn,
	);
	results.push({
		name: testCase.name,
		preparation: {
			firstKeptEntryId: preparation.firstKeptEntryId,
			isSplitTurn: preparation.isSplitTurn,
			messagesToSummarize: preparation.messagesToSummarize.length,
			turnPrefixMessages: preparation.turnPrefixMessages.length,
			tokensBefore: preparation.tokensBefore,
			previousSummary: preparation.previousSummary ?? null,
		},
		requests,
		summary: result.summary,
		details: result.details,
		usageTotal: result.usage.totalTokens,
	});
}
console.log(JSON.stringify(results, null, 2));
