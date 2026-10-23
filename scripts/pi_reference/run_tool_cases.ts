// Run tool-call cases through pi's own tools and agent-loop tool path (runToolCall),
// recording what the model would see. Used to (re)generate tests/parity/fixtures.
//
// Usage (from any directory, after `npm ci && npm run build` in the pi checkout):
//   node run_tool_cases.ts <pi-root> <cases.json> > tool_results.json
import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const [piRoot, casesPath] = process.argv.slice(2);
const src = (pkg: string, rel: string) => pathToFileURL(join(piRoot, "packages", pkg, "src", rel)).href;

const tools = await import(src("coding-agent", "core/tools/index.ts"));
const { runToolCall } = await import(src("agent", "agent-loop.ts"));

interface Case {
	name: string;
	tool: string;
	args: Record<string, unknown>;
	files?: Record<string, string>;
	binaryFiles?: Record<string, string>;
	dirs?: string[];
	symlinks?: Record<string, string>;
	gitInit?: boolean;
	readBack?: string[];
}

const cases: Case[] = JSON.parse(readFileSync(casesPath, "utf-8"));
const results: unknown[] = [];

for (const testCase of cases) {
	const root = realpathSync(mkdtempSync(join(tmpdir(), "pi-parity-")));
	try {
		for (const dir of testCase.dirs ?? []) mkdirSync(join(root, dir), { recursive: true });
		for (const [path, content] of Object.entries(testCase.files ?? {})) {
			mkdirSync(dirname(join(root, path)), { recursive: true });
			writeFileSync(join(root, path), content);
		}
		for (const [path, base64] of Object.entries(testCase.binaryFiles ?? {})) {
			mkdirSync(dirname(join(root, path)), { recursive: true });
			writeFileSync(join(root, path), Buffer.from(base64, "base64"));
		}
		for (const [path, target] of Object.entries(testCase.symlinks ?? {})) symlinkSync(target, join(root, path));
		if (testCase.gitInit) mkdirSync(join(root, ".git"), { recursive: true });

		const all = [
			tools.createReadTool(root),
			tools.createBashTool(root, { exposeSessionEnvironment: false }),
			tools.createEditTool(root),
			tools.createWriteTool(root),
			tools.createGrepTool(root),
			tools.createFindTool(root),
			tools.createLsTool(root),
		];
		const toolCall = { type: "toolCall", id: "call_1", name: testCase.tool, arguments: testCase.args };
		const outcome = await runToolCall(toolCall, {
			tools: all,
			assistantMessage: { role: "assistant", content: [toolCall] },
			context: { systemPrompt: "", messages: [], tools: all },
		});
		const content = outcome.result.content.map((block: any) =>
			block.type === "text"
				? { type: "text", text: block.text.split(root).join("<ROOT>") }
				: { type: block.type, mimeType: block.mimeType },
		);
		const files: Record<string, string | null> = {};
		for (const path of testCase.readBack ?? []) {
			try {
				files[path] = readFileSync(join(root, path), "utf-8");
			} catch {
				files[path] = null;
			}
		}
		results.push({ name: testCase.name, isError: outcome.isError, content, files });
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

console.log(JSON.stringify(results, null, 2));
