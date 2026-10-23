// Dump pi's model-facing tool declarations and system prompt for the parity tests.
// Run from a pi checkout: node --experimental-strip-types dump_declarations.ts <pi-root> <cwd>
import { pathToFileURL } from "node:url";
import { join, resolve } from "node:path";

const [piRoot, cwd] = process.argv.slice(2);
const src = (rel: string) => pathToFileURL(join(piRoot, "packages/coding-agent/src", rel)).href;

const tools = await import(src("core/tools/index.ts"));
const { buildSystemPromptSections } = await import(src("core/system-prompt.ts"));

const definitions = [
	tools.createReadToolDefinition(cwd),
	tools.createBashToolDefinition(cwd),
	tools.createEditToolDefinition(cwd),
	tools.createWriteToolDefinition(cwd),
	tools.createGrepToolDefinition(cwd),
	tools.createFindToolDefinition(cwd),
	tools.createLsToolDefinition(cwd),
];
const declarations = definitions.map((d: any) => ({
	name: d.name,
	description: d.description,
	parameters: JSON.parse(JSON.stringify(d.parameters)),
	promptSnippet: d.promptSnippet ?? null,
	promptGuidelines: d.promptGuidelines ?? [],
}));

const snippets = Object.fromEntries(declarations.filter((d) => d.promptSnippet).map((d) => [d.name, d.promptSnippet]));
const guidelines = Object.fromEntries(declarations.map((d) => [d.name, d.promptGuidelines]));
const prompt = (selectedTools: string[], extra: Record<string, unknown> = {}) =>
	buildSystemPromptSections({ cwd, selectedTools, toolSnippets: snippets, toolGuidelines: guidelines, ...extra });
const skill = (name: string, description: string, disableModelInvocation = false) => ({
	name,
	description,
	filePath: `/skills/${name}/SKILL.md`,
	baseDir: `/skills/${name}`,
	disableModelInvocation,
});
const resources = {
	appendSystemPrompt: "Extra instructions.",
	contextFiles: [
		{ path: "/home/u/.pi/agent/AGENTS.md", content: "Global rules." },
		{ path: "/work/AGENTS.md", content: "Project rules." },
	],
	skills: [skill("demo", "Use <tags> & 'quotes' \"here\""), skill("hidden", "Never listed", true)],
};

console.log(
	JSON.stringify(
		{
			declarations,
			prompts: {
				default: prompt(["read", "bash", "edit", "write"]),
				all: prompt(["read", "bash", "edit", "write", "grep", "find", "ls"]),
				readonly: prompt(["read", "grep", "find", "ls"]),
				resources: prompt(["read", "bash", "edit", "write"], resources),
				bashSkills: prompt(["bash"], { skills: resources.skills }),
			},
		},
		null,
		2,
	)
		// The <docs> section names the local pi checkout; keep machine paths out of fixtures.
		.replaceAll(resolve(piRoot), "<pi-root>"),
);
