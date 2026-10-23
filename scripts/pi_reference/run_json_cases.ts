// Run pi's parseToolCallArguments (packages/ai/src/utils/json-parse.ts) over the
// argument texts in tests/parity/json_cases.json and print the results as JSON.
// Run: node run_json_cases.ts <pi-root> <cases.json>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [piRoot, casesPath] = process.argv.slice(2);
const { parseToolCallArguments } = await import(pathToFileURL(join(piRoot, "packages/ai/src/utils/json-parse.ts")).href);
const cases: string[] = JSON.parse(readFileSync(casesPath, "utf-8"));
console.log(JSON.stringify(cases.map((text) => parseToolCallArguments(text)), null, 1));
