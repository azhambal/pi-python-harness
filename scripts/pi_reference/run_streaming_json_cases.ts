// Pi's partial argument values at every prefix, not just completed tool calls.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [root, casesPath] = process.argv.slice(2);
const { parseStreamingJson } = await import(pathToFileURL(join(root, "packages/ai/src/utils/json-parse.ts")).href);
const cases = JSON.parse(readFileSync(casesPath, "utf-8"));
console.log(JSON.stringify(cases.map((text: string) => parseStreamingJson(text)), null, 1));
