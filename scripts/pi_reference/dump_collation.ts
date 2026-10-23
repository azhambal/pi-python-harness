// Dump per-character ICU collation weights so pi_agent._collation can reproduce
// Node's String.prototype.localeCompare (used by pi's ls tool to sort entries).
// Run: node dump_collation.ts > src/pi_agent/data/collation.json
//
// Each character gets [primary, secondary, tertiary] ranks; primary 0 marks a
// character ICU ignores at the primary level (controls, combining marks).
// Characters ICU expands into several letters (æ = ae, ß = ss, ﬁ = fi) or into a
// letter plus a combining mark (ø = o + U+0338) are listed under "expansions"
// as [string they collate like, 1 if they still differ from it by accent (æ)].
// "atomic" lists precomposed letters ICU sorts as letters of their own (й), which
// must not be split into base letter and mark.

const RANGES: Array<[number, number]> = [
	[0x0000, 0x052f], // Latin, IPA, combining marks, Greek, Cyrillic
	[0x1e00, 0x1fff], // Latin Extended Additional, Greek Extended
	[0x2000, 0x206f], // General Punctuation
	[0x20a0, 0x20cf], // Currency Symbols
	[0x2100, 0x2bff], // Letterlike, arrows, math, technical, box drawing, dingbats
	[0xfb00, 0xfb06], // Latin ligatures
	[0x1f000, 0x1faff], // Emoji and pictographs
];

const variant = new Intl.Collator("en-US");
const accent = new Intl.Collator("en-US", { sensitivity: "accent" });
const base = new Intl.Collator("en-US", { sensitivity: "base" });

const chars: string[] = [];
for (const [start, end] of RANGES) {
	for (let cp = start; cp <= end; cp++) {
		if (cp >= 0xd800 && cp <= 0xdfff) continue;
		chars.push(String.fromCodePoint(cp));
	}
}
chars.sort((a, b) => variant.compare(a, b) || a.codePointAt(0)! - b.codePointAt(0)!);

const table: Record<string, [number, number, number]> = {};
let primary = 0;
let secondary = 0;
let tertiary = 0;
let previous: string | undefined;
const groupBase = new Map<string, string>();
let currentBase = "";
for (const char of chars) {
	const ignorable = base.compare(char, "") === 0;
	if (previous === undefined || base.compare(previous, char) !== 0) {
		primary += 1;
		secondary = 1;
		tertiary = 1;
		currentBase = char;
	} else if (accent.compare(previous, char) !== 0) {
		secondary += 1;
		tertiary = 1;
	} else if (variant.compare(previous, char) !== 0) {
		tertiary += 1;
	}
	table[char.codePointAt(0)!.toString(16)] = [ignorable ? 0 : primary, secondary, tertiary];
	groupBase.set(char, currentBase);
	previous = char;
}

// A character whose base-level weight equals a longer string is an expansion.
const LETTERS = "abcdefghijklmnopqrstuvwxyz";
const pairs: string[] = [];
for (const a of LETTERS) for (const b of LETTERS) pairs.push(a + b);
const MARKS: string[] = [];
for (let cp = 0x0300; cp <= 0x036f; cp++) MARKS.push(String.fromCodePoint(cp));
const expansions: Record<string, [string, number]> = {};
for (const char of chars) {
	if (base.compare(char, "") === 0) continue;
	const code = char.codePointAt(0)!.toString(16);
	const decomposed = char.normalize("NFKD").replace(/\p{M}/gu, "");
	const candidates = decomposed.length > 1 ? [decomposed] : /\p{Script=Latin}/u.test(char) ? pairs : [];
	const match = candidates.find((candidate) => base.compare(char, candidate) === 0);
	if (match !== undefined) {
		expansions[code] = [match, accent.compare(char, match) === 0 ? 0 : 1];
		continue;
	}
	const letter = groupBase.get(char)!;
	if (char.normalize("NFD") === char && letter !== char && accent.compare(char, letter) !== 0) {
		const marked = MARKS.map((mark) => letter + mark).find((candidate) => accent.compare(char, candidate) === 0);
		if (marked !== undefined) expansions[code] = [marked, 0];
	}
}

const atomic: string[] = [];
for (const char of chars) {
	const decomposed = char.normalize("NFD");
	if (decomposed !== char && base.compare(char, decomposed.replace(/\p{M}/gu, "")) !== 0) {
		atomic.push(char.codePointAt(0)!.toString(16));
	}
}

console.log(
	JSON.stringify({ icu: process.versions.icu, unicode: process.versions.unicode, weights: table, expansions, atomic }),
);
