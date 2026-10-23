// Full processImage outputs, including base64 payloads, from Pi's Photon codecs.
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const [root, casesPath] = process.argv.slice(2);
// Pi may choose a worker or its in-process fallback; compare the resulting bytes.
const { processImage } = await import(pathToFileURL(join(root,"packages/coding-agent/src/utils/image-process.ts")).href);
const {createReadTool}=await import(pathToFileURL(join(root,"packages/coding-agent/src/core/tools/read.ts")).href);
const cases = JSON.parse(readFileSync(casesPath,"utf-8"));
const results = [];
for(const item of cases) {
 const result=await processImage(Buffer.from(item.data,"base64"),item.mimeType,item.options);
 const dir=mkdtempSync(join(tmpdir(),'pi-image-'));
 try {
  writeFileSync(join(dir,'image'),Buffer.from(item.data,'base64'));
  const tool=createReadTool(dir,{autoResizeImages:item.options?.autoResizeImages});
  const read=await tool.execute('c',{path:'image'},undefined,undefined,{cwd:dir,model:{input:['text','image'],inputLimits:{images:{resize:item.options?.resizeOptions}}}});
  results.push({name:item.name,result,readContent:read.content});
 } finally {rmSync(dir,{recursive:true,force:true});}
}
console.log(JSON.stringify(results,null,1));
