// Record both Intl ordering and actual ls output (including readdir tie ordering).
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
const [root,casesPath]=process.argv.slice(2);
const {createLsTool}=await import(pathToFileURL(join(root,'packages/coding-agent/src/core/tools/ls.ts')).href);
const cases=JSON.parse(readFileSync(casesPath,'utf-8'));
const results=[];
for(const item of cases){
 const dir=mkdtempSync(join(tmpdir(),'pi-collation-'));
 try{
  for(const name of item.names) writeFileSync(join(dir,name),'');
  const tool=createLsTool(dir);
  const result=await tool.execute('c',item.limit===undefined?{}:{limit:item.limit});
  results.push({name:item.name,ordered:[...item.names].sort((a,b)=>a.toLowerCase().localeCompare(b.toLowerCase())),content:result.content});
 }finally{rmSync(dir,{recursive:true,force:true});}
}
console.log(JSON.stringify({icu:process.versions.icu,unicode:process.versions.unicode,locale:Intl.Collator().resolvedOptions().locale,results},null,1));
