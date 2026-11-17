import { readFileSync,writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createInterface } from 'node:readline';
import { loadFitter } from './wasm-host.mjs';
const [fixturePath,outputPath,repeatsText='5']=process.argv.slice(2);
const bytes=readFileSync(new URL('../target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm',import.meta.url));
const engine=await loadFitter(bytes,readFileSync(fixturePath));
const metadata={runtime:process.version,engine:process.versions.v8,wasmBytes:bytes.length,probes:engine.probes()};
if(outputPath==='--serve'){
 console.log(JSON.stringify({ready:true,...metadata}));
 for await(const line of createInterface({input:process.stdin})){
  const {index}=JSON.parse(line);const start=performance.now();engine.run(index);const ms=performance.now()-start;
  console.log(JSON.stringify({index,ms,...engine.result()}));
 }
}else{
 const cases=[];
 for(let index=0;index<engine.count;index++){
  engine.run(index);const times=[];
  for(let pass=0;pass<Number(repeatsText);pass++){const start=performance.now();engine.run(index);times.push(performance.now()-start);}
  cases.push({index,times_ms:times,...engine.result()});console.log(index,times.map(x=>x.toFixed(2)).join(','));
 }
 writeFileSync(outputPath,JSON.stringify({...metadata,cases}));
}
