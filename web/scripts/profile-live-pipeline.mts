// Refit saved raw detections, isolating CPU work from the camera/GPU/search.
// Optional baseline comparison requires every returned fitted value to match.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {pipelineFrame} from '../src/lib/pose-fitter.ts';
import {BufferedPosePipeline} from '../../pose-fitter/runtime/buffered-pipeline.mjs';
import {loadCpuFitter} from '../../pose-fitter/runtime/wasm-host.mjs';

const [input,output,...args]=process.argv.slice(2);
if(!input||!output||args.length&&!(args.length===2&&args[0]==='--baseline-wasm'))
  throw Error('Usage: tsx scripts/profile-live-pipeline.mts CAPTURE.json REPORT.json [--baseline-wasm OLD.wasm]');
const captureBytes=await readFile(input),recording=JSON.parse(captureBytes.toString());
const run=recording.runs?.[0];
const rows=run?.frameSequences?run.frameSequences.map((i:number)=>recording.frames[i]):recording.frames;
assert.ok(rows.length,'Recording contains no frames');
const config=rows[0].result.fittedPose?.fittingConfig;
assert.ok(config,'Recording must include the fitting clock/configuration');
assert.ok(rows.every((row:any)=>row.result.fittedPose?.fittingConfig?.segment===config.segment),
  'Profile one fitting segment at a time');
const base=new URL('../public/pose/',import.meta.url);
const manifest=JSON.parse(await readFile(new URL('pipeline.json',base),'utf8'));
const geometry=JSON.parse(await readFile(new URL(manifest.geometry,base),'utf8'));
const fixture=new TextEncoder().encode(JSON.stringify({geometry,workloads:[],probes:[]}));
const cvModule=await import(new URL(manifest.opencvModule,base).href);
const cvBytes=await readFile(new URL(manifest.opencvWasm,base));
const hash=(bytes:Uint8Array|string)=>createHash('sha256').update(bytes).digest('hex');
async function profile(wasm:URL){
  const bytes=await readFile(wasm);
  const cv=await cvModule.default({wasmBinary:cvBytes});
  const engine=await loadCpuFitter(bytes,fixture,cv);
  const samples=new Map<string,number[]>();
  const pipeline=new BufferedPosePipeline(async job=>{
    const start=performance.now(),value=engine.call('pipeline_job_json',job);
    const stage=String(job.type),times=samples.get(stage)??[];
    times.push(performance.now()-start);samples.set(stage,times);return value;
  },{fps:config.fps,turningStartFrame:config.turningStartFrame,centers:config.centers});
  const start=performance.now();
  const pending=rows.map((row:any)=>pipeline.push(pipelineFrame(row.result,Math.round(row.timestampSeconds*config.fps))));
  await pipeline.flush();const fits=await Promise.all(pending);
  const report={wasm:wasm.pathname,wasmSha256:hash(bytes),wallMs:performance.now()-start,
    fitsSha256:hash(JSON.stringify(fits)),stages:Object.fromEntries([...samples].map(([stage,times])=>{
      const sorted=[...times].sort((a,b)=>a-b),totalMs=times.reduce((a,b)=>a+b,0);
      return [stage,{jobs:times.length,totalMs,meanMs:totalMs/times.length,p95Ms:sorted[Math.floor(times.length*.95)]}];
    }))};
  pipeline.cancel();return {fits,report};
}
const baseline=args[1]?await profile(pathToFileURL(resolve(args[1]))):undefined;
const current=await profile(new URL(manifest.rust,base));
if(baseline)assert.deepEqual(current.fits,baseline.fits,'Fitted output differs from baseline');
const report={input:resolve(input),inputSha256:hash(captureBytes),frames:rows.length,
  scope:'Fresh CPU fitting from raw detections; excludes camera capture, detector, search and browser worker scheduling. Saved camera prelude is unavailable.',
  exactOutputMatch:baseline?true:undefined,baseline:baseline?.report,current:current.report,
  cpuTimeReduction:baseline?1-current.report.wallMs/baseline.report.wallMs:undefined};
await writeFile(output,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
