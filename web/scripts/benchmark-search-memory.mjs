/** Compare identical forward work in fresh workers. Cached fitted inputs isolate
 * search; WASM memory reports allocated pages, not live-object heap usage. */
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {loadCpuFitter} from '../../pose-fitter/runtime/wasm-host.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
if(isMainThread){
  const [input,baseline,current,output,statesText='4096',passesText='2']=process.argv.slice(2);
  const states=Number(statesText),passes=Number(passesText);
  if(!output||!Number.isSafeInteger(states)||states<=0||!Number.isSafeInteger(passes)||passes<=0)
    throw Error('Usage: node benchmark-search-memory.mjs INPUT BASELINE.wasm CURRENT.wasm REPORT [STATES=4096] [PASSES=2]');
  const runs={baseline:[],current:[]};
  for(let pass=0;pass<passes;pass++){
    // Alternate order; no old search heap remains while its comparison runs.
    for(const kind of pass%2?['current','baseline']:['baseline','current']){
      const result=await new Promise((resolveResult,reject)=>{
        const worker=new Worker(new URL(import.meta.url),{workerData:{input:resolve(input),wasm:resolve(kind==='baseline'?baseline:current),states}});
        worker.once('message',async value=>{await worker.terminate();resolveResult(value);});
        worker.once('error',reject);
      });
      runs[kind].push(result);console.log(JSON.stringify({pass,kind,...result}));
    }
  }
  const reference=runs.baseline[0];
  for(const run of [...runs.baseline,...runs.current]){
    assert.equal(run.traceSha256,reference.traceSha256,'A search progress snapshot differs');
    assert.equal(run.resultSha256,reference.resultSha256,'Final search output differs');
  }
  const median=values=>{const a=[...values].sort((a,b)=>a-b);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
  const summary=Object.fromEntries(Object.entries(runs).map(([kind,values])=>[kind,{
    searchMs:median(values.map(run=>run.searchMs)),statesPerSecond:median(values.map(run=>run.statesPerSecond)),
    wasmMemoryBytes:median(values.map(run=>run.memoryEndBytes)),
  }]));
  const report={input:resolve(input),scope:'Forward search only, fresh worker per run, same fitted banks and full snapshot equality. WASM allocated-memory high-water mark; not browser live FPS.',
    runtime:process.version,states,passes,exactOutputMatch:true,summary,runs};
  await writeFile(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({exactOutputMatch:true,summary}));
}else{
  const bytes=await readFile(workerData.wasm),inputBytes=await readFile(workerData.input),input=JSON.parse(inputBytes);
  const geometry=JSON.parse(await readFile(new URL('../../pose-fitter/tests/fixtures/cpu-golden.json',import.meta.url),'utf8')).geometry;
  const engine=await loadCpuFitter(bytes,new TextEncoder().encode(JSON.stringify({geometry,workloads:[],probes:[]})),
    {_pose_cv(){throw Error('Search cannot run pose fitting');}});
  const trace=createHash('sha256');let searchMs=0;
  const call=request=>{
    const started=performance.now(),value=engine.call('move_search_json',request);
    if(request.type==='advance')searchMs+=performance.now()-started;
    if(value.Err)throw Error(value.Err);
    trace.update(JSON.stringify(value.Ok));return value.Ok;
  };
  call({type:'init',config:{...input.config,bidirectional:false,state_limit:workerData.states}});
  for(const frame of input.frames)call({type:'frame',frame});
  let result=call({type:'end'});const memoryStartBytes=engine.memoryBytes;
  while(['running','waiting_for_frames'].includes(result.status))result=call({type:'advance',budget:128});
  parentPort.postMessage({wasmSha256:hash(bytes),inputSha256:hash(inputBytes),frames:input.frames.length,
    status:result.status,states:result.stats.evaluated_states,searchMs,statesPerSecond:result.stats.evaluated_states*1000/searchMs,
    memoryStartBytes,memoryEndBytes:engine.memoryBytes,traceSha256:trace.digest('hex'),resultSha256:hash(JSON.stringify(result))});
}
