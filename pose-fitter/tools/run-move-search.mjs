import{readFileSync,writeFileSync}from'node:fs';
import{loadCpuFitter}from'../runtime/wasm-host.mjs';
const[inputPath,outputPath,mode='batch']=process.argv.slice(2),input=JSON.parse(readFileSync(inputPath));
const bytes=readFileSync(new URL('../target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm',import.meta.url));
const geometry=JSON.parse(readFileSync(new URL('../tests/fixtures/cpu-golden.json',import.meta.url))).geometry;
const engine=await loadCpuFitter(bytes,Buffer.from(JSON.stringify({geometry,workloads:[],probes:[]})),{
 _pose_cv(){throw new Error('Move search must not call a camera kernel');}
});
const call=request=>{const result=engine.call('move_search_json',request);if(result.Err)throw new Error(result.Err);return result.Ok;};
call({type:'init',config:input.config});let prepareMs=0,searchMs=0,maxFrontier=0;
const tick=budget=>{const start=performance.now(),result=call({type:'advance',budget});searchMs+=performance.now()-start;return result;};
for(let i=0;i<input.frames.length;i++){
 const start=performance.now();call({type:'frame',frame:input.frames[i]});prepareMs+=performance.now()-start;
 if(mode==='stream'){
  const result=tick(128);maxFrontier=Math.max(maxFrontier,result.stats.furthest_frames);
  if(result.stats.furthest_frames>i+1)throw new Error('Search crossed its available-frame frontier');
 }
}
call({type:'end'});
const probes=(mode==='batch'?input.probes:[]).map(p=>call({type:'probe',sequence:p.sequence,state:p.state,cue:p.cue}));
let result;do{result=tick(10000);}while(result.status==='running');
const output={mode,result,probes,prepare_seconds:prepareMs/1000,search_seconds:searchMs/1000,max_stream_frontier:maxFrontier};
writeFileSync(outputPath,JSON.stringify(output));console.log(input.name,mode,result.status,result.stats.evaluated_states,prepareMs.toFixed(1),'ms prep',searchMs.toFixed(1),'ms search');
