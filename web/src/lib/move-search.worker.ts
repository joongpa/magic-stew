import { loadCpuFitter, type OpenCvModule } from "../../../pose-fitter/runtime/wasm-host.mjs";
import type { MoveSearchResult } from "./move-search-types";
let engine: Awaited<ReturnType<typeof loadCpuFitter>>;
let initialized: Promise<void>;
let timer:ReturnType<typeof setTimeout>|undefined;
let prepareMs=0,searchMs=0,lastReport=0,fingerprint="";
function call(request:Record<string,unknown>):MoveSearchResult {
  const result=engine.call("move_search_json",request);
  if (result.Err) throw new Error(result.Err);
  return result.Ok;
}
function report(result:MoveSearchResult,force=false) {
  const now=performance.now();
  if (force||now-lastReport>=100||result.status!=="running") {
    lastReport=now;self.postMessage({type:"result",result:{...result,timings:{prepareMs,searchMs},pipelineVersion:fingerprint}});
  }
}
function schedule(){if(timer===undefined)timer=setTimeout(tick,0);}
function tick(){
  timer=undefined;
  try {
    const start=performance.now(),result=call({type:"advance",budget:128});searchMs+=performance.now()-start;
    report(result);if(result.status==="running")schedule();
  }catch(cause){fail(cause);}
}
function fail(cause:unknown){if(timer!==undefined)clearTimeout(timer);timer=undefined;
  self.postMessage({type:"error",message:cause instanceof Error?cause.message:String(cause)});}
self.onmessage=({data}:MessageEvent)=>{
  if(data.type==="init"){
    initialized=(async()=>{
      const base=new URL("pose/",data.baseUrl),response=await fetch(new URL("pipeline.json",base),{cache:"no-store"});
      if(!response.ok)throw new Error("Complete Rust/WASM runtime is missing. Run npm run sync:runtime.");
      const manifest=await response.json();
      const [binary,geometry]=await Promise.all([fetch(new URL(manifest.rust,base)),fetch(new URL(manifest.geometry,base))]);
      if(!binary.ok||!geometry.ok)throw new Error("Move search runtime files are missing");
      // Move search imports no camera kernels and downloads no OpenCV module.
      const noCamera={_pose_cv(){throw new Error("Move search attempted camera inference");}} as unknown as OpenCvModule;
      engine=await loadCpuFitter(await binary.arrayBuffer(),new TextEncoder().encode(JSON.stringify({geometry:await geometry.json(),workloads:[],probes:[]})),noCamera);
      fingerprint=manifest.rust;report(call({type:"init",config:data.config}),true);
    })();void initialized.catch(fail);return;
  }
  void initialized?.then(()=>{
    if(data.type==="frame"){
      const start=performance.now(),result=call({type:"frame",frame:data.frame});prepareMs+=performance.now()-start;
      report(result);schedule();
    }else if(data.type==="end"){report(call({type:"end"}),true);schedule();}
  }).catch(fail);
};
