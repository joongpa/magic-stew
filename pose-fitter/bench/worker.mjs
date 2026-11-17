import { loadFitter } from '../tools/wasm-host.mjs';

// A single persistent instance drains the complete buffer; no deadline skips jobs.
self.onmessage=async({data:{repeats}})=>{
  const [wasmResponse,fixtureResponse]=await Promise.all([
    fetch('../target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm'),
    fetch('../fixtures/cpu-reference.json'),
  ]);
  const bytes=await wasmResponse.arrayBuffer(), fixtureBytes=new Uint8Array(await fixtureResponse.arrayBuffer());
  const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',fixtureBytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  const loaded=performance.now(),engine=await loadFitter(bytes,fixtureBytes);
  const initMs=performance.now()-loaded;
  self.postMessage({type:'progress',message:'Checking projection and identical-seed solver parity…'});
  const probes=engine.probes(),cases=[];
  for(let index=0;index<engine.count;index++){
    self.postMessage({type:'progress',message:`Fitting buffered workload ${index+1}/${engine.count}…`});
    engine.run(index); // full untimed warm-up
    const times=[];let outputMs=0,result;
    for(let pass=0;pass<repeats;pass++){
      const start=performance.now();engine.run(index);times.push(performance.now()-start);
      const outputStart=performance.now();result=engine.result();outputMs+=performance.now()-outputStart;
    }
    const median=[...times].sort((a,b)=>a-b)[Math.floor(times.length/2)];
    cases.push({index,times_ms:times,median_ms:median,mean_output_ms:outputMs/repeats,...result});
    self.postMessage({type:'case',index,median,count:result.bank.length});
  }
  self.postMessage({type:'done',report:{fixture_sha256:digest,wasmBytes:bytes.byteLength,init_ms:initMs,repeats,probes,cases}});
};
