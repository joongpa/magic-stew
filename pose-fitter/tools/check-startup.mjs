import{readFileSync,writeFileSync}from'node:fs';
import createOpenCv from '../target/opencv-pnp.mjs';
import{loadCpuFitter}from'./cpu-wasm-host.mjs';
const [fixturePath='pose-fitter/fixtures/startup-reference.json',outputPath='pose-fitter/results/startup-check.json']=process.argv.slice(2);
const fixture=readFileSync(fixturePath),data=JSON.parse(fixture);
const geometry=data.geometry??JSON.parse(readFileSync(new URL('../tests/fixtures/cpu-golden.json',import.meta.url))).geometry;
const cv=await createOpenCv();
const e=await loadCpuFitter(readFileSync(new URL('../target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm',import.meta.url)),Buffer.from(JSON.stringify({geometry,workloads:[],probes:[]})),cv);
const rows=[];
for(const item of data.cases){
 const actual=e.call('startup_json',item.input);
 const discrete=m=>({sticker_index:m.sticker_index,indices:m.indices,sticker_type:m.sticker_type,sequences:m.sequences,mean_presence:m.mean_presence});
 const matches=actual.matches.length===item.matches.length&&actual.matches.every((m,i)=>JSON.stringify(discrete(m))===JSON.stringify(discrete(item.matches[i]))&&Math.abs(m.center_error-item.matches[i].center_error)<1e-8);
 const first=actual.candidates[0],expected=item.candidates[0];
 const poseError=first&&expected?Math.max(...first.pose.rvec.map((v,i)=>Math.abs(v-expected.pose.rvec[i])),...first.pose.tvec.map((v,i)=>Math.abs(v-expected.pose.tvec[i]))):null;
 const distances=actual.candidates.map(c=>Math.min(...item.candidates.map(x=>Math.max(...c.pose.rvec.map((v,i)=>Math.abs(v-x.pose.rvec[i])),...c.pose.tvec.map((v,i)=>Math.abs(v-x.pose.tvec[i]))))));
 const cameras=actual.candidates.length===item.candidates.length&&distances.every(v=>v<1e-4);
 rows.push({name:item.name,frame:item.frame,groupsMatch:matches,cameraSetMatches:cameras,selectedCameraRawError:poseError,actual});
 if(!matches||!cameras)process.exitCode=1;
 console.log(item.name,item.frame,'groups',matches,'camera set',cameras,'selected camera raw error',poseError);
}
writeFileSync(outputPath,JSON.stringify({rows}));
