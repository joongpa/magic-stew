import{readFileSync,writeFileSync}from'node:fs';
import{isDeepStrictEqual}from'node:util';
import{loadCpuFitter}from'./cpu-wasm-host.mjs';
import createOpenCv from '../target/opencv-pnp.mjs';
import{BufferedPosePipeline}from'../runtime/buffered-pipeline.mjs';
const [fixturePath,outputPath]=process.argv.slice(2),data=JSON.parse(readFileSync(fixturePath));
const engine=await loadCpuFitter(readFileSync(new URL('../target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm',import.meta.url)),
  Buffer.from(JSON.stringify({geometry:data.geometry,workloads:[],probes:[]})),await createOpenCv());
const input={fps:data.fps,turningStartFrame:data.turning_start_frame,centers:data.expected_center_colors};
const stream=new BufferedPosePipeline(async job=>engine.call('pipeline_job_json',job),input);
const pending=data.frames.map(frame=>stream.push(frame));
await stream.flush();
const poses=await Promise.all(pending);
const batch=engine.call('pipeline_json',{fps:data.fps,turning_start_frame:data.turning_start_frame,expected_center_colors:data.expected_center_colors,frames:data.frames});
if(batch.Err)throw new Error(batch.Err);
const reference=batch.Ok.frames;
const issues=[];
for(let i=0;i<reference.length;i++){
 if(!isDeepStrictEqual(reference[i].bank,poses[i].bank)||reference[i].depth_recovery!==poses[i].depth_recovery)issues.push(reference[i].frame);
}
writeFileSync(outputPath,JSON.stringify({name:data.name,frames:reference.length,issues,poses}));
console.log(data.name,reference.length,'frames; streaming/batch differing frames:',issues);
if(issues.length)process.exitCode=1;
