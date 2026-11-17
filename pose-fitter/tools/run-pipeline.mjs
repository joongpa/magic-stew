import{readFileSync,writeFileSync}from'node:fs';
import{performance}from'node:perf_hooks';
import{createHash}from'node:crypto';
import createOpenCv from '../target/opencv-pnp.mjs';
import{loadCpuFitter}from'./cpu-wasm-host.mjs';
const [fixturePath,outputPath,...options]=process.argv.slice(2);
const bytes=readFileSync(fixturePath),data=JSON.parse(bytes),cv=await createOpenCv();
const wasm=readFileSync(new URL('../target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm',import.meta.url));
const events=[];
const e=await loadCpuFitter(wasm,Buffer.from(JSON.stringify({geometry:data.geometry,workloads:[],probes:[]})),cv,
    {onDiagnostic:message=>events.push({message,atMs:performance.now()})});
const option=name=>options.includes(name)?options[options.indexOf(name)+1]:undefined;
const stage=option('--stage');
if(stage&&!['startup','depth_targets','initial','neighbors','orientation','recovery'].includes(stage))throw new Error('Unknown CPU pipeline stage');
const input={fps:data.fps,turning_start_frame:data.turning_start_frame,expected_center_colors:data.expected_center_colors,
  capture_stages:options.includes('--capture-stages'),frames:data.frames.map(f=>({
    frame:f.frame,width:f.width,height:f.height,observations:f.observations,
    landmark_presence:f.landmark_presence,colors:f.colors,pixel_scale:f.pixel_scale,
  }))};
const checkpoint=option('--checkpoint');
let state=checkpoint?JSON.parse(readFileSync(checkpoint)):{};
if(checkpoint&&state.diagnostics?.fixture_sha256&&state.diagnostics.fixture_sha256!==createHash('sha256').update(bytes).digest('hex'))throw new Error('Checkpoint belongs to a different input fixture');
if(option('--checkpoint-stage'))state=state.stages[option('--checkpoint-stage')];
const started=performance.now();
const value=stage?e.call('pipeline_stage_json',{stage,input,state}):e.call('pipeline_json',input);
if(value.Err)throw new Error(value.Err);
const result=value.Ok??value;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const ended=performance.now();
const stageTimes=Object.fromEntries(events.filter(e=>e.message.startsWith('CPU pipeline: ')).map((e,i,list)=>[e.message.slice(14),(list[i+1]?.atMs??ended)-e.atMs]));
result.diagnostics={elapsedMs:ended-started,stageTimesMs:stageTimes,runtime:process.version,stage:stage??'all',
    fixture_sha256:hash(bytes),wasm_sha256:hash(wasm),opencv_sha256:hash(readFileSync(new URL('../target/opencv-pnp.wasm',import.meta.url)))};
writeFileSync(outputPath,JSON.stringify(result));
console.log(data.name,stage??'all',result.diagnostics.elapsedMs.toFixed(1),'ms',outputPath);
