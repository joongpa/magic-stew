<script lang="ts">
  // Isolated UI fixture: exercises long histories and old/new replay modes
  // without requesting camera permission or changing the user's timer session.
  import CubeTimer from '../src/lib/CubeTimer.svelte';
  import LatestTimerReplay from '../src/lib/LatestTimerReplay.svelte';
  import type {TimerReconstructionResult,TimerSolve} from '../src/lib/cube-timer';
  import type {TimerReplay} from '../src/lib/timer-replay';
  import {TimerReplayCache,type ReplayStorage} from '../src/lib/timer-replay-cache';
  const storageKey='live-recognition.timer.layout-check';
  const result:TimerReconstructionResult={status:'solved',moves:["U'","R'"],
    config:{scramble:'R U',fps:30,requested_start_seconds:0,source_origin_seconds:0,source_end_seconds:2},
    result:{status:'solved',steps:[{frame_index:0,geometry_option_index:0,state_moves:[]},
      {frame_index:25,geometry_option_index:0,state_moves:["U'"]},{frame_index:50,geometry_option_index:0,state_moves:["R'"]}],
      frames_received:60,frames_prepared:60,frames_searchable:60,blocked:0,pending:0,ended:true,
      solved_frame_index:50,stats:{evaluated_states:10,furthest_frames:60,branch_stage:0}}};
  const history:TimerSolve[]=Array.from({length:40},(_,i)=>({id:`layout-${i+1}`,scramble:'R U',elapsedMs:10000+i*123,
    startedAt:'2026-11-19T12:00:00Z',penalty:'none',reconstruction:{status:'solved',moves:result.moves,config:result.config}}));
  localStorage.setItem(storageKey,JSON.stringify(history));
  const reconstructions=Object.fromEntries(history.map(solve=>[solve.id,result]));
  const geometryFrames=Array.from({length:60},(_,i)=>({timestamp:i/30,result:{fittedPose:{
    sourceFrame:i,bank:[{axis:0,params:[.3,-.4,.15+i*.002],angles:[0,0,0],flat_fallback:false,score:0,nearest:[],valid:[]}],
    pose:{rvec:[.3,-.4,.15+i*.002],tvec:[0,0,500]},pose_flag_reason:null,wireframe:[],bootstrap:false,message:'',elapsedMs:0}}}));
  let replays=$state.raw<Record<string,TimerReplay>>({});
  let latestReplay=$state.raw<TimerReplay>(),latestPosition=$state<number>();
  let seekRequest=$state<{timestamp:number;revision:number}>(),seekRevision=0;
  function seekLatest(id:string,timestamp:number){
    if(id!==latestReplay?.id)return false;
    seekRequest={timestamp,revision:++seekRevision};return true;
  }
  const disk=new Map<string,TimerReplay>();
  const storage:ReplayStorage={async save(clip){disk.set(clip.id,clip);},async load(id){return disk.get(id);},async remove(id){disk.delete(id);}};
  const cache=new TimerReplayCache(storage,clip=>replays=clip?{[clip.id]:clip}:{});
  async function prepare(){
    const image=await fetch('/sample.jpg').then(response=>response.blob());
    for(const solve of history){
      cache.begin(solve.id);latestReplay={id:solve.id,start:0,end:2,clockFps:30,complete:true,missingImages:0,
        frames:geometryFrames.map(frame=>({...frame,image}))};cache.publish(latestReplay);cache.finish(solve.id,result);
    }
  }
  void prepare();
</script>
<div class="shell">
  <div class="beta-banner">Timer layout test fixture · Uses sample data only</div>
  <header class="topbar"><span class="brand"><span class="brand-logo-frame"><img class="brand-logo" src="/magic-stew-logo.png" alt="" width="64" height="64" /></span><span class="brand-label">Magic Stew</span></span><span>Cube timer</span></header>
  <main><div class="timer-layout" class:with-inline-replay={!!latestReplay}><div class="timer-controls">
    <CubeTimer active={true} cameraReady={false} latestReplayPosition={latestPosition} onlatestseek={seekLatest} {reconstructions} {replays} {storageKey}
      onreplayselect={id=>{void cache.select(id);}} onstart={()=>{}} onstop={()=>{}} onbusychange={()=>{}} />
  </div><div class="workspace timer-workspace"><section class="viewer" aria-label="Camera panel">
    <div class="viewer-toolbar">Camera preview</div><div class="reconstruction-pair with-timer-replay">
      <div class="stage"><img src="/sample.jpg" alt="Sample camera output" style="position:absolute;width:100%;height:100%;object-fit:contain" /></div>
      <LatestTimerReplay clip={latestReplay} reconstruction={result} {seekRequest} onpositionchange={timestamp=>latestPosition=timestamp} />
    </div>
    <div class="viewer-statusbar">30.0 FPS</div>
  </section></div></div></main>
</div>
