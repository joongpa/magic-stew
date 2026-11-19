<script lang="ts">
  import {untrack} from 'svelte';
  import type {TimerReplay} from './timer-replay';
  import type {TimerReconstructionResult} from './cube-timer';
  import {buildColorTimeline} from './cube-colors';
  import {buildProjectedCubeTimeline,projectedCubeAtFrame} from './projected-cube';
  import {poseIsFlagged,projectedWireframe} from './reconstruction-view';
  import {reviewIndex,steppedTimestamp,frameArrow} from './frame-review';
  import {PLAYBACK_SPEEDS,type PlaybackSpeed} from './playback-timing';
  import {timecode} from './solve-range';
  import ProjectedCube from './ProjectedCube.svelte';

  let {clip,reconstruction,disabled=false}:{clip:TimerReplay;reconstruction:TimerReconstructionResult;disabled?:boolean}=$props();
  let canvas:HTMLCanvasElement,surface:HTMLElement;
  let position=$state(0),playing=$state(false),speed=$state<PlaybackSpeed>(1),showPose=$state(true);
  let drawError=$state('');
  const steps=$derived(reconstruction.result?.steps??[]);
  const solved=$derived(reconstruction.result?.status==='solved'||reconstruction.result?.status==='already_solved');
  const start=$derived(clip.frames[0]?.timestamp??clip.start);
  const end=$derived(clip.frames.at(-1)?.timestamp??start);
  const index=$derived(reviewIndex(clip.frames,position));
  const current=$derived(clip.frames[index]);
  const frame=$derived(current?.result.fittedPose?.sourceFrame??Math.round(position*clip.clockFps));
  const step=$derived(steps.find(step=>step.frame_index===frame));
  const timeline=$derived.by(()=>{
    try{return buildProjectedCubeTimeline(clip.frames,buildColorTimeline(reconstruction.config?.scramble??'',steps),
      {fps:clip.clockFps,solved,steps});}catch{return [];}
  });
  const pose=$derived(solved||frame<=(steps.at(-1)?.frame_index??-Infinity)?projectedCubeAtFrame(timeline,frame):undefined);
  $effect(()=>{const initial=start;untrack(()=>{playing=false;position=initial;});});
  $effect(()=>{if(disabled)playing=false;});
  $effect(()=>{
    if(!playing)return;
    const origin=untrack(()=>position),wall=performance.now(),rate=speed,limit=end;
    let animation=0;
    const tick=(now:number)=>{
      position=Math.min(limit,origin+(now-wall)*rate/1000);
      if(position>=limit)playing=false;else animation=requestAnimationFrame(tick);
    };
    animation=requestAnimationFrame(tick);return()=>cancelAnimationFrame(animation);
  });
  $effect(()=>{
    const row=current,target=canvas,enabled=showPose,choice=step;
    if(!target||!row)return;
    let stale=false;
    void (async()=>{
      if(!row.image){drawError='Camera image unavailable for this frame';target.getContext('2d')?.clearRect(0,0,target.width,target.height);return;}
      try{
        const image=await createImageBitmap(row.image);
        try{
          if(stale)return;
          target.width=image.width;target.height=image.height;
          const context=target.getContext('2d')!;context.drawImage(image,0,0);drawError='';
          const fitted=row.result.fittedPose;
          if(enabled&&fitted?.pose&&!poseIsFlagged(fitted,choice?.pose_flagged)){
            context.strokeStyle='#64f6dd';context.lineWidth=Math.max(1.5,image.width/650);context.lineJoin='round';
            for(const polygon of projectedWireframe(fitted,choice?.geometry_option_index)){
              if(!polygon.length)continue;context.beginPath();context.moveTo(...polygon[0]);
              for(const point of polygon.slice(1))context.lineTo(...point);context.closePath();context.stroke();
            }
          }
        }finally{image.close();}
      }catch(error){if(!stale)drawError=error instanceof Error?error.message:String(error);}
    })();return()=>{stale=true;};
  });
  function seek(timestamp:number){playing=false;position=Math.max(start,Math.min(end,timestamp));}
  function advance(direction:-1|1){const next=steppedTimestamp(clip.frames,position,direction);if(next!==undefined)seek(next);}
  function toggle(){if(position>=end)position=start;playing=!playing;}
  function keydown(event:KeyboardEvent){
    if(disabled||event.defaultPrevented||!(event.target instanceof Element)||!surface?.contains(event.target))return;
    const direction=frameArrow(event,event.target);if(direction){event.preventDefault();advance(direction);}
  }
</script>

<svelte:window onkeydown={keydown} />
<section bind:this={surface} class="timer-replay" aria-label="Solve replay">
  <div class="timer-section-heading"><h3>Solve replay</h3><span>{clip.frames.length} processed frames{clip.complete?'':' · Finishing capture'}</span></div>
  <div class="reconstruction-pair with-projection">
    <div class="stage video-view"><canvas bind:this={canvas} aria-label="Timer replay camera frame"></canvas></div>
    <ProjectedCube {pose} sourceTimestamp={current?.timestamp??position} timestamp={timecode(position-clip.start)} {solved}
      {timeline} {steps} sourceFrame={frame} sourceFps={clip.clockFps} playbackSpeed={playing?speed:1} />
  </div>
  <div class="transport"><div class="transport-actions">
    <button class="small" disabled={disabled||clip.frames.length<2} onclick={toggle}>{playing?'Pause':'Play'}</button>
    <label class="playback-speed">Speed <select aria-label="Timer replay speed" bind:value={speed} disabled={disabled}>
      {#each PLAYBACK_SPEEDS as value}<option value={value}>{value}×</option>{/each}</select></label>
    <label class="toggle"><input type="checkbox" bind:checked={showPose} />Pose overlay</label>
  </div></div>
  <div class="scrubber"><span>{timecode(position-clip.start)}</span>
    <input type="range" aria-label="Timer replay position" min={start} max={end} step="0.001" value={position}
      disabled={disabled||clip.frames.length<2} oninput={event=>seek(Number(event.currentTarget.value))} />
    <span>{timecode(end-clip.start)}</span></div>
  <div class="frame-review" aria-label="Timer frame review">
    <button class="small" aria-label="Previous timer frame" disabled={disabled||index<=0} onclick={()=>advance(-1)}>←</button>
    <span>Frame {Math.max(0,index+1)} / {clip.frames.length}</span>
    <button class="small" aria-label="Next timer frame" disabled={disabled||index>=clip.frames.length-1} onclick={()=>advance(1)}>→</button>
    <span class="frame-key-hint">← / → to step frames</span>
  </div>
  {#if drawError}<p class="timer-error" role="status">{drawError}</p>{/if}
</section>
