<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { layerTransition } from "./cube-motion";
  import type { CubePose, Quaternion } from "./cube-motion";
  import { CAMERA_ORIENTATION, displayMoveTokens, reorientedCubePose, reorientedMoveTimeline } from "./cube-orientation";
  import { dragCubeView, nudgeCubeView } from "./cube-drag";
  import type { CubeDragKind, DragPoint } from "./cube-drag";
  import type { MoveStep } from "./move-search-types";
  import type { ProjectedCubeFrame } from "./projected-cube";
  import { CubeRenderer } from "./cube-renderer";
  import { playbackTiming } from "./playback-timing";
  let { pose, sourceTimestamp = 0, timestamp = "", solved = false, steps = [], timeline = [], sourceFrame = 0, sourceFps = 30, playbackSpeed = 1 }: {
    pose?:CubePose;sourceTimestamp?:number;timestamp?:string;solved?:boolean;
    steps?:readonly MoveStep[];timeline?:readonly ProjectedCubeFrame[];sourceFrame?:number;sourceFps?:number;playbackSpeed?:number
  } = $props();
  let rendered=$state.raw<CubePose>();
  let orientation=$state.raw<Quaternion>(CAMERA_ORIENTATION);
  const viewPose=$derived(rendered?reorientedCubePose(rendered,orientation):rendered);
  const reorientedMoves=$derived(timeline.length
    ? reorientedMoveTimeline(timeline,steps,orientation,{fps:sourceFps})
    : steps.map(step=>({frame:step.frame_index,moves:step.state_moves,rotation:false})));
  const moveTokens=$derived(displayMoveTokens(reorientedMoves));
  let surface:HTMLDivElement,orbitRadius=$state(0),hoverRoll=$state(false);
  let drag=$state.raw<{pointer:number;kind:CubeDragKind;start:DragPoint;view:Quaternion;cx:number;cy:number;radius:number}>();
  let dragAnimation=0,pendingOrientation:Quaternion|undefined;
  function finishDrag(commit=true){
    cancelAnimationFrame(dragAnimation);dragAnimation=0;
    if(commit&&pendingOrientation)orientation=pendingOrientation;
    pendingOrientation=undefined;
    const pointer=drag?.pointer;drag=undefined;hoverRoll=false;
    if(pointer!=null&&surface?.hasPointerCapture(pointer))surface.releasePointerCapture(pointer);
  }
  function pointerDown(event:PointerEvent){
    if(!rendered||drag||event.button!==0||!event.isPrimary)return;
    const rect=surface.getBoundingClientRect(),radius=Math.min(rect.width,rect.height)*.42;
    if(radius<=0)return;
    const cx=rect.left+rect.width/2,cy=rect.top+rect.height/2;
    const start:DragPoint=[(event.clientX-cx)/radius,(event.clientY-cy)/radius];
    drag={pointer:event.pointerId,kind:Math.hypot(...start)>=.85?"roll":"orbit",start,view:orientation,cx,cy,radius};
    surface.setPointerCapture(event.pointerId);surface.focus({preventScroll:true});event.preventDefault();
  }
  function pointerMove(event:PointerEvent){
    if(!drag){
      const rect=surface.getBoundingClientRect(),radius=Math.min(rect.width,rect.height)*.42;
      hoverRoll=radius>0&&Math.hypot(event.clientX-rect.left-rect.width/2,event.clientY-rect.top-rect.height/2)>=radius*.85;
      return;
    }
    if(event.pointerId!==drag.pointer)return;
    const point:DragPoint=[(event.clientX-drag.cx)/drag.radius,(event.clientY-drag.cy)/drag.radius];
    pendingOrientation=dragCubeView(drag.view,drag.start,point,drag.kind);
    if(!dragAnimation)dragAnimation=requestAnimationFrame(()=>{
      dragAnimation=0;if(pendingOrientation)orientation=pendingOrientation;pendingOrientation=undefined;
    });
    event.preventDefault();
  }
  function pointerEnd(event:PointerEvent){if(event.pointerId===drag?.pointer)finishDrag();}
  function orientationKeydown(event:KeyboardEvent){
    if(!rendered||event.altKey||event.ctrlKey||event.metaKey)return;
    const next=nudgeCubeView(orientation,event.key,event.shiftKey);
    if(next){finishDrag(false);orientation=next;event.preventDefault();event.stopPropagation();}
  }
  const activeMoveFrame=$derived(moveTokens.filter(token=>token.frame<=sourceFrame).at(-1)?.frame);
  let canvas:HTMLCanvasElement,renderer=$state.raw<CubeRenderer>(),renderError=$state("");
  onMount(()=>{
    try {renderer=new CubeRenderer(canvas);} catch(error){renderError=String(error);return;}
    const resize=new ResizeObserver(()=>{
      renderer?.resize();renderer?.draw(viewPose);orbitRadius=Math.min(canvas.clientWidth,canvas.clientHeight)*.42;
    });resize.observe(surface);
    return ()=>{finishDrag(false);resize.disconnect();renderer?.dispose();};
  });
  $effect(()=>renderer?.draw(viewPose));
  let lastTimestamp:number|undefined;
  $effect(()=>{
    // Animate in the original basis, then reorient each sampled mesh. This
    // preserves signed half-turn paths and in-flight motion when the view changes.
    const target=pose,stamp=sourceTimestamp,duration=playbackTiming(sourceFps,playbackSpeed).transitionMs;
    const previous=untrack(()=>rendered),adjacent=lastTimestamp!=null&&Math.abs(stamp-lastTimestamp)<=0.15;
    lastTimestamp=stamp;
    if(!adjacent||!previous||!target){rendered=target;return;}
    const transition=layerTransition(previous,target),start=performance.now();let animation=0;
    const tick=(now:number)=>{
      const progress=Math.min(1,(now-start)/duration);rendered=transition.sample(progress);
      if(progress<1)animation=requestAnimationFrame(tick);
    };
    animation=requestAnimationFrame(tick);
    // A new target cuts off this curve; the next curve starts at rendered,
    // never at this curve's abandoned endpoint. No animation queue builds up.
    return ()=>cancelAnimationFrame(animation);
  });
</script>

<section class="projected-cube-panel" aria-label="Projected cube">
  <div class="projected-cube-heading"><strong>Reconstruction</strong><span>{timestamp}</span></div>
  <div class="cube-orientation-controls">
    <p id="cube-drag-help">Drag the cube to turn &amp; tilt. Drag the ring to roll.</p>
    <div class="cube-drag-actions">
      <span id="cube-keyboard-help">Arrow keys: turn/tilt · Q/E: roll · Shift: fine</span>
      <button type="button" onclick={()=>{finishDrag(false);orientation=CAMERA_ORIENTATION;}}>Reset view</button>
    </div>
  </div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (The 3D application surface provides keyboard controls for all three axes.) -->
  <div bind:this={surface} class="cube-viewport adjustable" class:dragging={!!drag}
    class:roll-active={hoverRoll||drag?.kind==="roll"} class:empty-cube={!pose||!!renderError}
    role="application" tabindex="0" aria-disabled={!rendered}
    aria-label="Adjust cube orientation"
    aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Q E"
    aria-describedby="cube-drag-help cube-keyboard-help"
    onpointerdown={pointerDown} onpointermove={pointerMove} onpointerup={pointerEnd} onpointercancel={pointerEnd}
    onlostpointercapture={pointerEnd} onpointerleave={()=>{if(!drag)hoverRoll=false;}} onkeydown={orientationKeydown}>
    <canvas bind:this={canvas} class="projected-cube-model" aria-label="Colored reconstructed cube"></canvas>
    {#if !pose||renderError}
      <div class="projected-cube-empty">{renderError||"No reconstructed cube state for this frame"}</div>
    {/if}
    {#if pose&&!renderError}
      <svg class="cube-roll-ring" aria-hidden="true"><circle cx="50%" cy="50%" r={orbitRadius} /></svg>
      <span class="cube-roll-label" style:top={`calc(50% + ${orbitRadius}px - 9px)`} aria-hidden="true">↻ Roll</span>
    {/if}
  </div>
  {#if steps.length || pose&&!renderError}
    <div class="cube-orientation-moves" aria-label="Moves and rotations">
      <strong>{solved ? "Full reconstruction" : "Partial reconstruction"}</strong>
      {#if timeline.length}<span class="cube-notation-standard" aria-label="Move notation axes: U up, F down-left, R down-right">Notation axes: U ↑ · F ↙ · R ↘</span>{/if}
      {#each moveTokens as token}
        <span class:current={token.frame===activeMoveFrame} class:future={token.frame>sourceFrame}
          title={token.rotation?"Cube rotation from tracked orientation":undefined}
          aria-current={token.frame===activeMoveFrame?"step":undefined}>{token.move}</span>
      {:else}<span>Hold</span>{/each}
    </div>
  {/if}
</section>
