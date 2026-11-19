<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { layerTransition } from "./cube-motion";
  import type { CubePose, Quaternion } from "./cube-motion";
  import { CAMERA_ORIENTATION, reorientedCubePose, reorientedMoveTimeline, rotateCubeView } from "./cube-orientation";
  import type { MoveStep } from "./move-search-types";
  import type { ProjectedCubeFrame } from "./projected-cube";
  import { CubeRenderer } from "./cube-renderer";
  let { pose, sourceTimestamp = 0, timestamp = "", solved = false, steps = [], timeline = [], sourceFrame = 0 }: {
    pose?:CubePose;sourceTimestamp?:number;timestamp?:string;solved?:boolean;
    steps?:readonly MoveStep[];timeline?:readonly ProjectedCubeFrame[];sourceFrame?:number
  } = $props();
  let smooth=$state(true),rendered=$state.raw<CubePose>();
  let orientation=$state.raw<Quaternion>(CAMERA_ORIENTATION);
  const viewPose=$derived(rendered?reorientedCubePose(rendered,orientation):rendered);
  const reorientedMoves=$derived(reorientedMoveTimeline(timeline,steps,orientation));
  const adjusted=$derived(Math.abs(orientation[3])<1-1e-10);
  function reorient(move:string){
    if(rendered)orientation=rotateCubeView(orientation,rendered.rotation,move);
  }
  const activeMoveFrame=$derived(reorientedMoves.filter(step=>step.frame<=sourceFrame).at(-1)?.frame);
  let canvas:HTMLCanvasElement,renderer=$state.raw<CubeRenderer>(),renderError=$state("");
  onMount(()=>{
    try {renderer=new CubeRenderer(canvas);} catch(error){renderError=String(error);return;}
    const resize=new ResizeObserver(()=>{renderer?.resize();renderer?.draw(viewPose);});resize.observe(canvas);
    return ()=>{resize.disconnect();renderer?.dispose();};
  });
  $effect(()=>renderer?.draw(viewPose));
  let lastTimestamp:number|undefined;
  $effect(()=>{
    // Animate in the original basis, then reorient each sampled mesh. This
    // preserves signed half-turn paths and in-flight motion when the view changes.
    const target=pose,stamp=sourceTimestamp,enabled=smooth;
    const previous=untrack(()=>rendered),adjacent=lastTimestamp!=null&&Math.abs(stamp-lastTimestamp)<=0.15;
    lastTimestamp=stamp;
    if(!enabled||!adjacent||!previous||!target){rendered=target;return;}
    const transition=layerTransition(previous,target),start=performance.now();let animation=0;
    const tick=(now:number)=>{
      const progress=Math.min(1,(now-start)/100);rendered=transition.sample(progress);
      if(progress<1)animation=requestAnimationFrame(tick);
    };
    animation=requestAnimationFrame(tick);
    // A new target cuts off this curve; the next curve starts at rendered,
    // never at this curve's abandoned endpoint. No animation queue builds up.
    return ()=>cancelAnimationFrame(animation);
  });
</script>

<section class="projected-cube-panel" aria-label="Projected cube">
  <div class="projected-cube-heading"><strong>Projected cube</strong><span>{timestamp}</span></div>
  <label class="cube-smoothing"><input type="checkbox" bind:checked={smooth} /> Smooth cube motion</label>
  <div class="cube-orientation-controls">
    <div class="cube-orientation-rotations" aria-label="Reorient cube">
      {#each ["x","x'","y","y'","z","z'"] as move}
        <button type="button" onclick={()=>reorient(move)} disabled={!rendered} aria-label={`Reorient cube by ${move}`}>{move}</button>
      {/each}
      <button type="button" onclick={()=>orientation=CAMERA_ORIENTATION}>Reset to camera orientation</button>
    </div>
    <p>Orientation: {adjusted?"Adjusted":"Camera"}</p>
    <p>U is the face nearest display-up; F faces you. Controls and moves use these faces.</p>
  </div>
  <canvas bind:this={canvas} class="projected-cube-model" class:empty-cube={!pose||!!renderError} aria-label="Colored reconstructed cube"></canvas>
  {#if pose&&!renderError}
    <p class="projected-cube-caption">{solved ? "Reconstructed sticker colors" : "Candidate sticker colors"}</p>
    <div class="cube-orientation-moves" aria-label="Moves and rotations">
      <strong>Moves and rotations</strong>
      {#each reorientedMoves as step}
        <span class:current={step.frame===activeMoveFrame} class:future={step.frame>sourceFrame}
          title={step.rotation?"Cube rotation from tracked orientation":undefined}
          aria-current={step.frame===activeMoveFrame?"step":undefined}>{step.moves.join(" ")}</span>
      {:else}<span>Hold</span>{/each}
    </div>
  {:else}
    <div class="projected-cube-empty">{renderError||"No reconstructed cube state for this frame"}</div>
  {/if}
</section>
