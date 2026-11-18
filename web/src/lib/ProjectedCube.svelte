<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { layerTransition } from "./cube-motion";
  import type { CubePose } from "./cube-motion";
  import { CubeRenderer } from "./cube-renderer";
  let { pose, sourceTimestamp = 0, timestamp = "", solved = false }: {
    pose?:CubePose;sourceTimestamp?:number;timestamp?:string;solved?:boolean
  } = $props();
  let smooth=$state(true),rendered=$state.raw<CubePose>();
  let canvas:HTMLCanvasElement,renderer=$state.raw<CubeRenderer>(),renderError=$state("");
  onMount(()=>{
    try {renderer=new CubeRenderer(canvas);} catch(error){renderError=String(error);return;}
    const resize=new ResizeObserver(()=>{renderer?.resize();renderer?.draw(rendered);});resize.observe(canvas);
    return ()=>{resize.disconnect();renderer?.dispose();};
  });
  $effect(()=>renderer?.draw(rendered));
  let lastTimestamp:number|undefined;
  $effect(()=>{
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
  <canvas bind:this={canvas} class="projected-cube-model" class:empty-cube={!pose||!!renderError} aria-label="Colored reconstructed cube"></canvas>
  {#if pose&&!renderError}
    <p class="projected-cube-caption">{solved ? "Reconstructed sticker colors" : "Candidate sticker colors"}</p>
  {:else}
    <div class="projected-cube-empty">{renderError||"No reconstructed cube state for this frame"}</div>
  {/if}
</section>
