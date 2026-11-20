<script lang="ts">
  import {onMount} from 'svelte';
  import {formatTimer} from './cube-timer';
  import type {MoveStep} from './move-search-types';
  import {turnEvents,gaussianTpsAt,tpsCurve,committedTurnSpan,tpsSummary,type TurnEvent} from './tps';
  let {steps=[],turns,moves,fps,start,end,solveSeconds,cursor,complete=false,compact=false,onseek,label='TPS over time'}:{
    steps?:readonly MoveStep[];turns?:readonly TurnEvent[];moves?:readonly string[];fps:number;start:number;end:number;solveSeconds?:number;cursor?:number;complete?:boolean;compact?:boolean;
    onseek?:(timestamp:number)=>void;label?:string;
  }=$props();
  let surface:HTMLDivElement;
  let width=$state(640),hover=$state<number>();
  const height=$derived(compact?100:156);
  const left=32,right=12,top=10,bottom=24;
  const plotWidth=$derived(Math.max(1,width-left-right)),plotHeight=$derived(height-top-bottom);
  const events=$derived(complete?(turns??turnEvents(steps,fps)):[]);
  const count=$derived(events.length||turnEvents([{frame_index:0,state_moves:[...(moves??[])]}],1).length);
  const summary=$derived(tpsSummary(events,solveSeconds??committedTurnSpan(steps,fps),count));
  const points=$derived(tpsCurve(events,start,end));
  const peak=$derived(Math.max(0,...points.map(point=>point.tps)));
  const ceiling=$derived(Math.max(5,Math.ceil(peak/5)*5));
  const duration=$derived(Math.max(0,end-start));
  const x=(time:number)=>left+(time-start)/Math.max(duration,0.001)*plotWidth;
  const y=(rate:number)=>top+(1-rate/ceiling)*plotHeight;
  const path=$derived(points.map((point,i)=>`${i?'L':'M'}${x(point.timestamp).toFixed(2)},${y(point.tps).toFixed(2)}`).join(' '));
  const area=$derived(path?`${path} L${x(end)},${y(0)} L${x(start)},${y(0)} Z`:'');
  const highlighted=$derived(hover===undefined?cursor===undefined?undefined:Math.max(start,Math.min(end,cursor)):Math.max(start,Math.min(end,hover)));
  const rate=$derived(highlighted===undefined?undefined:gaussianTpsAt(events,highlighted));
  const ticks=$derived([0,0.25,0.5,0.75,1]);
  onMount(()=>{
    const observer=new ResizeObserver(()=>{width=Math.max(1,surface.clientWidth);});
    observer.observe(surface);width=Math.max(1,surface.clientWidth);return()=>observer.disconnect();
  });
  function pointer(event:PointerEvent){
    const rect=event.currentTarget instanceof SVGElement?event.currentTarget.getBoundingClientRect():undefined;
    if(!rect||!duration)return;
    hover=start+Math.max(0,Math.min(1,(event.clientX-rect.left-left)/plotWidth))*duration;
  }
  function keydown(event:KeyboardEvent){
    if(event.altKey||event.ctrlKey||event.metaKey)return;
    let next=highlighted??start;
    if(event.key==='ArrowLeft')next-=.05;
    else if(event.key==='ArrowRight')next+=.05;
    else if(event.key==='Home')next=start;
    else if(event.key==='End')next=end;
    else if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();if(highlighted!==undefined)onseek?.(highlighted);return;}
    else return;
    event.preventDefault();event.stopPropagation();hover=Math.max(start,Math.min(end,next));
  }
</script>

<section class="tps-graph" class:compact aria-label={label}>
  <div class="tps-heading"><strong title="Half turns count once. Gaussian smoothing with a 0.2 second standard deviation.">TPS</strong>
    <span class="tps-readout" aria-live="off">{#if complete && rate!==undefined && events.length}<strong>{rate.toFixed(1)} TPS</strong><span>{((highlighted??start)-start).toFixed(2)}s</span>{/if}</span>
  </div>
  <div class="tps-surface" bind:this={surface}>
    {#if complete}
    <svg viewBox={`0 0 ${width} ${height}`} style:height={`${height}px`} role="slider" tabindex="0"
      aria-label={label} aria-valuemin={0} aria-valuemax={duration} aria-valuenow={(highlighted??start)-start}
      aria-valuetext={`${((highlighted??start)-start).toFixed(2)} seconds, ${(rate??0).toFixed(1)} TPS`}
      aria-disabled={!duration} onpointermove={pointer} onpointerdown={pointer} onpointerleave={()=>hover=undefined}
      onclick={()=>{if(highlighted!==undefined)onseek?.(highlighted);}} onkeydown={keydown} onblur={()=>hover=undefined}>
      {#each [0,ceiling/2,ceiling] as tick}<line class="tps-grid" x1={left} x2={width-right} y1={y(tick)} y2={y(tick)} />
        <text class="tps-axis" x={left-6} y={y(tick)+3} text-anchor="end">{tick}</text>{/each}
      {#each ticks as tick}<text class="tps-axis" x={left+plotWidth*tick} y={height-7} text-anchor={tick===0?'start':tick===1?'end':'middle'}>{(duration*tick).toFixed(duration<10?1:0)}s</text>{/each}
      {#if events.length}
        <path class="tps-area" d={area} /><path class="tps-line" d={path} />
        {#if cursor!==undefined && cursor>=start && cursor<=end}<line class="tps-playhead" x1={x(cursor)} x2={x(cursor)} y1={top} y2={y(0)} />{/if}
        {#if highlighted!==undefined && rate!==undefined}<line class="tps-crosshair" x1={x(highlighted)} x2={x(highlighted)} y1={top} y2={y(0)} />
          <circle class="tps-dot" cx={x(highlighted)} cy={y(rate)} r="3.5" />{/if}
      {:else}<text class="tps-empty" x={left+plotWidth/2} y={top+plotHeight/2} text-anchor="middle">{count?'Turn timestamps unavailable':'No reconstructed turns yet'}</text>{/if}
    </svg>
    {:else}<div class="tps-unavailable" style:height={`${height}px`} role="status">TPS available after a full reconstruction.</div>{/if}
  </div>
  {#if complete}<dl class="tps-summary">
    <div><dt title={solveSeconds===undefined?'First to last reconstructed move':'Measured solve time'}>Solve time</dt><dd>{summary.seconds===undefined?'—':formatTimer(summary.seconds*1000)}</dd></div>
    <div><dt>Total moves</dt><dd>{summary.moves}</dd></div>
    <div><dt>Avg. TPS</dt><dd>{summary.averageTps===undefined?'—':summary.averageTps.toFixed(2)}</dd></div>
  </dl>{/if}
</section>
