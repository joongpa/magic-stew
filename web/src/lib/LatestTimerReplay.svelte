<script lang="ts">
  import TimerReplay from './TimerReplay.svelte';
  import type {TimerReplay as ReplayClip} from './timer-replay';
  import type {TimerReconstructionResult} from './cube-timer';
  let {clip,reconstruction,disabled=false,seekRequest,onpositionchange}:{
    clip?:ReplayClip;reconstruction?:TimerReconstructionResult;disabled?:boolean;
    seekRequest?:{timestamp:number;revision:number};onpositionchange?:(timestamp:number)=>void;
  }=$props();
</script>

<aside class="timer-inline-replay" aria-label="Latest solve replay">
  <div class="timer-inline-heading"><strong>Latest solve replay</strong></div>
  {#if clip && reconstruction}
    {#key clip.id}<TimerReplay {clip} {reconstruction} {disabled} {seekRequest} {onpositionchange} showRecording={false} />{/key}
  {:else}
    <div class="timer-inline-empty">{disabled?'Replay will be available after the solve.':'Complete a solve to view its replay.'}</div>
  {/if}
</aside>
