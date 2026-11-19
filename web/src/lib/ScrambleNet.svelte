<script lang="ts">
  import { onMount } from 'svelte';
  import type { TwistyPlayer } from 'cubing/twisty';
  let { scramble }: { scramble: string } = $props();
  let host: HTMLDivElement;
  let player = $state<TwistyPlayer>();
  let error = $state('');
  $effect(() => {
    if (player) {
      player.alg = scramble;
      player.timestamp = 'end';
    }
  });
  onMount(() => {
    let disposed = false;
    void import('cubing/twisty').then(({ TwistyPlayer }) => {
      if (disposed) return;
      const view = new TwistyPlayer({ puzzle: '3x3x3', visualization: '2D', alg: scramble,
        controlPanel: 'none', background: 'none', viewerLink: 'none' });
      view.timestamp = 'end';
      view.style.width = '100%'; view.style.height = '100%';
      host.replaceChildren(view); player = view;
    }).catch(cause => { if (!disposed) error = cause instanceof Error ? cause.message : String(cause); });
    return () => { disposed = true; host.replaceChildren(); };
  });
</script>

<div class="scramble-net" bind:this={host} role="img" aria-label={`Cube net after scramble: ${scramble}`}></div>
{#if error}<p class="timer-error" role="alert">Could not draw scramble: {error}</p>{/if}
