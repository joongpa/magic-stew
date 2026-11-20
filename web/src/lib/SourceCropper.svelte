<script lang="ts">
  import {onMount, untrack} from 'svelte';
  import {FULL_CROP, boundedCrop, cropFromPoints, cropPixels, dragCrop} from './source-crop';
  import type {CropHandle, SourceCrop} from './source-crop';

  let {url, width, height, initial, kind, onapply, oncancel}: {
    url: string; width: number; height: number; initial?: SourceCrop; kind: 'image' | 'video';
    onapply(crop: SourceCrop | undefined): void; oncancel(): void;
  } = $props();
  let dialog: HTMLDialogElement, surface: HTMLDivElement;
  let selection = $state<SourceCrop>(untrack(() => boundedCrop(initial ?? FULL_CROP)));
  let drawNew = $state(untrack(() => !initial));
  let gesture: {id: number; handle: CropHandle | 'draw'; x: number; y: number; start: SourceCrop} | undefined;
  const handles: CropHandle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
  const handleNames: Record<string, string> = {nw: 'top left', n: 'top', ne: 'top right', e: 'right', se: 'bottom right', s: 'bottom', sw: 'bottom left', w: 'left'};
  const pixels = $derived(cropPixels(selection, width, height) ?? {x: 0, y: 0, width, height});
  onMount(() => {dialog.showModal();});

  function point(event: PointerEvent) {
    const box = surface.getBoundingClientRect();
    return {x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
      y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))};
  }
  function begin(event: PointerEvent) {
    if (event.button !== 0) return;
    event.preventDefault();
    const p = point(event);
    const button = (event.target as Element).closest<HTMLElement>('[data-crop-handle]');
    const handle = button?.dataset.cropHandle as CropHandle | undefined;
    gesture = {id: event.pointerId, handle: handle && handle !== 'move' ? handle : drawNew || !handle ? 'draw' : 'move', ...p, start: {...selection}};
    if (gesture.handle === 'draw') selection = cropFromPoints(p, p);
    surface.setPointerCapture(event.pointerId);
    button?.focus({preventScroll: true});
  }
  function move(event: PointerEvent) {
    if (!gesture || event.pointerId !== gesture.id) return;
    const p = point(event);
    selection = gesture.handle === 'draw' ? cropFromPoints(gesture, p)
      : dragCrop(gesture.start, gesture.handle, p.x - gesture.x, p.y - gesture.y);
  }
  function end(event: PointerEvent) {
    if (!gesture || event.pointerId !== gesture.id) return;
    move(event); gesture = undefined; drawNew = false;
    if (surface.hasPointerCapture(event.pointerId)) surface.releasePointerCapture(event.pointerId);
  }
  function keyboard(event: KeyboardEvent, handle: CropHandle) {
    const step = event.shiftKey ? 0.02 : 0.002;
    const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
    const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
    if (!dx && !dy) return;
    event.preventDefault(); selection = dragCrop(selection, handle, dx, dy); drawNew = false;
  }
  function apply() {
    const crop = cropPixels(selection, width, height);
    onapply(crop ? {x: crop.x / width, y: crop.y / height, width: crop.width / width, height: crop.height / height} : undefined);
  }
</script>

<dialog bind:this={dialog} class="source-cropper" aria-labelledby="source-crop-title" oncancel={event => {event.preventDefault(); oncancel();}}>
  <div class="crop-heading"><h2 id="source-crop-title">Crop {kind}</h2><span>{pixels.width} × {pixels.height} px</span></div>
  <p class="crop-instructions">{drawNew ? 'Drag over the image to select an area.' : 'Drag inside the box to move it, or drag the handles to resize.'} Arrow keys adjust the focused control; Shift makes larger steps.</p>
  <div class="crop-surface" class:draw-new={drawNew} bind:this={surface} role="group" aria-label="Crop preview"
    onpointerdown={begin} onpointermove={move} onpointerup={end} onpointercancel={end}>
    <img src={url} alt="Full source for selecting the crop" draggable="false" />
    <div class="crop-selection" style:left={`${selection.x * 100}%`} style:top={`${selection.y * 100}%`}
      style:width={`${selection.width * 100}%`} style:height={`${selection.height * 100}%`}>
      <button type="button" class="crop-move" data-crop-handle="move" aria-label="Move crop selection" onkeydown={event => keyboard(event, 'move')}></button>
      <div class="crop-grid" aria-hidden="true"></div>
      {#each handles as handle}<button type="button" class="crop-handle {handle}" data-crop-handle={handle}
        aria-label={`Resize crop ${handleNames[handle]}`} onkeydown={event => keyboard(event, handle)}></button>{/each}
    </div>
  </div>
  <p class="crop-note">{kind === 'video' ? 'This crop applies to every frame and locks when reconstruction starts.' : 'Only this area will be sent to the detector.'}</p>
  <div class="crop-actions">
    <button type="button" class="small" onclick={() => {selection = {...FULL_CROP}; drawNew = true;}}>Reset to full frame</button>
    <div><button type="button" class="small" onclick={oncancel}>Cancel</button><button type="button" class="primary" onclick={apply}>Apply crop</button></div>
  </div>
</dialog>

<style>
  .source-cropper {width:min(880px,calc(100vw - 32px));max-height:94dvh;padding:22px;border:1px solid #4a5a41;border-radius:12px;background:#151e17;color:#dae8d0;overflow:auto;}
  .source-cropper::backdrop {background:#030906d9;}
  .crop-heading,.crop-actions,.crop-actions>div {display:flex;align-items:center;justify-content:space-between;gap:12px;}
  h2 {margin:0;font-size:18px;}
  .crop-heading span {font-size:12px;font-variant-numeric:tabular-nums;color:#c6f69b;}
  .crop-instructions,.crop-note {font-size:12px;line-height:1.5;color:#a4b19a;margin:12px 0;}
  .crop-surface {position:relative;width:fit-content;max-width:100%;margin:auto;overflow:hidden;touch-action:none;user-select:none;line-height:0;cursor:crosshair;}
  .crop-surface img {display:block;max-width:100%;max-height:60dvh;width:auto;height:auto;pointer-events:none;}
  .crop-selection {position:absolute;box-sizing:border-box;border:2px solid #c6f69b;box-shadow:0 0 0 200vmax #020805a6;}
  .crop-move {position:absolute;inset:0;width:100%;height:100%;background:transparent;border:0;border-radius:0;cursor:move;}
  .draw-new .crop-move {cursor:crosshair;}
  .crop-grid {position:absolute;inset:0;pointer-events:none;background:linear-gradient(to right,transparent 33.1%,#ffffff35 33.1%,#ffffff35 33.5%,transparent 33.5%,transparent 66.4%,#ffffff35 66.4%,#ffffff35 66.8%,transparent 66.8%),linear-gradient(to bottom,transparent 33.1%,#ffffff35 33.1%,#ffffff35 33.5%,transparent 33.5%,transparent 66.4%,#ffffff35 66.4%,#ffffff35 66.8%,transparent 66.8%);}
  .crop-handle {position:absolute;padding:0;width:24px;height:24px;border:0;border-radius:0;background:transparent;transform:translate(-50%,-50%);}
  .crop-handle::after {content:'';position:absolute;width:12px;height:12px;inset:6px;background:#c6f69b;border:1px solid #14200d;box-sizing:border-box;}
  .nw,.n,.ne {top:0;}.sw,.s,.se {top:100%;}.w,.e {top:50%;}
  .nw,.w,.sw {left:0;}.n,.s {left:50%;}.ne,.e,.se {left:100%;}
  .nw,.se {cursor:nwse-resize;}.ne,.sw {cursor:nesw-resize;}.n,.s {cursor:ns-resize;}.w,.e {cursor:ew-resize;}
  .crop-move:focus-visible,.crop-handle:focus-visible {outline:2px solid #64f6dd;outline-offset:-3px;}
  .crop-actions {margin-top:16px;flex-wrap:wrap;}.crop-actions>div {margin-left:auto;}
  @media(max-width:520px) {.source-cropper{padding:14px;}.crop-heading{align-items:flex-start;}.crop-heading span{font-size:11px;}.crop-instructions{font-size:11px;}}
</style>
