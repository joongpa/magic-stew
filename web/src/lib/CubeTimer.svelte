<script lang="ts">
  import { onMount } from 'svelte';
  import { CubeTimerClock, TIMER_HOLD_MS, formatTimer, solveTime, timerAverage, readTimerSolves, compactTimerReconstruction,timerReconstructionStatus,timerSessionRows } from './cube-timer';
  import type { Penalty, TimerSolve, TimerReconstructionResult } from './cube-timer';
  import ScrambleNet from './ScrambleNet.svelte';
  import TpsGraph from './TpsGraph.svelte';
  import TimerReplay from './TimerReplay.svelte';
  import type {TimerReplay as ReplayClip} from './timer-replay';

  let { active, cameraReady, reconstructions,latestReplayPosition,onlatestseek,replays={}, replayLoading=false,replayError='',onreplayselect,ondelete,onclearsession, onstart, onstop, onbusychange, ondownloaddiagnostics, storageKey = 'live-recognition.timer.v1' }: {
    active: boolean; cameraReady: boolean; reconstructions: Record<string, TimerReconstructionResult>;
    latestReplayPosition?:number;onlatestseek?:(id:string,timestamp:number)=>boolean;
    replays?:Record<string,ReplayClip>;ondelete?:(id:string)=>void;
    replayLoading?:boolean;replayError?:string;onreplayselect?:(id?:string)=>void;
    onclearsession?:(ids:string[])=>void;
    onstart: (id: string, scramble: string, clockMs: number) => void;
    onstop: (id: string, clockMs: number) => void;
    onbusychange: (busy: boolean) => void;
    ondownloaddiagnostics?: (id: string) => void;
    storageKey?: string;
  } = $props();
  const clock = new CubeTimerClock();
  let phase = $state<'idle' | 'holding' | 'running'>('idle');
  let now = $state(0), elapsed = $state(0), scramble = $state('');
  let loadingScramble = $state(false), scrambleError = $state(''), storageError = $state('');
  let history = $state<TimerSolve[]>([]), mounted = $state(false), initialized = false;
  let attempt=$state<{ id: string; scramble: string; startedAt: string }>();
  let selectedId = $state(''), modalOpen=$state(false);
  let solveDialog=$state<HTMLDialogElement>();
  let replayPosition=$state<number>(),replaySeek=$state<{timestamp:number;revision:number}>();
  let replaySeekRevision=0;
  let savedSession = '';
  const ready = $derived(phase === 'holding' && now - clock.heldAt >= TIMER_HOLD_MS);
  const locked = $derived(phase !== 'idle');
  const canStart = $derived(cameraReady && !!scramble && !loadingScramble && active && !modalOpen);
  const last = $derived(history.at(-1));
  const selected = $derived(history.find(solve => solve.id === selectedId));
  const reconstruction = $derived(selected ? reconstructions[selected.id] ?? selected.reconstruction : undefined);
  const latestReconstruction=$derived(attempt?reconstructions[attempt.id]:last?(reconstructions[last.id]??last.reconstruction):undefined);
  const reconstructionState=$derived(timerReconstructionStatus(latestReconstruction));
  const sessionRows=$derived(timerSessionRows(history));
  const lastReconstruction=$derived(last?(reconstructions[last.id]??last.reconstruction):undefined);
  const selectedReconstruction=$derived(reconstruction?.result?reconstruction:selected?replays[selected.id]?.reconstruction??reconstruction:undefined);
  const selectedStart=$derived(selectedReconstruction?.config?.requested_start_seconds??(selected?replays[selected.id]?.start:undefined)??0);
  const lastStart=$derived(lastReconstruction?.config?.requested_start_seconds??0);
  const selectedRow=$derived(sessionRows.find(row=>row.solve.id===selectedId));
  $effect(()=>{onreplayselect?.(active&&modalOpen?selected?.id:undefined);});
  $effect(()=>{
    if(!solveDialog)return;
    if(active&&modalOpen&&selected){if(!solveDialog.open)solveDialog.showModal();}
    else if(solveDialog.open)solveDialog.close();
  });
  function openSolve(id:string){if(!locked){selectedId=id;replayPosition=undefined;replaySeek=undefined;modalOpen=true;}}
  function seekReplay(timestamp:number){replaySeek={timestamp,revision:++replaySeekRevision};}
  function deleteSolve(id:string){
    ondelete?.(id);history=history.filter(solve=>solve.id!==id);modalOpen=false;selectedId='';
  }
  const best = $derived(history.length ? Math.min(...history.map(solveTime)) : undefined);
  const currentAverage5 = $derived(timerAverage(history, 5));
  const currentAverage12 = $derived(timerAverage(history, 12));

  async function nextScramble() {
    if (locked || loadingScramble) return;
    loadingScramble = true; scrambleError = '';
    try {
      const { randomScrambleForEvent } = await import('cubing/scramble');
      scramble = (await randomScrambleForEvent('333')).toString();
    } catch (cause) {
      scrambleError = cause instanceof Error ? cause.message : String(cause);
    } finally { loadingScramble = false; }
  }
  $effect(() => {
    if (active && !initialized) { initialized = true; void nextScramble(); }
  });
  $effect(() => {
    if (!active || !cameraReady) { clock.cancelHold(); phase = clock.phase; }
    if (!cameraReady && clock.phase === 'running') {
      clock.press('CameraInterrupted', performance.now()); phase = clock.phase; finish('DNF');
    }
  });
  $effect(() => { onbusychange(locked); });
  $effect(() => {
    if (!mounted) return;
    // Persist times/scrambles and compact reconstruction summaries, not fit banks.
    const solves = history.map(solve => {
      const value = reconstructions[solve.id] ?? solve.reconstruction;
      return { ...solve, reconstruction: compactTimerReconstruction(value) };
    });
    const serialized=JSON.stringify(solves);
    if(serialized===savedSession)return;
    try { localStorage.setItem(storageKey, serialized); savedSession=serialized; storageError = ''; }
    catch { storageError = 'Browser storage is unavailable or full. Download your session to keep it.'; }
  });

  function finish(penalty: Penalty = 'none') {
    if (!attempt) return;
    const saved: TimerSolve = { ...attempt, elapsedMs: clock.elapsedMs, penalty };
    onstop(saved.id, performance.timeOrigin + clock.startedAt + clock.elapsedMs);
    history = [...history, saved]; selectedId = saved.id; elapsed = clock.elapsedMs; attempt = undefined;
    void nextScramble();
  }
  function press(code: string, time: number, editable = false, repeat = false) {
    if (clock.phase !== 'running' && !canStart) return;
    const stopped = clock.press(code, time, editable, repeat);
    phase = clock.phase; now = time;
    if (stopped) finish();
  }
  function release(time: number) {
    if (!canStart && clock.phase === 'holding') clock.cancelHold();
    if (clock.release('Space', time)) {
      attempt = { id: crypto.randomUUID(), scramble, startedAt: new Date().toISOString() };
      onstart(attempt.id, attempt.scramble, performance.timeOrigin + time);
    }
    phase = clock.phase; now = time;
  }
  function keydown(event: KeyboardEvent) {
    if (!active || modalOpen) return;
    const editable = event.target instanceof Element && !!event.target.closest('input, textarea, select, button, a, [contenteditable="true"], [role="application"], [role="slider"], .timer-replay');
    if (clock.phase === 'running' || event.code === 'Space' && !editable && canStart) event.preventDefault();
    press(event.code, performance.now(), editable, event.repeat);
  }
  function keyup(event: KeyboardEvent) {
    if (active && event.code === 'Space' && clock.phase === 'holding') { event.preventDefault(); release(performance.now()); }
  }
  function setPenalty(id: string, penalty: Penalty) {
    history = history.map(solve => solve.id === id ? { ...solve, penalty } : solve);
  }
  function clearSession(){
    if(locked)return;
    onclearsession?.(history.map(solve=>solve.id));
    history=[];selectedId='';modalOpen=false;elapsed=0;
  }
  function downloadSession() {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, event: '333', generator: 'cubing.js',
      solves: history.map(solve => ({ ...solve, reconstruction: reconstructions[solve.id] ?? solve.reconstruction })) }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'cube-timer-session.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  onMount(() => {
    try {
      history = readTimerSolves(localStorage.getItem(storageKey)).map(solve => ({ ...solve,
        reconstruction: solve.reconstruction && !['solved', 'failed'].includes(solve.reconstruction.status)
          ? { ...solve.reconstruction, status: 'failed', message: 'Page reloaded before reconstruction finished' } : solve.reconstruction }));
    } catch { storageError = 'Browser storage is unavailable. Download your session to keep it.'; }
    mounted = true;
    let animation = 0;
    const tick = () => {
      if (clock.phase !== 'idle') now = performance.now();
      if (clock.phase === 'running') elapsed = Math.max(0, now - clock.startedAt);
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  });
</script>

<svelte:window onkeydown={keydown} onkeyup={keyup} onblur={() => { clock.cancelHold(); phase = clock.phase; }} />

<section class="cube-timer" aria-label="Cube timer">
  <div class="timer-primary">
    <div class="timer-reconstruction-status" data-state={reconstructionState.kind} role="status" aria-live="polite">
      <strong>{reconstructionState.label}</strong>
      {#if latestReconstruction?.status === 'failed'}<span>{latestReconstruction.message} · Partial reconstruction{latestReconstruction.moves?.length ? ' retained' : ' · No moves recovered'}</span>{/if}
    </div>
    <div class="timer-scramble">
      <div class="timer-scramble-text">
        <div class="timer-section-heading"><strong>Scramble</strong>
          <button class="small" onclick={nextScramble} disabled={locked || loadingScramble}>{loadingScramble ? 'Generating…' : 'New scramble'}</button></div>
        <p aria-label="Timer scramble">{scramble || (loadingScramble ? 'Generating scramble…' : 'No scramble loaded')}</p>
        {#if scrambleError}<p class="timer-error" role="alert">{scrambleError}</p>{/if}
      </div>
      {#if scramble}<ScrambleNet {scramble} />{/if}
    </div>
    <div class="timer-performance">
    <div class="timer-face" class:timer-ready={ready} class:timer-holding={phase === 'holding' && !ready}>
      <div>
        <output class="timer-digits" aria-label="Solve time">{formatTimer(elapsed)}</output>
        <p>{phase === 'running' ? 'Press any key to stop' : ready ? 'Release to start' : phase === 'holding' ? 'Keep holding…' : !cameraReady ? 'Connect a camera to begin' : 'Hold space, then release to start'}</p>
      </div>
      <button class="timer-touch" aria-label="Hold to start or tap to stop timer"
        disabled={!canStart && phase !== 'running'}
        onpointerdown={event => { if (event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); press('Space', performance.now()); }}
        onpointerup={event => { event.preventDefault(); release(performance.now()); }}
        onpointercancel={() => { clock.cancelHold(); phase = clock.phase; }}
        onkeydown={event => { if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); press('Space', performance.now(), false, event.repeat); } }}
        onkeyup={event => { if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); release(performance.now()); } }}
      >{phase === 'running' ? 'Stop' : ready ? 'Release to start' : 'Hold to start'}</button>
    </div>
    <TpsGraph compact steps={lastReconstruction?.result?.steps??[]} turns={lastReconstruction?.turnEvents} moves={lastReconstruction?.moves}
      fps={lastReconstruction?.config?.fps??1000} start={lastStart} end={lastStart+(last?.elapsedMs??0)/1000} solveSeconds={last?last.elapsedMs/1000:undefined}
      complete={lastReconstruction?.status==='solved'} cursor={latestReplayPosition} label="Latest solve TPS"
      onseek={timestamp=>{if(last&&!onlatestseek?.(last.id,timestamp)){openSolve(last.id);seekReplay(timestamp);}}} />
    </div>
  </div>

  <aside class="timer-session" aria-label="Timer session">
    <div class="timer-session-heading"><h2>Session</h2><span>{history.length} solves</span></div>
    <div class="timer-statistics" aria-label="Session statistics">
      <div><span>Best</span><strong>{formatTimer(best)}</strong></div>
      <div><span>ao5</span><strong>{formatTimer(currentAverage5)}</strong></div>
      <div><span>ao12</span><strong>{formatTimer(currentAverage12)}</strong></div>
    </div>
    <div class="timer-history-scroll">
      <table aria-label="Session solves"><thead><tr><th scope="col">#</th><th scope="col">Time</th><th scope="col">ao5</th><th scope="col">ao12</th><th scope="col" aria-label="Reconstruction">✓</th></tr></thead>
        <tbody>{#each [...sessionRows].reverse() as row (row.solve.id)}
          {@const rec = reconstructions[row.solve.id] ?? row.solve.reconstruction}
          <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions (The first cell has a native keyboard-accessible button that also opens this row.) -->
          <tr class:selected={modalOpen && selectedId===row.solve.id} class:locked onclick={()=>openSolve(row.solve.id)}>
            <td><button class="timer-row-select" disabled={locked} aria-label={`View solve ${row.number}: ${formatTimer(solveTime(row.solve))}`} onclick={event=>{event.stopPropagation();openSolve(row.solve.id);}}>{row.number}</button></td>
            <td class="timer-history-time">{formatTimer(solveTime(row.solve))}{row.solve.penalty === '+2' ? '+' : ''}</td>
            <td>{formatTimer(row.ao5)}</td><td>{formatTimer(row.ao12)}</td>
            <td class="timer-reconstruction-check" title={rec?timerReconstructionStatus(rec).label:'Pending'}>
              {#if rec?.status==='solved'}<span class="success" aria-label="Reconstruction successful">✓</span>
              {:else if rec && !['failed','solved'].includes(rec.status)}<span aria-label="Reconstruction in progress">…</span>
              {:else}<span aria-label={rec?'Reconstruction failed':'No reconstruction'}>—</span>{/if}
            </td>
          </tr>
        {:else}<tr><td colspan="5" class="timer-session-empty">Your solves will appear here.</td></tr>{/each}</tbody>
      </table>
    </div>
    {#if storageError}<p class="timer-error" role="status">{storageError}</p>{/if}
    <div class="timer-session-actions">
      <button class="small" onclick={downloadSession} disabled={!history.length || locked}>Download session</button>
      <button class="small subtle" onclick={clearSession} disabled={!history.length || locked}>Clear session</button>
    </div>
  </aside>
</section>

<dialog class="timer-solve-modal" bind:this={solveDialog} onclose={()=>modalOpen=false}
  aria-labelledby="timer-solve-title" onpointerdown={event=>{
    if(event.target!==solveDialog)return;
    const rect=solveDialog!.getBoundingClientRect();
    if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)solveDialog!.close();
  }}>
  {#if selected && modalOpen}
    <div class="timer-modal-heading"><div><h2 id="timer-solve-title">Solve #{selectedRow?.number} · {formatTimer(solveTime(selected))}</h2>
      <p>ao5 {formatTimer(selectedRow?.ao5)} · ao12 {formatTimer(selectedRow?.ao12)}</p></div>
      <button class="small" onclick={()=>solveDialog?.close()}>Close</button></div>
    <div class="timer-modal-content">
      <div class="timer-modal-summary">
      <div class="timer-modal-scramble"><div><h3>Scramble</h3><p>{selected.scramble}</p></div><ScrambleNet scramble={selected.scramble} /></div>
      <div class="timer-modal-result" role="status"><strong>{reconstruction?timerReconstructionStatus(reconstruction).label:'Pending'}</strong>
        {#if reconstruction?.message}<span>{reconstruction.message}</span>{/if}
      </div>
      {#if reconstruction?.moves?.length && !replays[selected.id]}<p class="timer-modal-moves"><strong>{reconstruction.status==='solved'?'Full reconstruction':'Partial reconstruction'}</strong> {reconstruction.moves.join(' ')}</p>{/if}
      <TpsGraph compact steps={selectedReconstruction?.result?.steps??[]} turns={selectedReconstruction?.turnEvents} moves={selectedReconstruction?.moves}
        fps={selectedReconstruction?.config?.fps??replays[selected.id]?.clockFps??1000}
        start={selectedStart} end={selectedStart+selected.elapsedMs/1000} solveSeconds={selected.elapsedMs/1000} cursor={replayPosition}
        complete={selectedReconstruction?.status==='solved'} label="Selected solve TPS" onseek={seekReplay} />
      </div>
      {#if replays[selected.id] && reconstruction}
        {#key selected.id}<TimerReplay clip={replays[selected.id]}
          reconstruction={selectedReconstruction??reconstruction} disabled={!active} seekRequest={replaySeek} onpositionchange={timestamp=>replayPosition=timestamp} />{/key}
      {:else if replayLoading}<p class="timer-note">Loading projected-cube replay…</p>
      {:else}<p class="timer-note">No projected-cube replay is available for this solve.</p>{/if}
      {#if replayError}<p class="timer-error" role="status">{replayError}</p>{/if}
    </div>
    <div class="timer-modal-actions">
      <label>Penalty <select aria-label="Solve penalty" value={selected.penalty} onchange={event=>setPenalty(selected.id,event.currentTarget.value as Penalty)}>
        <option value="none">OK</option><option value="+2">+2</option><option value="DNF">DNF</option></select></label>
      {#if reconstructions[selected.id]?.diagnosticRunId && ondownloaddiagnostics}<button class="small" onclick={()=>ondownloaddiagnostics?.(selected.id)}>Download diagnostics</button>{/if}
      <button class="small subtle" onclick={()=>deleteSolve(selected.id)}>Delete solve</button>
    </div>
  {/if}
</dialog>
