<script lang="ts">
  import { onMount } from 'svelte';
  import { CubeTimerClock, TIMER_HOLD_MS, formatTimer, solveTime, timerAverage, readTimerSolves, compactTimerReconstruction } from './cube-timer';
  import type { Penalty, TimerSolve, TimerReconstructionResult } from './cube-timer';
  import ScrambleNet from './ScrambleNet.svelte';
  import TimerReplay from './TimerReplay.svelte';
  import type {TimerReplay as ReplayClip} from './timer-replay';

  let { active, cameraReady, reconstructions, replays={}, ondelete, onstart, onstop, onbusychange, ondownloaddiagnostics, storageKey = 'live-recognition.timer.v1' }: {
    active: boolean; cameraReady: boolean; reconstructions: Record<string, TimerReconstructionResult>;
    replays?:Record<string,ReplayClip>;ondelete?:(id:string)=>void;
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
  let attempt: { id: string; scramble: string; startedAt: string } | undefined;
  let selectedId = $state('');
  let savedSession = '';
  const ready = $derived(phase === 'holding' && now - clock.heldAt >= TIMER_HOLD_MS);
  const locked = $derived(phase !== 'idle');
  const canStart = $derived(cameraReady && !!scramble && !loadingScramble && active);
  const last = $derived(history.at(-1));
  const selected = $derived(history.find(solve => solve.id === selectedId) ?? last);
  const reconstruction = $derived(selected ? reconstructions[selected.id] ?? selected.reconstruction : undefined);
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
    if (!active) return;
    const editable = event.target instanceof Element && !!event.target.closest('input, textarea, select, button, a, [contenteditable="true"]');
    if (clock.phase === 'running' || event.code === 'Space' && !editable && canStart) event.preventDefault();
    press(event.code, performance.now(), editable, event.repeat);
  }
  function keyup(event: KeyboardEvent) {
    if (active && event.code === 'Space' && clock.phase === 'holding') { event.preventDefault(); release(performance.now()); }
  }
  function setPenalty(id: string, penalty: Penalty) {
    history = history.map(solve => solve.id === id ? { ...solve, penalty } : solve);
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
  <div class="timer-scramble">
    <div class="timer-section-heading"><strong>3×3×3 · Random-state scramble</strong>
      <button class="small" onclick={nextScramble} disabled={locked || loadingScramble}>{loadingScramble ? 'Generating…' : 'New scramble'}</button></div>
    <p aria-label="Timer scramble">{scramble || (loadingScramble ? 'Generating scramble…' : 'No scramble loaded')}</p>
    {#if scramble}<ScrambleNet {scramble} />{/if}
    {#if scrambleError}<p class="timer-error" role="alert">{scrambleError}</p>{/if}
  </div>
  <div class="timer-face" class:timer-ready={ready} class:timer-holding={phase === 'holding' && !ready}>
    <p class="timer-mode">{phase === 'running' ? 'SOLVING' : ready ? 'READY' : phase === 'holding' ? 'HOLD' : 'CUBE TIMER'}</p>
    <output class="timer-digits" aria-label="Solve time">{formatTimer(elapsed)}</output>
    <p>{phase === 'running' ? 'Press any key to stop' : ready ? 'Release to start' : phase === 'holding' ? 'Keep holding…' : !cameraReady ? 'Connect the camera below to begin' : 'Hold space, then release to start · Any key to stop'}</p>
    <button class="timer-touch" aria-label="Hold to start or tap to stop timer"
      disabled={!canStart && phase !== 'running'}
      onpointerdown={event => { if (event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); press('Space', performance.now()); }}
      onpointerup={event => { event.preventDefault(); release(performance.now()); }}
      onpointercancel={() => { clock.cancelHold(); phase = clock.phase; }}
      onkeydown={event => { if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); press('Space', performance.now(), false, event.repeat); } }}
      onkeyup={event => { if (event.code === 'Space' || event.code === 'Enter') { event.preventDefault(); release(performance.now()); } }}
    >{phase === 'running' ? 'Stop' : ready ? 'Release to start' : 'Hold to start'}</button>
    <p class="timer-note">Automatic reconstruction · Timer start to stop + 0.3s · 10s search limit after capture</p>
  </div>
  <div class="timer-statistics" aria-label="Session statistics">
    <div><span>Solves</span><strong>{history.length}</strong></div>
    <div><span>Best</span><strong>{formatTimer(best)}</strong></div>
    <div><span>ao5</span><strong>{formatTimer(currentAverage5)}</strong></div>
    <div><span>ao12</span><strong>{formatTimer(currentAverage12)}</strong></div>
  </div>
  <div class="timer-history">
    <div class="timer-section-heading"><h2>Session</h2><button class="small" onclick={downloadSession} disabled={!history.length || locked}>Download session</button></div>
    {#if storageError}<p class="timer-error" role="status">{storageError}</p>{/if}
    {#if history.length}
      <div class="timer-history-scroll"><table><thead><tr><th>Solve</th><th>Time</th><th>Penalty</th><th>Reconstruction</th><th></th></tr></thead>
        <tbody>{#each [...history].reverse() as solve, index (solve.id)}
          {@const rec = reconstructions[solve.id] ?? solve.reconstruction}
          <tr class:selected={selected?.id === solve.id}>
            <td><button class="text-button" disabled={locked} onclick={() => selectedId = solve.id}>#{history.length - index}</button></td>
            <td class="timer-history-time">{formatTimer(solveTime(solve))}{solve.penalty === '+2' ? '+' : ''}</td>
            <td><select aria-label={`Penalty for solve ${history.length - index}`} value={solve.penalty} disabled={locked} onchange={event => setPenalty(solve.id, event.currentTarget.value as Penalty)}>
              <option value="none">OK</option><option value="+2">+2</option><option value="DNF">DNF</option></select></td>
            <td>{rec?.status ?? 'Pending'}{rec?.status === 'solved' ? ` · ${rec.moves?.length ?? 0} moves` : ''}</td>
            <td><button class="small subtle" aria-label={`Delete solve ${history.length - index}`} disabled={locked} onclick={() => {ondelete?.(solve.id);history = history.filter(row => row.id !== solve.id);}}>Delete</button></td>
          </tr>
        {/each}</tbody></table></div>
      {#if selected}<details open class="timer-solve-details"><summary>Solve details · {formatTimer(solveTime(selected))}</summary>
        <p><strong>Scramble</strong> {selected.scramble}</p>
        <ScrambleNet scramble={selected.scramble} />
        <p role="status"><strong>Reconstruction</strong> {reconstruction?.message ?? reconstruction?.status ?? 'Pending'}{reconstruction?.frames ? ` · ${reconstruction.frames} frames` : ''}</p>
        {#if reconstruction?.result}<p class="timer-note" aria-label="Timer search progress">
          {reconstruction.result.frames_prepared} frames prepared · {reconstruction.result.stats.furthest_frames} frames searched · {reconstruction.result.stats.evaluated_states.toLocaleString()} states
        </p>{/if}
        {#if reconstructions[selected.id]?.diagnosticRunId && ondownloaddiagnostics}<button class="small" disabled={locked}
          onclick={() => ondownloaddiagnostics?.(selected.id)}>Download diagnostics</button>{/if}
        {#if reconstruction?.moves?.length}<p><strong>{reconstruction.status === 'solved' ? 'Solution' : 'Current candidate'}</strong> {reconstruction.moves.join(' ')}</p>{/if}
        {#if reconstruction?.config}<p class="timer-note">Source range {reconstruction.config.requested_start_seconds.toFixed(3)}–{reconstruction.config.source_end_seconds.toFixed(3)}s (includes 0.3s after stop)</p>{/if}
        {#if replays[selected.id]&&reconstruction}
          {#if !['solved','failed'].includes(reconstruction.status)}<p class="timer-note">Pose replay is available while reconstruction searches.</p>{/if}
          {#key selected.id}<TimerReplay clip={replays[selected.id]} {reconstruction} disabled={locked||!active} />{/key}
        {:else if reconstruction&&['solved','failed'].includes(reconstruction.status)}
          <p class="timer-note">Replay is kept for solves captured in this page session.</p>
        {/if}
      </details>{/if}
    {:else}<p class="muted">Your solve times, scrambles, and reconstructions will appear here.</p>{/if}
  </div>
</section>
