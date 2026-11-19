import assert from 'node:assert/strict';
import test from 'node:test';
import { CubeTimerClock, formatTimer, timerAverage, readTimerSolves, solveTime, compactTimerReconstruction } from './cube-timer';
import type { TimerSolve } from './cube-timer';
const solve = (elapsedMs: number, penalty: TimerSolve['penalty'] = 'none'): TimerSolve =>
  ({ id: String(elapsedMs), scramble: 'R U', elapsedMs, penalty, startedAt: '2026-11-19' });

test('hold-space starts only on release after 300ms, including repeated keydowns', () => {
  const timer = new CubeTimerClock();
  timer.press('Space', 100); assert.equal(timer.release('Space', 399), false);
  timer.press('Space', 500); timer.press('Space', 700, false, true);
  assert.equal(timer.release('Space', 800), true);
  assert.equal(timer.startedAt, 800); assert.equal(timer.phase, 'running');
});
test('every newly pressed key stops a running solve, including modifiers and focused inputs', () => {
  for (const key of ['Space', 'KeyR', 'Enter', 'Escape', 'ShiftLeft', 'ArrowLeft', 'Tab']) {
    const timer = new CubeTimerClock(); timer.press('Space', 0); timer.release('Space', 300);
    assert.equal(timer.press(key, 1534, true), 'stop', key);
    assert.equal(timer.elapsedMs, 1234); assert.equal(timer.phase, 'idle');
    assert.equal(timer.release('Space', 1600), false, 'stop key release cannot restart');
  }
});
test('auto-repeat, focused fields, early release and cancelled holds cannot start/stop accidentally', () => {
  const timer = new CubeTimerClock();
  timer.press('KeyA', 0); timer.press('Space', 0, true); assert.equal(timer.phase, 'idle');
  timer.press('Space', 100); timer.cancelHold(); assert.equal(timer.release('Space', 500), false);
  timer.press('Space', 600); timer.release('Space', 900);
  timer.press('Space', 1000, false, true); assert.equal(timer.phase, 'running');
});
test('averages drop best and worst, include +2, tolerate one DNF and reject two', () => {
  const times = [1000, 2000, 3000, 4000, 5000].map(ms => solve(ms));
  assert.equal(timerAverage(times, 5), 3000);
  assert.equal(timerAverage(times.slice(0, 4), 5), undefined);
  assert.equal(timerAverage([...times.slice(0, 4), solve(5000, 'DNF')], 5), 3000);
  assert.equal(timerAverage([...times.slice(0, 3), solve(4000, 'DNF'), solve(5000, 'DNF')], 5), Infinity);
  assert.equal(solveTime(solve(1000, '+2')), 3000);
  assert.equal(timerAverage([solve(1000, '+2'), ...times.slice(1)], 5), 10000 / 3);
  assert.equal(formatTimer(61239), '1:01.23'); assert.equal(formatTimer(Infinity), 'DNF');
});
test('history parsing rejects corrupt data without losing valid solves', () => {
  assert.deepEqual(readTimerSolves('{bad'), []);
  assert.deepEqual(readTimerSolves(JSON.stringify([null, solve(1234), { ...solve(-1) }, { ...solve(2), penalty: 'wrong' }])), [solve(1234)]);
});
test('search progress does not change the persisted session until a lifecycle or terminal result changes',()=>{
 const first=compactTimerReconstruction({status:'searching',frames:100,moves:['R']});
 const later=compactTimerReconstruction({status:'searching',frames:100,moves:['R','U','R']});
 assert.equal(JSON.stringify(first),JSON.stringify(later));
 assert.deepEqual(compactTimerReconstruction({status:'solved',moves:['R','U']})?.moves,['R','U']);
});
