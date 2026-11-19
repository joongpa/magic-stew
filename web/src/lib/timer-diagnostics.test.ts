import assert from 'node:assert/strict';
import test from 'node:test';
import { DetectionRecording } from './detection-recording';
import { TimerDiagnostics } from './timer-diagnostics';
import { TimerReconstruction } from './timer-reconstruction';
import type { FrameResult } from './detector';
import type { RecordingContext } from './detection-recording';
import type { RunMetadata } from './run-provenance';
import type { MoveSearchResult } from './move-search-types';
import type { TimerReconstructionResult } from './cube-timer';
const metadata: RunMetadata = { policyVersion: 1, appOrigin: 'http://localhost', poseWorkers: 3, admissionCapacity: 12,
  detector: { poseSize: 256, backend: 'webgpu', preprocessing: 'gpu', settings: {
    thresholds: [.25, .4, .25, .9, .9, .9, .9, .9, .9], presenceThreshold: .05, bboxThreshold: .25, roiPadding: .4 } },
  timer: { solveId: 'first', startClockMs: 100000, cameraClockOriginMs: 90000 } };
const context: RecordingContext = { sourceId: 4, sourceName: 'Live camera', sourceKind: 'camera', sourceFps: 30,
  segment: 1, timestampSeconds: 0, backend: 'webgpu', settings: metadata.detector.settings };
const fitted = { detections: [], fittedPose: { bank: [] } } as unknown as FrameResult;

test('a fitting failure is durably captured before search, with exact cause and only solve-window detections', () => {
  const recording = new DetectionRecording(), saved: ReturnType<DetectionRecording['exportRun']>[] = [];
  const diagnostics = new TimerDiagnostics(recording, async id => { saved.push(structuredClone(recording.exportRun(id))); });
  const updates = new Map<string, TimerReconstructionResult>();
  const timer = new TimerReconstruction(() => { throw Error('Search must not start for this failed fit'); },
    (id, update) => updates.set(id, diagnostics.update(id, update)));
  const config = timer.start('first', 'R U', 10, 30);
  diagnostics.start('first', 4, config, metadata);
  const admit = (timestamp: number, frame: FrameResult) => {
    const ids = timer.solveIds(timestamp);
    if (ids.length) {
      const sequence = recording.append(frame, { ...context, timestampSeconds: timestamp });
      for (const id of ids) diagnostics.frame(id, sequence);
    }
    timer.accept(timestamp, frame);
  };
  const failure = { detections: [], fittingError: 'CPU pose initialization: CPU pose files are incomplete' } as unknown as FrameResult;
  admit(9.9, fitted); admit(10.033, failure);
  timer.stop('first', 11);
  admit(10.5, { ...failure, fittingError: 'Pose fitting cancelled' });
  admit(11.299, fitted); admit(11.3, fitted); admit(12, fitted);
  assert.equal(updates.get('first')?.message, 'Pose fitting failed at 10.033s: CPU pose initialization: CPU pose files are incomplete');
  assert.ok(updates.get('first')?.diagnosticRunId);
  const capture = diagnostics.export('first');
  assert.deepEqual(capture.frames.map(row => row.timestampSeconds), [10.033, 10.5, 11.299]);
  assert.equal(recording.length, 3, 'idle frames do not enter recording');
  assert.equal(capture.frames[0].result.fittingError, failure.fittingError);
  assert.equal(capture.runs[0].config.source_end_seconds, 11.3);
  assert.equal(capture.runs[0].config.source_origin_seconds, 10.033);
  assert.equal(capture.runs[0].metadata.timer?.startClockMs, 100000);
  assert.match(saved.at(-1)!.runs[0].error!, /initialization: CPU pose files are incomplete/);
});

test('queued searches keep separate diagnostic runs, complete search journals, and per-solve ranges', () => {
  const recording = new DetectionRecording(), saved: string[] = [];
  const diagnostics = new TimerDiagnostics(recording, async id => { saved.push(id); });
  const workers: { result: (value: MoveSearchResult, ended: boolean) => void }[] = [];
  const timer = new TimerReconstruction((config, onResult, _onError, id) => {
    const worker = { result: (value: MoveSearchResult, ended: boolean) => {
      diagnostics.search(id, config, value, [{ type: 'end' }], ended); onResult(value, ended);
    } }; workers.push(worker);
    assert.equal(config.diagnostics, true, 'timer diagnostics are enabled by default');
    return { submit() {}, end: async () => {}, dispose() {} };
  }, (id, update) => { diagnostics.update(id, update); });
  for (const [id, start] of [['first', 10], ['second', 20]] as const) {
    const config = timer.start(id, 'R U', start, 30); diagnostics.start(id, 4, config, metadata);
    const sequence = recording.append(fitted, { ...context, timestampSeconds: start }); diagnostics.frame(id, sequence);
    timer.accept(start, fitted); timer.stop(id, start + 1); timer.accept(start + 1.3, fitted);
  }
  const solved = { status: 'solved', steps: [] } as unknown as MoveSearchResult;
  workers[0].result(solved, true); workers[1].result(solved, true);
  for (const [id, timestamp] of [['first', 10], ['second', 20]] as const) {
    const capture = diagnostics.export(id);
    assert.equal(capture.frames.length, 1); assert.equal(capture.frames[0].timestampSeconds, timestamp);
    assert.deepEqual(capture.runs[0].actions, [{ type: 'end' }]);
    assert.equal(capture.runs[0].config.requested_start_seconds, timestamp);
    assert.equal(capture.runs[0].config.source_end_seconds, timestamp + 1.3);
    assert.equal(capture.runs[0].result?.status, 'solved');
    assert.ok(saved.includes(capture.runs[0].id));
  }
});

test('longer solves checkpoint their linked detections without recording idle frames', () => {
  const recording = new DetectionRecording(), saved: ReturnType<DetectionRecording['exportRun']>[] = [];
  let now = 0;
  const diagnostics = new TimerDiagnostics(recording, async id => { saved.push(structuredClone(recording.exportRun(id))); }, () => now);
  const timer = new TimerReconstruction(() => { throw Error('Not searching yet'); }, () => {});
  diagnostics.start('first', 4, timer.start('first', 'R', 10, 30), metadata);
  now = 1001;
  diagnostics.frame('first', recording.append(fitted, { ...context, timestampSeconds: 10.1 }));
  assert.equal(saved.length, 2); assert.equal(saved[1].frames.length, 1);
  assert.throws(() => diagnostics.frame('first', recording.append(fitted, { ...context, sourceId: 5 })), /different source/);
});
test('input-end acknowledgements on ongoing search reports do not rewrite the full capture',()=>{
 const recording=new DetectionRecording(),saved:string[]=[];
 const diagnostics=new TimerDiagnostics(recording,async id=>{saved.push(id);});
 const timer=new TimerReconstruction(()=>{throw Error('Not used');},()=>{});
 const config=timer.start('first','R',0,30);diagnostics.start('first',4,config,metadata);
 const before=saved.length;
 for(let i=0;i<500;i++){
  const result={status:'running',steps:[]} as unknown as MoveSearchResult;
  diagnostics.search('first',config,result,[],true);
  diagnostics.update('first',{status:'searching',result});
 }
 assert.equal(saved.length,before);
 diagnostics.update('first',{status:'failed',message:'Search state limit'});
 assert.equal(saved.length,before+1);
});
