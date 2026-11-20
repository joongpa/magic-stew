import assert from 'node:assert/strict';
import test from 'node:test';
import {turnEvents,gaussianTpsAt,tpsCurve,TPS_SIGMA_SECONDS,committedTurnSpan,tpsSummary} from './tps';
import {compactTimerReconstruction} from './cube-timer';
import type {TimerReconstructionResult} from './cube-timer';
const step=(frame_index:number,...state_moves:string[])=>({frame_index,state_moves});
const near=(actual:number,expected:number,tolerance=1e-10)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);

test('HTM events merge split halves at their midpoint, while preserving inverses, holds, batches and rotations',()=>{
 const steps=[step(30,'R'),step(31),step(32,'R'),step(33,'R'),step(34,"R'"),step(35,'U2','U2'),
   step(36,'F','F'),step(37,'L','x','L'),step(38,'r',"Rw"),step(39,'M2','y','S')];
 const before=JSON.stringify(steps),events=turnEvents(steps,30);
 assert.deepEqual(events.map(event=>event.move),['R2','R',"R'",'U2','U2','F2','L','L','Rw2','M2','S']);
 near(events[0].timestamp,31/30);near(events[5].timestamp,36/30);
 assert.equal(JSON.stringify(steps),before);
});

test('video and irregular camera clocks retain absolute source times without nominal-FPS rounding',()=>{
 const camera=turnEvents([step(10500,'R'),step(10537,'R'),step(10710,'U')],1000);
 near(camera[0].timestamp,10.5185);near(camera[1].timestamp,10.71);
 const fps=60000/1001,video=turnEvents([step(374,'F2')],fps);
 near(video[0].timestamp,374/fps);
 assert.deepEqual(turnEvents([step(1,'R')],0),[]);
});

test('Gaussian uses a 0.2 second standard deviation and normalized one-turn impulses',()=>{
 assert.equal(TPS_SIGMA_SECONDS,.2);
 const event=turnEvents([step(30,'R2')],30),peak=1/(.2*Math.sqrt(2*Math.PI));
 near(gaussianTpsAt(event,1),peak);
 near(gaussianTpsAt(event,.8),peak*Math.exp(-.5));
 near(gaussianTpsAt(event,1.2),peak*Math.exp(-.5));
 assert.equal(gaussianTpsAt([],1),0);
});

test('the full Gaussian curve integrates to HTM count without normalizing away boundary tails',()=>{
 const events=turnEvents([step(0,'R2'),step(10,'U'),step(10,'F')],10);
 let integral=0;const dt=.002;
 for(let t=-2;t<3;t+=dt)integral+=gaussianTpsAt(events,t)*dt;
 near(integral,3,1e-6);
 const inside=tpsCurve(events,0,1);
 assert.equal(inside[0].timestamp,0);assert.equal(inside.at(-1)?.timestamp,1);
 assert.ok(inside.every(point=>Number.isFinite(point.tps)&&point.tps>=0));
 assert.deepEqual(tpsCurve(events,1,1),[]);
});

test('per-solve persisted summaries retain only the small timed turn list needed to restore TPS',()=>{
 const value={status:'solved',config:{fps:1000},result:{steps:[step(12000,'R'),step(12040,'R'),step(12200,"U'")]}} as TimerReconstructionResult;
 const compact=compactTimerReconstruction(value)!;
 assert.deepEqual(compact.turnEvents,[{timestamp:12.02,move:'R2'},{timestamp:12.2,move:"U'"}]);
 assert.equal(compact.result,undefined);
 assert.deepEqual(compactTimerReconstruction(compact)?.turnEvents,compact.turnEvents);
 assert.equal(compactTimerReconstruction({...value,status:'searching'})?.turnEvents,undefined);
});

test('upload solve time excludes clip lead-in, holds and tail, while counts use grouped HTM events',()=>{
 const steps=[step(0),step(300,'R'),step(306,'R'),step(420,'U2'),step(600,'x'),step(900)];
 const events=turnEvents(steps,60),seconds=committedTurnSpan(steps,60);
 assert.equal(seconds,2);assert.deepEqual(tpsSummary(events,seconds),{moves:2,seconds:2,averageTps:1});
 assert.deepEqual(tpsSummary(events,3.5),{moves:2,seconds:3.5,averageTps:2/3.5},'timer uses measured elapsed time rather than clip or move span');
});

test('unknown single-commit spans and zero-time solves never invent a clip-length average or produce infinity',()=>{
 const steps=[step(0),step(30,'R2'),step(600)],events=turnEvents(steps,30);
 assert.equal(committedTurnSpan(steps,30),undefined);
 assert.deepEqual(tpsSummary(events,undefined),{moves:1,seconds:undefined,averageTps:undefined});
 assert.equal(tpsSummary(events,0).averageTps,undefined);
 assert.equal(tpsSummary(events,Infinity).seconds,undefined);
});

test('legacy timer summaries can retain known move totals without inventing a timed curve',()=>{
 const moves=["R","R","U2","y","F'"];
 const count=turnEvents([step(0,...moves)],1).length;
 assert.deepEqual(tpsSummary([],2,count),{moves:3,seconds:2,averageTps:1.5});
});
