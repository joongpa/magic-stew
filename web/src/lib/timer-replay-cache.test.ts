import assert from 'node:assert/strict';
import test from 'node:test';
import {TimerReplayCache,type ReplayStorage} from './timer-replay-cache';
import type {TimerReplay} from './timer-replay';
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
const clip=(id:string,complete=true):TimerReplay=>({id,start:0,end:1,clockFps:1000,complete,
  frames:[{timestamp:.1,result:{},image:new Blob(['image'])}],missingImages:0});
function setup(){
 const disk=new Map<string,TimerReplay>();let displayed:TimerReplay|undefined;
 const storage:ReplayStorage={async save(value){disk.set(value.id,structuredClone(value));},async load(id){return disk.get(id);},async remove(id){disk.delete(id);}};
 const cache=new TimerReplayCache(storage,value=>displayed=value);
 return {disk,storage,cache,get displayed(){return displayed;},getDisplayed:()=>displayed};
}
test('completed solves leave no retained replay cache and only the selected replay is loaded',async()=>{
 const state=setup();
 for(let i=0;i<50;i++){const id=String(i);state.cache.begin(id);state.cache.publish(clip(id));state.cache.finish(id);await tick();}
 assert.equal(state.cache.pendingCount,0);assert.equal(state.displayed,undefined);assert.equal(state.disk.size,50);
 await state.cache.select('1');assert.equal(state.getDisplayed()?.id,'1');assert.equal(state.cache.pendingCount,0);
 await state.cache.select('49');assert.equal(state.getDisplayed()?.id,'49');assert.equal(state.cache.pendingCount,0);
 await state.cache.select(undefined);assert.equal(state.displayed,undefined);
});
test('early solutions wait for the complete replay tail and persist the reconstruction for reload',async()=>{
 const state=setup();state.cache.begin('a');await state.cache.select('a');state.cache.publish(clip('a',false));
 state.cache.finish('a',{status:'failed',message:'Reconstruction timed out after 10 seconds'});
 await tick();assert.equal(state.disk.size,0);assert.equal(state.cache.pendingCount,1);
 state.cache.publish(clip('a'));await tick();assert.equal(state.cache.pendingCount,0);
 assert.equal(state.disk.get('a')?.reconstruction?.status,'failed');
 const reloaded=new TimerReplayCache(state.storage,value=>{if(value)assert.equal(value.reconstruction?.status,'failed');});
 await reloaded.select('a');assert.equal(reloaded.pendingCount,0);
});
test('stale disk reads cannot replace a newer selection, and clearing a session removes saved replays',async()=>{
 const state=setup();let release!:(value:TimerReplay)=>void;
 state.storage.load=id=>id==='old'?new Promise(resolve=>{release=resolve;}):Promise.resolve(clip(id));
 const first=state.cache.select('old');await state.cache.select('new');release(clip('old'));await first;
 assert.equal(state.displayed?.id,'new');
 state.cache.begin('a');state.cache.publish(clip('a'));state.cache.finish('a');await tick();
 await state.cache.select('a');state.cache.remove('a');state.cache.clear();await tick();
 assert.equal(state.displayed,undefined);assert.equal(state.disk.size,0);assert.equal(state.cache.pendingCount,0);
});
test('deleting while the archive writes cannot resurrect a saved replay',async()=>{
 const state=setup();let release!:()=>void;
 state.storage.save=value=>new Promise<void>(resolve=>{release=()=>{state.disk.set(value.id,value);resolve();};});
 state.cache.begin('a');state.cache.publish(clip('a'));state.cache.finish('a');state.cache.remove('a');
 release();await tick();assert.equal(state.disk.size,0);assert.equal(state.cache.pendingCount,0);
});
test('failed replay saves do not accumulate completed clips in memory',async()=>{
 const state=setup();state.storage.save=async()=>{throw Error('Storage full');};
 state.cache.begin('a');await state.cache.select('a');state.cache.publish(clip('a'));state.cache.finish('a');
 await tick();assert.equal(state.cache.pendingCount,0);assert.equal(state.displayed?.id,'a');
 await state.cache.select(undefined);assert.equal(state.displayed,undefined);
});

test('only the latest solve keeps camera images in memory; historical and reloaded replays preserve geometry only',async()=>{
 const state=setup();
 state.cache.begin('a');state.cache.publish(clip('a'));state.cache.finish('a');await tick();
 assert.equal(state.disk.get('a')?.frames[0].image,undefined,'camera images never enter storage');
 await state.cache.select('a');assert.ok(state.getDisplayed()?.frames[0].image,'latest footage survives archiving in this tab');
 state.cache.begin('b');assert.equal(state.getDisplayed()?.frames[0].image,undefined,'starting a solve releases earlier footage');
 state.cache.publish(clip('b'));state.cache.finish('b');await tick();
 await state.cache.select('b');assert.ok(state.getDisplayed()?.frames[0].image);
 await state.cache.select('a');assert.equal(state.getDisplayed()?.frames[0].image,undefined);
 assert.equal(state.getDisplayed()?.frames.length,1,'historical replay retains pose timing');
 const reloaded=new TimerReplayCache(state.storage,value=>{if(value)assert.equal(value.frames[0].image,undefined);});
 await reloaded.select('b');
});
test('late publications from earlier solves cannot restore their camera images',async()=>{
 const state=setup();state.cache.begin('a');state.cache.publish(clip('a',false));
 state.cache.begin('b');state.cache.publish(clip('a'));state.cache.finish('a');await tick();
 await state.cache.select('a');assert.equal(state.getDisplayed()?.frames[0].image,undefined);
 assert.equal(state.disk.get('a')?.frames[0].image,undefined);
});
