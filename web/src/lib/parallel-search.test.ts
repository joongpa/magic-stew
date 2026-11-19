import assert from 'node:assert/strict';
import test from 'node:test';
import {ParallelSearch,packMeetings} from '../../../pose-fitter/runtime/parallel-search.mjs';
const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
const state=Array.from({length:54},(_,i)=>i);
function setup(limit=1000,rejectJoin=false,backwardInitFails=false){
 const calls:any[][]=[[],[]],pending:any[]=[[],[]],disposed=[false,false],reports:any[]=[],errors:any[]=[];
 const result=(count=0,status='running')=>({status,stats:{evaluated_states:count,furthest_frames:0,branch_stage:0},steps:[],pending:1,solved_frame_index:null,pipelineVersion:'test.wasm',solved_anchor_frame:1});
 const driver=new ParallelSearch({spawn:direction=>({
  request:async request=>{calls[direction].push(request);
   if(direction===1&&request.type==='init'&&backwardInitFails)throw Error('backward startup failed');
   if(request.type==='advance_side')return new Promise((resolve,reject)=>pending[direction].push({request,resolve,reject}));
   if(request.type==='path')return {steps:[{frame_index:1,state_moves:['R']}]};
   if(request.type==='complete_join'){if(rejectJoin)throw Error('illegal joined edge');return {...result(0,'solved'),stats:request.stats};}
   return result();},dispose:()=>{disposed[direction]=true;}}),
  report:message=>reports.push(message),error:error=>errors.push(error),now:()=>0});
 driver.request({type:'init',config:{scramble:'R',bidirectional:true,state_limit:limit}});
 return {driver,calls,pending,disposed,reports,errors,result};
}
test('parallel frontiers start only after fitting ends and have concurrent work in flight',async()=>{
 const h=setup();h.driver.request({type:'frame',frame:{sequence:0}});
 assert.deepEqual(h.calls,[[],[]]);h.driver.request({type:'end'});await tick();
 assert.equal(h.pending[0].length,1);assert.equal(h.pending[1].length,1);
 assert.equal(h.pending[0][0].request.direction,0);assert.equal(h.pending[1][0].request.direction,1);
 assert.equal(h.reports.at(-1).result.status,'running');
 h.driver.dispose();assert.deepEqual(h.disposed,[true,true]);
});
test('each worker has its own full budget and backward exhaustion cannot stop forward',async()=>{
 const h=setup(257);h.driver.request({type:'end'});await tick();
 assert.equal(h.pending[0][0].request.budget,128);assert.equal(h.pending[1][0].request.budget,128);
 h.pending[0].shift().resolve(h.result(128));await tick();
 assert.equal(h.pending[0][0].request.budget,128);
 h.pending[1].shift().resolve(h.result(128));await tick();
 h.pending[1].shift().resolve(h.result(256));await tick();
 assert.equal(h.pending[1][0].request.budget,1);
 h.pending[1].shift().resolve(h.result(257));await tick();
 assert.equal(h.pending[1].length,0);
 assert.equal(h.disposed[0],false);
 h.pending[0].shift().resolve(h.result(256));await tick();
 assert.equal(h.pending[0][0].request.budget,1);
 h.pending[0].shift().resolve(h.result(257));await tick();
 const r=h.reports.at(-1).result;assert.equal(r.status,'state_limit');assert.equal(r.stats.evaluated_states,514);
 assert.equal(r.stats.forward_expanded,257);assert.equal(r.stats.backward_expanded,257);
 assert.deepEqual(h.disposed,[true,true]);
});
test('packed meetings require identical depth, geometry cue, and all encoded stickers',()=>{
 const h=setup(),driver:any=h.driver;
 assert.equal(driver.meet(0,packMeetings([[0,10,0,state]])),undefined);
 assert.equal(driver.meet(1,packMeetings([[2,11,0,state],[3,10,1,state],[4,10,0,[...state.slice(0,53),52]]])),undefined);
 assert.deepEqual(driver.meet(1,packMeetings([[5,10,0,state]])),[0,5]);h.driver.dispose();
});
test('a meeting is validated by Rust before a solved result is published',async()=>{
 const h=setup();h.driver.request({type:'frame',frame:{sequence:0}});h.driver.request({type:'end'});await tick();
 h.pending[0].shift().resolve({...h.result(1,'exhausted'),meeting_buffer:packMeetings([[7,1,0,state]])});await tick();
 assert.ok(h.reports.every(message=>message.result.status!=='solved'));
 h.pending[1].shift().resolve({...h.result(1),meeting_buffer:packMeetings([[8,1,0,state]])});await tick();
 assert.equal(h.reports.at(-1).result.status,'solved');assert.equal(h.reports.at(-1).result.stats.evaluated_states,2);
 const join=h.calls[0].find(request=>request.type==='complete_join');assert.equal(join.forward_node,7);
 assert.deepEqual(join.backward_steps,[{frame_index:1,state_moves:['R']}]);assert.equal(h.calls[1].at(-1).type,'path');
 assert.ok(h.reports.flatMap(message=>message.actions).some(action=>action.type==='complete_join'));
 assert.deepEqual(h.errors,[]);assert.deepEqual(h.disposed,[true,true]);
});
test('an invalid joined path is an error and never a solved reconstruction',async()=>{
 const h=setup(1000,true);h.driver.request({type:'end'});await tick();
 h.pending[0].shift().resolve({...h.result(1,'exhausted'),meeting_buffer:packMeetings([[7,1,0,state]])});await tick();
 h.pending[1].shift().resolve({...h.result(1),meeting_buffer:packMeetings([[8,1,0,state]])});await tick();
 assert.equal(h.errors.length,1);assert.match(String(h.errors[0]),/illegal joined edge/);
 assert.ok(h.reports.every(message=>message.result.status!=='solved'));assert.deepEqual(h.disposed,[true,true]);
});
test('cancelling a source before EOF never spawns search workers',()=>{
 const h=setup();h.driver.request({type:'frame',frame:{sequence:0}});h.driver.dispose();h.driver.request({type:'end'});
 assert.deepEqual(h.calls,[[],[]]);
});
test('a failed backward worker leaves the independent forward worker running',async()=>{
 const h=setup();h.driver.request({type:'end'});await tick();
 h.pending[1].shift().reject(Error('backward memory limit'));await tick();
 assert.equal(h.disposed[0],false);assert.equal(h.disposed[1],true);
 assert.equal(h.reports.at(-1).result.search_direction,'forward');
 h.pending[0].shift().resolve(h.result(5,'solved'));await tick();
 assert.equal(h.reports.at(-1).result.status,'solved');assert.equal(h.reports.at(-1).result.stats.forward_expanded,5);
 assert.deepEqual(h.errors,[]);
});
test('backward initialization failure still starts the independent forward search',async()=>{
 const h=setup(1000,false,true);h.driver.request({type:'end'});await tick();
 assert.equal(h.disposed[0],false);assert.equal(h.disposed[1],true);
 assert.equal(h.pending[0].length,1);assert.equal(h.pending[1].length,0);
 h.pending[0].shift().resolve(h.result(5,'solved'));await tick();
 assert.equal(h.reports.at(-1).result.status,'solved');assert.equal(h.reports.at(-1).result.search_direction,'forward');
 assert.deepEqual(h.errors,[]);
});
