import assert from 'node:assert/strict';
import test from 'node:test';
import {DiagnosticWriter} from './diagnostic-writer';
test('large diagnostic writes are serialized and repeated requests snapshot only the latest queued state',async()=>{
 const snapshots:{id:string;revision:number}[]=[],sent:unknown[]=[];let revision=1,release!:(value?:unknown)=>void;
 const writer=new DiagnosticWriter(id=>{const value={id,revision};snapshots.push(value);return value;},async value=>{
  sent.push(value);if(sent.length===1)await new Promise(resolve=>{release=resolve;});
 });
 const first=writer.write('run');revision=2;
 const pending=Array.from({length:100},()=>writer.write('run'));revision=101;
 assert.equal(snapshots.length,1,'queued requests do not copy full pose captures');
 release();await Promise.all([first,...pending]);
 assert.deepEqual(sent,[{id:'run',revision:1},{id:'run',revision:101}]);
});
test('a failed diagnostic write rejects its callers and does not block later runs',async()=>{
 let count=0;const writer=new DiagnosticWriter(id=>id,async()=>{if(++count===1)throw Error('Disk unavailable');});
 await assert.rejects(writer.write('first'),/Disk unavailable/);await writer.write('second');assert.equal(count,2);
});
