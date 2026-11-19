import assert from 'node:assert/strict';
import test from 'node:test';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import {localDiagnostics} from '../../scripts/local-diagnostics';

test('automatic local captures save exact run data and cannot overwrite a newer revision',async t=>{
 const directory=await mkdtemp(join(tmpdir(),'recognition-diagnostic-test-'));
 let handler:any;
 const plugin=localDiagnostics(directory);
 (plugin.configureServer as Function)({middlewares:{use(_path:string,value:any){handler=value;}}});
 const server=createServer((req,res)=>{void handler(req,res);});
 await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 t.after(async()=>{server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));await rm(directory,{recursive:true,force:true});});
 const address=server.address();assert.ok(address&&typeof address!=='string');
 const url=`http://127.0.0.1:${address.port}/__diagnostics/run`;
 const payload={id:'protocol-capture-test',revision:2,frames:[{raw:'exact-detections'}],runs:[{config:{fps:59.94}}]};
 const send=(value:any)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
 assert.equal((await send(payload)).status,200);
 assert.equal((await send({...payload,revision:1})).status,202);
 assert.deepEqual(JSON.parse(await readFile(join(directory,'protocol-capture-test.json'),'utf8')),payload);
 const latest=JSON.parse(await readFile(join(directory,'latest.json'),'utf8'));
 assert.equal(latest.revision,2);assert.equal(latest.path,join(directory,'protocol-capture-test.json'));
});
