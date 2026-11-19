import type {Plugin,Connect} from 'vite';
import {mkdir,writeFile,rename} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
/** Local dev/preview only. Save actual app inputs locally so diagnostics never need
 * hand-reconstructed settings or another interactive browser test. */
export function localDiagnostics(directory=fileURLToPath(new URL('../../pose-fitter/results/browser-runs/',import.meta.url))):Plugin {
  const revisions=new Map<string,number>();
  const writes=new Map<string,Promise<void>>();
  function install(middlewares:Connect.Server) {
    middlewares.use('/__diagnostics/run',async(req,res)=>{
      try {
        if(req.method!=='POST'){res.statusCode=405;res.end('POST required');return;}
        if(req.headers.origin&&new URL(req.headers.origin).host!==req.headers.host){res.statusCode=403;res.end('Same origin required');return;}
        const chunks:Buffer[]=[];let count=0;
        for await(const chunk of req){const bytes=Buffer.from(chunk);count+=bytes.length;
          if(count>64*1024*1024){res.statusCode=413;res.end('Diagnostic capture too large');return;}chunks.push(bytes);}
        const text=Buffer.concat(chunks).toString('utf8'),data=JSON.parse(text);
        if(typeof data.id!=='string'||!/^[-A-Za-z0-9_:.]{10,150}$/.test(data.id)||!Array.isArray(data.frames)||!Array.isArray(data.runs)){
          res.statusCode=400;res.end('Invalid run capture');return;}
        const id=data.id.replaceAll(':','_'),revision=Number(data.revision);
        if(!Number.isFinite(revision)){res.statusCode=400;res.end('Missing capture revision');return;}
        if(revision<(revisions.get(id)??-1)){res.statusCode=202;res.end('Newer capture already saved');return;}
        revisions.set(id,revision);
        const task=(writes.get(id)??Promise.resolve()).catch(()=>{}).then(async()=>{
          if(revision<(revisions.get(id)??-1))return;
          await mkdir(directory,{recursive:true});
          const file=join(directory,id+'.json'),temporary=file+'.tmp';
          await writeFile(temporary,text);await rename(temporary,file);
          const run=data.runs.find((r:any)=>r.id===data.id);
          await writeFile(join(directory,'latest.json'),JSON.stringify({id,path:file,revision,
            source:run?.metadata?.source,poseSize:run?.metadata?.detector?.poseSize,
            status:run?.error?'error':run?.result?.status??'collecting',error:run?.error,
            frames:data.frames.length,config:run?.config})+'\n');
        });
        writes.set(id,task);await task;
        res.setHeader('Content-Type','application/json');res.end(JSON.stringify({saved:id}));
      }catch(error){res.statusCode=500;res.end(error instanceof Error?error.message:String(error));}
    });
  }
  return {name:'local-reconstruction-diagnostics',apply:'serve',
    configureServer(server){install(server.middlewares);},configurePreviewServer(server){install(server.middlewares);}};
}
