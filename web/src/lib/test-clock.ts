import type {TestContext} from 'node:test';

/** Deterministic host timers for the project's Node 18 test runtime. */
export function fakeTimers(t:TestContext){
  let now=0,id=0;
  const tasks=new Map<number,{at:number;run:()=>void}>();
  t.mock.method(globalThis,'setTimeout',(run:()=>void,delay=0)=>{
    const handle=++id;tasks.set(handle,{at:now+delay,run});
    return handle as unknown as ReturnType<typeof setTimeout>;
  });
  t.mock.method(globalThis,'clearTimeout',(handle:unknown)=>{tasks.delete(handle as number);});
  return {tick(ms:number){
    const end=now+ms;
    while(true){
      const next=[...tasks].filter(([,task])=>task.at<=end).sort((a,b)=>a[1].at-b[1].at||a[0]-b[0])[0];
      if(!next)break;
      tasks.delete(next[0]);now=next[1].at;next[1].run();
    }
    now=end;
  }};
}
