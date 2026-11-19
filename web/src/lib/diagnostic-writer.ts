/** One snapshot in flight. Repeated requests retain only run IDs until dispatch,
 * so queued writes serialize the latest state instead of copying stale captures. */
export class DiagnosticWriter {
  private queue = new Map<string, {resolve():void;reject(error:unknown):void}[]>();
  private busy=false;
  constructor(private snapshot:(id:string)=>unknown,private send:(value:unknown)=>Promise<void>) {}
  write(id:string):Promise<void> {
    return new Promise((resolve,reject)=>{
      const waiters=this.queue.get(id)??[];waiters.push({resolve,reject});this.queue.set(id,waiters);
      void this.pump();
    });
  }
  private async pump() {
    if(this.busy)return;this.busy=true;
    try {
      while(this.queue.size){
        const [id,waiters]=this.queue.entries().next().value!;this.queue.delete(id);
        try{await this.send(this.snapshot(id));for(const waiter of waiters)waiter.resolve();}
        catch(error){for(const waiter of waiters)waiter.reject(error);}
      }
    }finally{this.busy=false;}
  }
}
