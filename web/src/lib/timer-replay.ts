import type {FrameResult} from './detector';
import {TIMER_TAIL_SECONDS} from './cube-timer';

export interface TimerReplayFrame {timestamp:number;result:FrameResult;image?:Blob}
export interface TimerReplay {id:string;start:number;end:number;clockFps:number;frames:readonly TimerReplayFrame[];missingImages:number;complete:boolean}
interface Clip {id:string;start:number;end?:number;clockFps:number;closed:boolean;frames:Promise<TimerReplayFrame>[];
  ready:TimerReplayFrame[];published:number;timer?:ReturnType<typeof setTimeout>}

/** Only processed frames inside a timed solve are encoded. No camera images or
 * detections from idle time are retained, and encoding never blocks fitting. */
export class TimerReplayStore {
  private clips=new Map<string,Clip>();
  constructor(private publish:(clip:TimerReplay)=>void) {}
  start(id:string,start:number,clockFps:number){this.clips.set(id,{id,start,clockFps,closed:false,frames:[],ready:[],published:0});}
  stop(id:string,stop:number){const clip=this.clips.get(id);if(clip&&!clip.closed){clip.end=stop+TIMER_TAIL_SECONDS;this.emit(clip);}}
  capture(timestamp:number,result:FrameResult,encode:()=>Promise<Blob>){
    let image:Promise<Blob|undefined>|undefined;
    for(const clip of this.clips.values()){
      if(clip.closed||timestamp<clip.start)continue;
      if(clip.end!==undefined&&timestamp>=clip.end){void this.finish(clip);continue;}
      // convertToBlob snapshots pixels synchronously; the presenter's canvas
      // can then be reused while the browser encodes that snapshot off-thread.
      if(!image){try{image=encode().catch(()=>undefined);}catch{image=Promise.resolve(undefined);}}
      const index=clip.frames.length;
      clip.frames.push(image.then(image=>{
        const frame={timestamp,result,image};clip.ready[index]=frame;
        if(clip.end!==undefined&&!clip.closed&&this.clips.get(clip.id)===clip&&clip.timer===undefined)
          clip.timer=setTimeout(()=>{clip.timer=undefined;this.emit(clip);},100);
        return frame;
      }));
    }
  }
  closeInput(){for(const clip of this.clips.values())if(!clip.closed)void this.finish(clip);}
  finishThrough(timestamp:number){for(const clip of this.clips.values())if(!clip.closed&&clip.end!==undefined&&timestamp>=clip.end)void this.finish(clip);}
  remove(id:string){clearTimeout(this.clips.get(id)?.timer);this.clips.delete(id);}
  clear(){for(const clip of this.clips.values())clearTimeout(clip.timer);this.clips.clear();}
  private emit(clip:Clip){
    if(this.clips.get(clip.id)!==clip)return;
    let count=0;while(clip.ready[count])count++;
    if(!count||!clip.closed&&count===clip.published)return;
    clip.published=count;const frames=clip.ready.slice(0,count);
    this.publish({id:clip.id,start:clip.start,end:clip.end??frames.at(-1)?.timestamp??clip.start,
      clockFps:clip.clockFps,frames,missingImages:frames.filter(frame=>!frame.image).length,complete:clip.closed});
  }
  private async finish(clip:Clip){
    clip.closed=true;clearTimeout(clip.timer);await Promise.all(clip.frames);
    if(this.clips.get(clip.id)!==clip)return;
    this.emit(clip);
    this.clips.delete(clip.id);
  }
}
