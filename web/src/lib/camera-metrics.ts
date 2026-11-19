/** A bounded source-time window measures delivered capture frames, independently
 * of the camera's advertised setting and delayed processing completions. */
export class CameraFrameRate {
  private times:number[]=[];
  observe(timestamp:number):number {
    if(!Number.isFinite(timestamp))return 0;
    if(this.times.length&&timestamp<=this.times.at(-1)!)this.times=[];
    this.times.push(timestamp);
    while(this.times.length>2&&timestamp-this.times[0]>2)this.times.shift();
    return this.times.length>1?(this.times.length-1)/(timestamp-this.times[0]):0;
  }
  reset(){this.times=[];}
}
export interface CameraCaptureMetrics {captureFps:number;skippedFrames:number;browserSkippedFrames:number;copyMs?:number;deliveryMs?:number}
/** Saturation limits admission, never stops the consumer or evicts queued work. */
export class CameraFrameBuffer {
  pending=0;skipped=0;
  constructor(readonly capacity=60) {}
  admit(){if(this.pending>=this.capacity){this.skipped++;return false;}this.pending++;return true;}
  acknowledge(){this.pending=Math.max(0,this.pending-1);}
}
