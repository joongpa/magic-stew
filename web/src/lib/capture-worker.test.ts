import assert from 'node:assert/strict';
import test from 'node:test';

test('idle capture keeps the source clock moving and resumes bitmap delivery without restarting the camera',async()=>{
  const globals=globalThis as unknown as Record<string,unknown>;
  const previousSelf=Object.getOwnPropertyDescriptor(globalThis,'self');
  const previousCanvas=Object.getOwnPropertyDescriptor(globalThis,'OffscreenCanvas');
  const messages:any[]=[];let copies=0,closed=0;
  const scope={onmessage:undefined as undefined|((event:{data:any})=>Promise<void>),postMessage:(message:unknown)=>messages.push(message)};
  class Canvas {
    constructor(public width:number,public height:number){}
    getContext(){return {drawImage(){copies++;}};}
    transferToImageBitmap(){return {close(){}};}
  }
  Object.defineProperty(globalThis,'self',{value:scope,configurable:true});
  Object.defineProperty(globalThis,'OffscreenCanvas',{value:Canvas,configurable:true});
  let controller!:ReadableStreamDefaultController<any>;
  const readable=new ReadableStream({start(value){controller=value;}});
  const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
  let running:Promise<void>|undefined;
  try{
    await import('./capture.worker');
    running=scope.onmessage!({data:{type:'start',readable,sourceFps:30}});
    const frame=(index:number)=>({timestamp:Math.round(index*1e6/30),displayWidth:640,displayHeight:480,close(){closed++;}});
    controller.enqueue(frame(0));await tick();
    assert.equal(messages.filter(message=>message.type==='frame').length,1);
    await scope.onmessage!({data:{type:'ack'}});
    await scope.onmessage!({data:{type:'idle'}});
    for(let i=1;i<=30;i++)controller.enqueue(frame(i));await tick();
    assert.equal(copies,1,'idle frames are drained without bitmap copies or detector admission');
    assert.ok(messages.some(message=>message.type==='stats'&&message.metrics.captureFps>29));
    await scope.onmessage!({data:{type:'resume'}});
    for(let i=31;i<=40;i++)controller.enqueue(frame(i));await tick();
    const frames=messages.filter(message=>message.type==='frame');
    assert.equal(frames.length,11);assert.equal(copies,11);
    assert.ok(Math.abs(frames[1].timestamp-31/30)<1e-6,'resumed frames retain elapsed idle source time');
    assert.ok(frames.every(message=>message.clockOriginMs===frames[0].clockOriginMs));
    // Reproduce the driver burst from the failed solve, then duplicates and a
    // regressed source timestamp. The native frames must still be released.
    for(const timestamp of [640006740,640007205,640011442,640006000,640011442,640057699])
      controller.enqueue({...frame(0),timestamp});
    await tick();
    const afterBurst=messages.filter(message=>message.type==='frame');
    assert.deepEqual(afterBurst.slice(-3).map(message=>message.timestamp),[640.006740,640.011442,640.057699]);
    assert.equal(afterBurst.length,14);assert.equal(copies,14,'collisions are skipped before allocating/copying bitmaps');
    assert.equal(afterBurst.at(-1).metrics.skippedFrames,3);
    assert.equal(closed,47,'native frames are released during idle, active capture and timestamp rejection');
    assert.equal(messages.some(message=>message.type==='stopped'),false);
  }finally{
    controller.close();await running;
    if(previousSelf)Object.defineProperty(globalThis,'self',previousSelf);else delete globals.self;
    if(previousCanvas)Object.defineProperty(globalThis,'OffscreenCanvas',previousCanvas);else delete globals.OffscreenCanvas;
  }
});
