import assert from "node:assert/strict";
import test from "node:test";
import {reviewIndex,steppedTimestamp,frameArrow} from "./frame-review";
test("arrow stepping visits each cached result at 60fps, including trim boundaries and repeated requests",()=>{
 const rows=Array.from({length:4},(_,i)=>({timestamp:(348+i*2)/60}));
 let requested=rows[0].timestamp;
 for(let i=1;i<rows.length;i++){requested=steppedTimestamp(rows,requested,1)!;assert.equal(requested,rows[i].timestamp);}
 assert.equal(steppedTimestamp(rows,requested,1),requested);
 for(let i=2;i>=0;i--){requested=steppedTimestamp(rows,requested,-1)!;assert.equal(requested,rows[i].timestamp);}
 assert.equal(steppedTimestamp(rows,requested,-1),requested);
 assert.equal(reviewIndex(rows,5.834),1);assert.equal(reviewIndex([],0),-1);
 assert.equal(steppedTimestamp([],0,1),undefined);
});
test("arrow shortcuts respect text editing and modifiers but override the video range's fractional steps",()=>{
 const event={key:"ArrowRight",altKey:false,ctrlKey:false,metaKey:false,shiftKey:false};
 const target=(editing:boolean,label:string|null=null)=>({closest:()=>editing?{}:null,getAttribute:()=>label}) as unknown as Element;
 assert.equal(frameArrow(event,null),1);assert.equal(frameArrow({...event,key:"ArrowLeft"},null),-1);
 assert.equal(frameArrow(event,target(true)),0);assert.equal(frameArrow(event,target(true,"Video position")),1);
 assert.equal(frameArrow({...event,ctrlKey:true},null),0);
 assert.equal(frameArrow({...event,key:"ArrowUp"},null),0);
});
