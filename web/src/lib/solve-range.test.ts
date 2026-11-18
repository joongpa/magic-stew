import assert from "node:assert/strict";
import test from "node:test";
import { trimRange } from "./solve-range";
test("trim handles preserve order, stay inside the source and retain a frame",()=>{
 const range={start:2,end:8};
 assert.deepEqual(trimRange(range,"start",-3,10,60),{start:0,end:8});
 assert.equal(trimRange(range,"start",9,10,60).start,8-1/60);
 assert.deepEqual(trimRange(range,"end",20,10,60),{start:2,end:10});
 assert.equal(trimRange(range,"end",1,10,60).end,2+1/60);
 assert.equal(trimRange(range,"start",NaN,10,60),range);
});
