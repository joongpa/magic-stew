import assert from "node:assert/strict";
import test from "node:test";
import oracle from "../../../pose-fitter/tests/fixtures/cube-colors.json";
import {buildColorTimeline,colorsAtFrame,centerColorsForScramble} from "./cube-colors";
test("reconstructed sticker colors match Python cube states through face, wide, slice and rotation moves",()=>{
  for(const entry of oracle.cases){
    const timeline=buildColorTimeline(entry.scramble,entry.steps);
    assert.deepEqual(timeline.initial,entry.initial);
    for(const step of entry.steps)assert.deepEqual(colorsAtFrame(timeline,step.frame_index,true),step.expected);
    // Seeking backwards selects an immutable snapshot, without applying moves twice.
    assert.deepEqual(colorsAtFrame(timeline,0,true),entry.initial);
    assert.deepEqual(colorsAtFrame(timeline,entry.steps[0].frame_index,true),entry.steps[0].expected);
    assert.deepEqual(colorsAtFrame(timeline,10000,true),entry.steps.at(-1)!.expected);
    assert.equal(colorsAtFrame(timeline,10000,false),undefined);
  }
});
test("pose orientation uses the scramble's center frame, including Tommy's final wide move",()=>{
 const keyed=(scramble:string)=>Object.fromEntries(centerColorsForScramble(scramble).map(c=>[`${c.axis}/${c.side}`,c.color]));
 assert.deepEqual(keyed("R U2 F L'"),keyed(""));
 assert.deepEqual(keyed("R2 L' B2 D2 L' D2 F' L F2 R2 F2 D' R2 U L2 F2 D L2 R F2 Rw'"),{
  "2/1":"blue","2/-1":"green","1/-1":"white","1/1":"yellow","0/1":"red","0/-1":"orange"});
});
