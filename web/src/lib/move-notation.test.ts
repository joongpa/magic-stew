import assert from "node:assert/strict";
import test from "node:test";
import turns from "../../../pose-fitter/tables/render-turns.json";
import { applyCubeMove, buildColorTimeline, stateAtFrame } from "./cube-colors";
import { axisAngle, multiply } from "./cube-motion";
import type { CubePose } from "./cube-motion";
import type { MoveStep } from "./move-search-types";
import { CAMERA_ORIENTATION, displayMoveTokens, reorientedMoveTimeline } from "./cube-orientation";
import type { ReorientedMoveStep } from "./cube-orientation";

const initial=stateAtFrame(buildColorTimeline("",[]),0,true)!;
const batch=(frame:number,moves:string[],rotation=false):ReorientedMoveStep=>({frame,moves,rotation});

test("moves committed together become individually spaced display tokens",()=>{
 const source=[batch(10,["R","U","F'"]),batch(12,["L","D"])],snapshot=JSON.stringify(source);
 assert.deepEqual(displayMoveTokens(source),[
  {frame:10,move:"R",rotation:false},{frame:10,move:"U",rotation:false},{frame:10,move:"F'",rotation:false},
  {frame:12,move:"L",rotation:false},{frame:12,move:"D",rotation:false},
 ]);
 assert.equal(JSON.stringify(source),snapshot,"display formatting must preserve committed batches and timestamps");
});

test("double turns group inside a batch and across batch boundaries",()=>{
 const tokens=displayMoveTokens([batch(10,["R","R","U"]),batch(12,["U","F","F'","Rw'","Rw'"]),batch(14,["x","x"],true)]);
 assert.deepEqual(tokens,[
  {frame:10,move:"R2",rotation:false},{frame:12,move:"U2",rotation:false},
  {frame:12,move:"F",rotation:false},{frame:12,move:"F'",rotation:false},
  {frame:12,move:"Rw2",rotation:false},{frame:14,move:"x2",rotation:true},
 ]);
 assert.deepEqual(displayMoveTokens([batch(0,["R","R","R"])]).map(token=>token.move),["R2","R"]);
});

test("grouped double turns preserve the exact sticker permutation for every supported quarter turn",()=>{
 for(const move of Object.keys(turns).filter(move=>!move.endsWith("2"))){
  const tokens=displayMoveTokens([batch(0,[move]),batch(2,[move])]);
  assert.equal(tokens.length,1,move);assert.equal(tokens[0].frame,2);
  assert.equal(tokens[0].move,move.replace(/'$/,"")+"2");
  const original=applyCubeMove(applyCubeMove(initial,move,[]),move,[]);
  const grouped=applyCubeMove(initial,tokens[0].move,[]);
  assert.deepEqual(grouped,original,move);
 }
});

test("formatting preserves inverse pairs, half turns and intervening rotations",()=>{
 const tokens=displayMoveTokens([
  batch(0,["R","R'","U2","U2","L"]),batch(1,["y"],true),batch(2,["L","x"]),batch(3,["x"],true),
 ]);
 assert.deepEqual(tokens.map(token=>token.move),["R","R'","U2","U2","L","y","L","x","x"]);
 assert.equal(displayMoveTokens([batch(0,["R’","R’"])])[0].move,"R2");
});

test("converted frame batches get identical formatting regardless of how commits were batched",()=>{
 const camera=multiply(axisAngle([Math.PI-Math.asin(Math.sqrt(2/3)),0,0]),axisAngle([0,0,-Math.PI/4]));
 const pose:CubePose={rotation:camera,axis:0,angles:[0,0,0],state:initial};
 const rows=[{frame:0,pose},{frame:4,pose}];
 const step=(frame_index:number,state_moves:string[]):MoveStep=>({frame_index,state_moves,geometry_option_index:0});
 const together=reorientedMoveTimeline(rows,[step(0,["R","R","U","F","F"])],CAMERA_ORIENTATION);
 const separate=reorientedMoveTimeline(rows,[step(0,["R"]),step(1,["R","U","F"]),step(2,["F"])],CAMERA_ORIENTATION);
 assert.deepEqual(displayMoveTokens(together).map(token=>token.move),["R2","U","F2"]);
 assert.deepEqual(displayMoveTokens(separate).map(token=>token.move),["R2","U","F2"]);
});
