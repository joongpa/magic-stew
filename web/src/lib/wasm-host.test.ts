import assert from "node:assert/strict";
import test from "node:test";
import { loadCpuFitter } from "../../../pose-fitter/runtime/wasm-host.mjs";

const encoder=new TextEncoder();
const reportedPointer=-2147380216;
const high=reportedPointer>>>0;
const expected={Ok:{status:"running",states:790294}};
const noCamera={HEAPF64:new Float64Array(0),_malloc:()=>0,_free:()=>{},_pose_cv:()=>{throw new Error("Unexpected camera call");}};

function unsigned(value:number):number[] {
 const bytes:number[]=[];let n=BigInt(value);
 do{let byte=Number(n&127n);n>>=7n;if(n)byte|=128;bytes.push(byte);}while(n);
 return bytes;
}
function signed(value:bigint):number[] {
 const bytes:number[]=[];let n=value;
 while(true){const byte=Number(n&127n);n>>=7n;
  if((n===0n&&!(byte&64))||(n===-1n&&!!(byte&64))){bytes.push(byte);return bytes;}
  bytes.push(byte|128);
 }
}
const vector=(items:number[][])=>[...unsigned(items.length),...items.flat()];
const name=(text:string)=>[...unsigned(encoder.encode(text).length),...encoder.encode(text)];
const section=(id:number,bytes:number[])=>[id,...unsigned(bytes.length),...bytes];
const i32=(value:number)=>[0x41,...signed(BigInt.asIntN(32,BigInt(value)))];
const body=(code:number[])=>{const bytes=[0,...code,0x0b];return [...unsigned(bytes.length),...bytes];};
const f64Bytes=(values:number[])=>[...new Uint8Array(new Float64Array(values).buffer)];
const cvPointers=[0x8001a000,0x8001a020,0x8001a040,0,0x8001a090,0x8001a100];

/** Tiny real WASM module with the production pointer ABI. High-address tests
 * reserve a sparse >2 GiB memory but touch only a few pages, not 2 GiB of RAM.
 * No mocking of WASM i32/i64 signedness or JavaScript typed-array offsets. */
function fixture(options:{input?:number;output?:number;grow?:boolean;diagnostic?:boolean;cv?:boolean}={}) {
 const input=options.input??64,output=options.output??4096,payload=[...encoder.encode(JSON.stringify(expected))];
 const diagnostic=encoder.encode("high-address diagnostic"),diagnosticPointer=0x8001b000;
 const types=[
  [0x60,...vector([[0x7f]]),...vector([[0x7f]])],
  [0x60,...vector([[0x7f],[0x7f]]),0],
  [0x60,...vector([[0x7f],[0x7f]]),...vector([[0x7f]])],
  [0x60,...vector([[0x7f],[0x7f]]),...vector([[0x7e]])],
  [0x60,...vector(Array.from({length:10},()=>[0x7f])),...vector([[0x7f]])],
 ];
 const imports:number[][]=[];
 if(options.diagnostic)imports.push([...name("env"),...name("pose_diagnostic"),0,1]);
 const cvIndex=imports.length;
 if(options.cv)imports.push([...name("env"),...name("pose_cv"),0,4]);
 const imported=imports.length;
 const packed=BigInt.asIntN(64,(BigInt(output)<<32n)|BigInt(payload.length));
 const request:number[]=[];
 if(options.grow)request.push(...i32(1),0x40,0,0x1a);
 if(options.diagnostic)request.push(...i32(diagnosticPointer),...i32(diagnostic.length),0x10,0);
 if(options.cv){
  for(const arg of [0,cvPointers[0],cvPointers[1],1,cvPointers[2],cvPointers[3],0,cvPointers[4],cvPointers[5],6])request.push(...i32(arg));
  request.push(0x10,...unsigned(cvIndex),0x1a);
  // Verify Rust receives the numerical result at the real high output pointer.
  for(let i=0;i<6;i++)request.push(...i32(cvPointers[5]+i*8),0x2b,3,0,0x44,...f64Bytes([i+1]),
   0x62,0x04,0x40,0x00,0x0b);
 }
 request.push(0x42,...signed(packed));
 const entries:{pointer:number;bytes:number[]}[]=[{pointer:output,bytes:payload}];
 if(options.diagnostic)entries.push({pointer:diagnosticPointer,bytes:[...diagnostic]});
 if(options.cv){
  for(const [index,values] of [[0,[2,3,4]],[1,[5,6]],[2,[1,0,0,0,1,0,0,0,1]],[4,[7,8,9,10,11,12]]] as [number,number[]][])
   entries.push({pointer:cvPointers[index],bytes:f64Bytes(values)});
 }
 const minimum=Math.ceil(Math.max(input+1024,...entries.map(entry=>entry.pointer+entry.bytes.length),options.cv?cvPointers[5]+48:0)/65536);
 const exports=[
  [...name("memory"),2,0],
  ...["alloc","dealloc","load_fixtures","move_search_json"].map((text,index)=>[...name(text),0,...unsigned(imported+index)]),
  ...["result_json","probes_json","fit_browser_frame"].map(text=>[...name(text),0,...unsigned(imported+3)]),
 ];
 return new Uint8Array([0,97,115,109,1,0,0,0,
  ...section(1,vector(types)),
  ...(imports.length?section(2,vector(imports)):[]),
  ...section(3,vector([[0],[1],[2],[3]])),
  ...section(5,vector([[0,...unsigned(minimum)]])),
  ...section(7,vector(exports)),
  ...section(10,vector([body(i32(input)),body([]),body(i32(3)),body(request)])),
  ...section(11,vector(entries.map(entry=>[0,...i32(entry.pointer),0x0b,...unsigned(entry.bytes.length),...entry.bytes]))),
 ]);
}

for(const [label,input,output] of [
 ["below 2 GiB",64,4096],
 ["high fixture and request allocations",high,4096],
 ["signed packed result pointer",64,high],
 ["high allocations and results with memory growth",high,high+4096],
] as const){
 test(`CPU WASM bridge handles ${label}`,async()=>{
  const bytes=fixture({input,output,grow:true});
  assert.equal(WebAssembly.validate(bytes),true);
  const engine=await loadCpuFitter(bytes,encoder.encode("fixture"),noCamera);
  assert.deepEqual(engine.call("move_search_json",{type:"advance",budget:512}),expected);
  assert.deepEqual(engine.call("move_search_json",{type:"status"}),expected);
 });
}

test("standalone WASM bridge shares unsigned allocation and packed-result handling",async()=>{
 const {loadFitter}=await import(new URL("../../../pose-fitter/tools/wasm-host.mjs",import.meta.url).href);
 const engine=await loadFitter(fixture({input:high,output:high+4096,grow:true}),encoder.encode("fixture"));
 assert.equal(engine.count,3);
 assert.deepEqual(engine.result(),expected);
 assert.deepEqual(engine.fitBrowser({frame:1}),expected);
 assert.deepEqual(engine.probes(),expected);
});

test("diagnostic callback reads a signed high-address WASM pointer",async t=>{
 const messages:string[]=[];
 t.mock.method(console,"error",()=>{});
 const engine=await loadCpuFitter(fixture({diagnostic:true}),encoder.encode("fixture"),noCamera,
  {onDiagnostic:message=>messages.push(message)});
 assert.deepEqual(engine.call("move_search_json",{}),expected);
 assert.deepEqual(messages,["high-address diagnostic"]);
});

test("OpenCV bridge copies high-address Rust inputs and outputs using an unsigned OpenCV allocation",async()=>{
 const base=0x80020000,heap=new Float64Array(new ArrayBuffer(base+1024));
 const released:number[]=[],calls:number[][]=[];
 const cv={HEAPF64:heap,_malloc:()=>base|0,_free:(pointer:number)=>{released.push(pointer);},
  _pose_cv:(...args:number[])=>{
   calls.push(args);
   const [mode,objects,pixels,count,intrinsic,_distortion,distortionCount,guess,output,capacity]=args;
   assert.deepEqual([mode,count,distortionCount,capacity],[0,1,0,6]);
   assert.deepEqual([...heap.subarray(objects/8,objects/8+3)],[2,3,4]);
   assert.deepEqual([...heap.subarray(pixels/8,pixels/8+2)],[5,6]);
   assert.deepEqual([...heap.subarray(intrinsic/8,intrinsic/8+9)],[1,0,0,0,1,0,0,0,1]);
   assert.deepEqual([...heap.subarray(guess/8,guess/8+6)],[7,8,9,10,11,12]);
   heap.set([1,2,3,4,5,6],output/8);return 1;
  }};
 const engine=await loadCpuFitter(fixture({cv:true}),encoder.encode("fixture"),cv);
 assert.deepEqual(engine.call("move_search_json",{}),expected);
 assert.equal(calls.length,1);assert.deepEqual(released,[base]);
});
