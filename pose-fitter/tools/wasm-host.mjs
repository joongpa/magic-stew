// Browser and Node share the exact same standalone WASM ABI.
export async function loadFitter(bytes, fixtureBytes) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const e=instance.exports;
  const ptr=e.alloc(fixtureBytes.length)>>>0;
  new Uint8Array(e.memory.buffer,ptr,fixtureBytes.length).set(fixtureBytes);
  const count=e.load_fixtures(ptr,fixtureBytes.length);
  e.dealloc(ptr,fixtureBytes.length);
  if(count<0)throw new Error('Invalid fitter fixture');
  const decode=packed=>{
    const ptr=Number((packed>>32n)&0xffffffffn),len=Number(packed&0xffffffffn);
    // Memory views are reacquired after each call: WASM memory may have grown.
    try{return JSON.parse(new TextDecoder().decode(new Uint8Array(e.memory.buffer,ptr,len)));}
    finally{e.dealloc(ptr,len);}
  };
  return { count, run:index=>e.run_case(index), result:()=>decode(e.result_json()), probes:()=>decode(e.probes_json()),
    callBrowser:(method,frame)=>{
      const input=new TextEncoder().encode(JSON.stringify(frame)),ptr=e.alloc(input.length)>>>0;
      new Uint8Array(e.memory.buffer,ptr,input.length).set(input);
      let result;
      try{result=e[method](ptr,input.length);}finally{e.dealloc(ptr,input.length);}
      return decode(result);
    },
    fitBrowser(frame){return this.callBrowser('fit_browser_frame',frame);},
    refineBrowser(frame){return this.callBrowser('fit_browser_neighbors',frame);}
  };
}
