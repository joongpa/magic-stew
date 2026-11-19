/** Keep large diagnostic JSON serialization off the camera/timer UI thread. */
self.onmessage=async({data}:MessageEvent<{baseUrl:string;capture:unknown}>)=>{
  try{
    const response=await fetch(new URL('__diagnostics/run',data.baseUrl),{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data.capture)});
    if(!response.ok)throw Error(await response.text());
    self.postMessage({ok:true});
  }catch(error){self.postMessage({ok:false,error:error instanceof Error?error.message:String(error)});}
};
export {};
