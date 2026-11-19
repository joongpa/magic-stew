import scheduler from '../../../pose-fitter/runtime/buffered-pipeline.mjs?raw';
import policy from '../../../pose-fitter/runtime/reconstruction-policy.mjs?raw';
import searchDriver from '../../../pose-fitter/runtime/incremental-search.mjs?raw';
import parallelDriver from '../../../pose-fitter/runtime/parallel-search.mjs?raw';
import frameAdapter from './pose-fitter.ts?raw';
import colors from './cube-colors.ts?raw';
const sha=async(source:string)=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(source)))].map(v=>v.toString(16).padStart(2,'0')).join('');
let pending:Promise<{scheduler:string;policy:string;searchDriver:string;frameAdapter:string;colors:string}>|undefined;
export function runtimeFingerprints(){return pending??=Promise.all([sha(scheduler),sha(policy),sha(searchDriver+'\n'+parallelDriver),sha(frameAdapter),sha(colors)])
  .then(([scheduler,policy,searchDriver,frameAdapter,colors])=>({scheduler,policy,searchDriver,frameAdapter,colors}));}
