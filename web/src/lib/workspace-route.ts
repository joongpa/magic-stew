export type WorkspaceTab = 'detector' | 'timer';

export function workspacePath(tab:WorkspaceTab,basePath='/'){
  const root=basePath.replace(/\/+$/,'');
  return tab==='timer'?`${root}/timer`:`${root}/`;
}
export function workspaceFromPath(pathname:string,basePath='/'):WorkspaceTab{
  return pathname.replace(/\/+$/,'')===workspacePath('timer',basePath)?'timer':'detector';
}
