"""Build pinned OpenCV camera kernels; no Svelte, browser, or detector needed."""
from pathlib import Path
import argparse,json,os,subprocess

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emsdk',type=Path,default=os.environ.get('EMSDK'))
    p.add_argument('--source',type=Path,required=True,help='Official OpenCV 5.0.0 source directory')
    p.add_argument('--build-dir',type=Path,default=ROOT/'pose-fitter/target/opencv-build')
    p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args()
    if not a.emsdk:p.error('--emsdk or EMSDK is required')
    sdk=a.emsdk.resolve();source=a.source.resolve();build=a.build_dir.resolve()
    version=(source/'modules/core/include/opencv2/core/version.hpp').read_text()
    if not all(f'#define CV_VERSION_{key}' in version for key in ('MAJOR','MINOR','REVISION')):p.error('Invalid OpenCV source tree')
    import re
    numbers=[int(re.search(rf'#define CV_VERSION_{key}\s+(\d+)',version)[1]) for key in ('MAJOR','MINOR','REVISION')]
    if numbers!=[5,0,0]:p.error('This parity build is pinned to OpenCV 5.0.0')
    env=os.environ.copy();env['PATH']=str(sdk/'upstream/emscripten')+os.pathsep+env['PATH'];env['EM_CONFIG']=str(sdk/'.emscripten')
    def run(argv):subprocess.run([str(v) for v in argv],check=True,env=env,cwd=ROOT)
    if not(build/'CMakeCache.txt').exists():
        flags=dict(BUILD_LIST='core,geometry',BUILD_SHARED_LIBS='OFF',BUILD_TESTS='OFF',BUILD_PERF_TESTS='OFF',
            BUILD_EXAMPLES='OFF',BUILD_opencv_apps='OFF',BUILD_JAVA='OFF',BUILD_opencv_python3='OFF',BUILD_opencv_js='OFF',
            WITH_IPP='OFF',WITH_LAPACK='OFF',WITH_EIGEN='OFF',WITH_OPENCL='OFF',WITH_ITT='OFF',WITH_PTHREADS_PF='OFF',
            CV_ENABLE_INTRINSICS='OFF',CPU_BASELINE='',CPU_DISPATCH='',CMAKE_BUILD_TYPE='Release',CMAKE_CXX_FLAGS='-ffp-contract=off',CMAKE_CXX_STANDARD='17')
        run([sdk/'upstream/emscripten/emcmake','cmake','-S',source,'-B',build,'-G','Ninja',*[f'-D{k}={v}' for k,v in flags.items()]])
    run(['cmake','--build',build,'--target','opencv_geometry','--parallel',a.jobs])
    target=ROOT/'pose-fitter/target';target.mkdir(exist_ok=True)
    run([sdk/'upstream/emscripten/em++',ROOT/'pose-fitter/tools/opencv-pnp.cpp',
        *[build/f'lib/libopencv_{name}.a' for name in ('geometry','flann','core')],build/'3rdparty/lib/libzlib.a',
        f'-I{build}',*[f'-I{source}/modules/{name}/include' for name in ('core','geometry','flann')],
        '-O3','-ffp-contract=off','-fexceptions','-sDISABLE_EXCEPTION_CATCHING=0','-sMODULARIZE=1','-sEXPORT_ES6=1',
        '-sENVIRONMENT=web,worker,node','-sALLOW_MEMORY_GROWTH=1','-sFILESYSTEM=0',
        '-sEXPORTED_FUNCTIONS=_pose_cv,_malloc,_free','-sEXPORTED_RUNTIME_METHODS=HEAPF64','-o',target/'opencv-pnp.mjs'])
    print(target/'opencv-pnp.wasm')

if __name__=='__main__':main()
