"""Export immutable cube conventions and color rules; no poses or search results."""
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as r
from scripts.search_independent_keypoint_full_solve import PreparedReplayScorer

def export_render_geometry():
 # The separate viewer uses the same physical face IDs/layers as the fitter,
 # with a fixed camera instead of the recorded source-camera translation.
 faces=r._turn_gpu_static_geometry()
 data=dict(cube_size=r.CUBE_SIZE_MM,faces=faces[2].tolist(),normals=faces[5].tolist(),
  layers=[r._parallel_slab_geometry(axis)[2].tolist() for axis in range(3)])
 out=ROOT/'pose-fitter/tables/render-geometry.json'
 out.write_text(json.dumps(data,separators=(',',':'))+'\n');print(out)
 # OpenCV oracle verifies browser Rodrigues/layer conventions independently.
 import cv2,numpy as np
 cases=[]
 for axis,angles,rvec in [(0,[0.,0.,0.],[0.,0.,0.]),(0,[23.,0.,-37.],[.3,-.5,.1]),
                          (1,[-42.,0.,16.],[-.6,.4,.7]),(2,[31.,0.,-28.],[.2,.8,-.4])]:
  orientation=cv2.Rodrigues(np.array(rvec))[0];projected=[]
  for i,face in enumerate(data['faces']):
   vector=np.zeros(3);vector[axis]=np.deg2rad(angles[data['layers'][axis][i]])
   rot=orientation@cv2.Rodrigues(vector)[0]
   points=np.asarray(face)@rot.T/data['cube_size'];center=points.mean(axis=0)
   if np.dot(rot@np.asarray(data['normals'][i]),np.array([0.,0.,-4.])-center)<=0:continue
   projected.append(dict(face=i,depth=float(center[2]+4.),points=(3.*points[:,:2]/(points[:,2:3]+4.)).tolist()))
  cases.append(dict(axis=axis,angles=angles,params=rvec+[150.,-200.,900.,angles[0],angles[2]],expected=projected))
 oracle=ROOT/'pose-fitter/tests/fixtures/cube-projection.json'
 oracle.write_text(json.dumps(dict(cases=cases),separators=(',',':'))+'\n');print(oracle)
 # Signed slab angles for the presentation animator, using CubeState's exact
 # notation conventions, including wide, slice and whole-cube moves.
 from live_recognition.cube_state import FACE_AXES
 turns={}
 for face in ('U','D','L','R','F','B','Uw','Dw','Lw','Rw','Fw','Bw','M','E','S','x','y','z','r','l','u','d','f','b'):
  for suffix in ('',"'",'2'):
   wide=face[0] in 'rludfb' or face.endswith('w')
   base=face[0].upper() if face[0] in 'rludfb' else face[0]
   vector,side=FACE_AXES[base];axis=int(np.argmax(np.abs(vector)))
   layers=range(3) if side is None else (1,) if side==0 else (range(1,3) if vector[axis]>0 else range(2)) if wide else (2 if vector[axis]>0 else 0,)
   angles=[0,0,0]
   for layer in layers:angles[layer]=vector[axis]*(1 if suffix=="'" else -2 if suffix=='2' else -1)*90
   turns[face+suffix]=dict(axis=axis,angles=angles)
 out=ROOT/'pose-fitter/tables/render-turns.json'
 out.write_text(json.dumps(turns,separators=(',',':'))+'\n');print(out)

def main():
 export_render_geometry()
 s=PreparedReplayScorer(r,{},'cpu');identity=r.CubeGeometryState.identity();labels=sorted(s.identity_indices)
 groups=[]
 for axis in range(3):
  for offsets in itertools.product((-1,0,1,2),repeat=3):
   moves=tuple(r.cube_rotation_to_notation(v) for v in r.parallel_layer_rotations(axis,tuple(90.*q for q in offsets)))
   groups.append(dict(axis=axis,offsets=offsets,moves=moves,sources=s.move_sources(moves).tolist()))
 moves={}
 for group in groups:
  for move in group['moves']:
   moves[move]=s.move_sources((move,)).tolist()
 # All ordinary notation, including whole rotations used by supplied scrambles.
 for face in ('U','D','L','R','F','B','Uw','Dw','Lw','Rw','Fw','Bw','M','E','S','x','y','z','r','l','u','d','f','b'):
  for suffix in ('',"'",'2'):
   token=face+suffix
   try:moves[token]=s.move_sources((token,)).tolist()
   except (ValueError,KeyError):pass
 colors={i.label:r.CubeState.solved().color_at((i.cubie_index,i.axis,i.side)) for i in r.canonical_sticker_identities()}
 data=dict(version=1,identity=list(s.key(identity)),geometry_positions=s.geometry_positions.tolist(),
  face_groups=s.face_groups,solved_colors=[colors[label] for label in labels],groups=groups,moves=moves,
  confusion={k:sorted(v) for k,v in r.POSE_26M_COLOR_CONFUSIONS.items()},
  keypoint_scale=r.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION,sticker_scale=r.PNP_TURN_CENTER_PENALTY_SCALE)
 out=ROOT/'pose-fitter/tables/move-tables.json';out.write_text(json.dumps(data,separators=(',',':'))+'\n');print(out)
 cases=[]
 for scramble,steps in [
  ('',[(2,['R']),(4,['U']),(6,["U'","R'"])]),
  ("R U R' U'",[(10,['U']),(12,['R']),(14,["U'","R'"])]),
  ("Rw' F2 U Lw",[(20,['M',"E'",'S2']),(22,['x',"y'",'z2']),(24,['Uw2',"Dw'",'Bw'])]),
  ("R2 U2",[(30,['U2','R2'])]),
  ("r2 d’",[(40,['Dw','Rw2'])]),
 ]:
  cube=r.CubeState.from_scramble(r.parse_scramble_moves(scramble));state=identity
  colors={label:cube.colors[position] for position,label in identity.facelets.items()}
  def face_colors():return [colors[state.facelets[s.positions[index]]] for index in s.geometry_positions]
  case=dict(scramble=scramble,initial=face_colors(),steps=[])
  for frame,moves_for_frame in steps:
   state=state.apply_moves(moves_for_frame);case['steps'].append(dict(frame_index=frame,state_moves=moves_for_frame,expected=face_colors()))
  cases.append(case)
 oracle=ROOT/'pose-fitter/tests/fixtures/cube-colors.json';oracle.write_text(json.dumps(dict(cases=cases),separators=(',',':'))+'\n');print(oracle)
if __name__=='__main__':main()
