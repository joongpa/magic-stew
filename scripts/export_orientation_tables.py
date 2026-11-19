"""Export only immutable cube symmetries and sticker-face permutations."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
g=json.loads((ROOT/'pose-fitter/tests/fixtures/cpu-golden.json').read_text())['geometry']
centers=np.asarray(g['faces']).mean(axis=1);matrices=np.asarray(g['symmetries']);permutations=[]
for s in matrices:
 points=centers@s.T;indices=np.linalg.norm(points[:,None]-centers[None],axis=2).argmin(axis=1)
 assert np.max(np.linalg.norm(points-centers[indices],axis=1))<1e-5
 permutations.append(indices.tolist())
output=dict(matrices=matrices.tolist(),rvecs=[cv2.Rodrigues(s)[0].ravel().tolist() for s in matrices],
 face_permutations=permutations,identity_index=next(i for i,s in enumerate(matrices) if np.array_equal(s,np.eye(3))))
(ROOT/'pose-fitter/tables/orientations.json').write_text(json.dumps(output,separators=(',',':'))+'\n')
print('Exported 24 immutable cube orientations; no fitted poses or scored rows')
