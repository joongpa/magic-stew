import itertools

import numpy as np
import pytest

from live_recognition.sticker_keypoint_order import clockwise_sticker_order, center_cyclic_orders, LANDMARK_SEQUENCES


def test_all_nine_face_cells_and_input_permutations():
    for x,y in itertools.product(range(3),repeat=2):
        q=np.array([(x,y),(x+1,y),(x+1,y+1),(x,y+1)],dtype=float)
        roles=np.array([0 if a in (0,3) and b in (0,3) else
                        1 if a in (0,3) or b in (0,3) else 2 for a,b in q])
        kind='corner' if 0 in roles else 'edge' if 1 in roles else 'center'
        for roll in [0.,.4,1.4,2.9]:
            R=np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
            projected=(q@R.T)/(.05*q[:,0:1]+1)
            canonical=None
            for permutation in itertools.permutations(range(4)):
                ix=np.array(permutation)
                order=clockwise_sticker_order(projected[ix],roles[ix],kind)
                result=projected[ix][order]
                assert tuple(roles[ix][order])==LANDMARK_SEQUENCES[kind]
                area=np.sum(result[:,0]*np.roll(result[:,1],-1)-result[:,1]*np.roll(result[:,0],-1))
                assert area>0
                if canonical is None:canonical=result
                np.testing.assert_allclose(result,canonical)


def test_center_equivalence_accepts_four_rotations_but_not_reflections():
    orders=center_cyclic_orders()
    assert len(set(map(tuple,orders)))==4
    assert not set(map(tuple,orders)) & set(map(tuple,orders[:,::-1]))
    assert all((np.diff(np.r_[row,row[0]])%4==1).all() for row in orders)


def test_invalid_quadrilateral_or_roles_are_rejected():
    with pytest.raises(ValueError,match='roles'):
        clockwise_sticker_order([[0,0],[1,0],[1,1],[0,1]],[0,0,1,2],'corner')
    with pytest.raises(ValueError,match='nondegenerate'):
        clockwise_sticker_order([[0,0],[1,0],[2,0],[3,0]],[2,2,2,2],'center')
