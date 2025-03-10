import numpy as np
import vegas_params as vp

def test_StructArray_from_uniforms_and_fixed_values():
    e = vp.collections.StructArray(x = vp.Uniform([0,1]), y=42, z=vp.Uniform([[0,0],[10,20]]))
    s = e.sample(100)
    assert s.shape == (100,)
    assert s['x'].shape == (100,1)
    assert np.all(s['x']<=1) & np.all(s['x']>=0)
    assert s['y'].shape == (100,1)
    assert np.all(s['y']==42)
    assert s['z'].shape == (100,2)
    assert np.all(s['z']>=0) & np.all(s['z'][:,0]<=10) & np.all(s['z'][:,1]<=20)

def test_StructArray_from_vectors_and_scalars():
    e = vp.collections.StructArray(s = vp.Scalar(vp.Uniform([0,1])), v=vp.Vector([0,0,1]))
    s = e.sample(100)
    assert s.shape == (100,)
    assert s['s'].shape == (100,1)
    assert s['v'].shape == (100,3)