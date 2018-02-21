from math import pi
from functools import partial

import pytest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np

from geovoronoi._geom import angle_between_pts, inner_angle_between_vecs

# hypothesis generator shortcuts
real_floats = partial(st.floats, allow_nan=False, allow_infinity=False)
real_coords_2d = partial(st.lists, elements=real_floats(), min_size=2, max_size=2)


@given(a=real_coords_2d(), b=real_coords_2d())
def test_inner_angle_between_vecs(a, b):
    a = np.array(a)
    b = np.array(b)
    origin = np.array((0, 0))

    ang = inner_angle_between_vecs(a, b)

    if np.allclose(a, origin) or np.allclose(b, origin):
        assert np.isnan(ang)
    else:
        assert 0 <= ang <= pi


@given(a=real_coords_2d(), b=real_coords_2d())
def test_angle_between_pts(a, b):
    a = np.array(a)
    b = np.array(b)

    ang = angle_between_pts(a, b)

    if np.allclose(a, b):
        assert np.isnan(ang)
    else:
        assert 0 <= ang <= 2*pi
