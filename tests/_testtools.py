from functools import partial

import numpy as np
import hypothesis.strategies as st
from hypothesis.extra import numpy as st_numpy

# hypothesis generator shortcuts
real_floats = partial(st.floats, allow_nan=False, allow_infinity=False, width=32)
real_coords_2d = partial(st.lists, elements=real_floats(), min_size=2, max_size=2)


def coords_2d_array():
    return st_numpy.arrays(np.float_, st_numpy.array_shapes(2, 2),
                           elements=st.floats(allow_nan=False, allow_infinity=False))\
                    .filter(lambda arr: arr.shape[1] == 2)
