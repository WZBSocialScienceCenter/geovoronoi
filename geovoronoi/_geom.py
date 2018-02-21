import numpy as np


def angle_between_pts(a, b, ref_vec=(1.0, 0.0)):
    ang = inner_angle_between_vecs(a - b, np.array(ref_vec))
    if not np.isnan(ang) and a[1] < b[1]:
        ang = 2 * np.pi - ang

    return ang


def inner_angle_between_vecs(a, b):
    origin = np.array((0, 0))
    if np.allclose(a, origin) or np.allclose(b, origin):
        return np.nan

    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    ang = np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0))

    return ang
