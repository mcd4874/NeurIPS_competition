# f = open("ABMx10.txt", "r")
# print(f.readline())
import pandas as pd
import numpy as np
from collections import OrderedDict
from mne.channels.montage import make_dig_montage
# data = pd.read_csv(file_name, sep=" ")
# print(data)

_str = 'U100'

def _str_names(ch_names):
    return [str(ch_name) for ch_name in ch_names]
def _safe_np_loadtxt(fname, **kwargs):
    out = np.genfromtxt(fname, **kwargs)
    ch_names = _str_names(out['f0'])
    others = tuple(out['f%d' % ii] for ii in range(1, len(out.dtype.fields)))
    return (ch_names,) + others
def _sph_to_cart(sph_pts):
    """Convert spherical coordinates to Cartesion coordinates.

    Parameters
    ----------
    sph_pts : ndarray, shape (n_points, 3)
        Array containing points in spherical coordinates (rad, azimuth, polar)

    Returns
    -------
    cart_pts : ndarray, shape (n_points, 3)
        Array containing points in Cartesian coordinates (x, y, z)

    """
    assert sph_pts.ndim == 2 and sph_pts.shape[1] == 3
    sph_pts = np.atleast_2d(sph_pts)
    cart_pts = np.empty((len(sph_pts), 3))
    cart_pts[:, 2] = sph_pts[:, 0] * np.cos(sph_pts[:, 2])
    xy = sph_pts[:, 0] * np.sin(sph_pts[:, 2])
    cart_pts[:, 0] = xy * np.cos(sph_pts[:, 1])
    cart_pts[:, 1] = xy * np.sin(sph_pts[:, 1])
    return cart_pts

def _read_theta_phi_in_degrees(fname, head_size, fid_names=None,
                               add_fiducials=False):
    ch_names, theta, phi = _safe_np_loadtxt(fname, skip_header=1,
                                            dtype=(_str, 'i4', 'i4'))
    if add_fiducials:
        # Add fiducials based on 10/20 spherical coordinate definitions
        # http://chgd.umich.edu/wp-content/uploads/2014/06/
        # 10-20_system_positioning.pdf
        # extrapolated from other sensor coordinates in the Easycap layouts
        # https://www.easycap.de/wp-content/uploads/2018/02/
        # Easycap-Equidistant-Layouts.pdf
        assert fid_names is None
        fid_names = ['Nasion', 'LPA', 'RPA']
        ch_names.extend(fid_names)
        theta = np.append(theta, [115, -115, 115])
        phi = np.append(phi, [90, 0, 0])

    radii = np.full(len(phi), head_size)
    pos = _sph_to_cart(np.array([radii, np.deg2rad(phi), np.deg2rad(theta)]).T)
    ch_pos = OrderedDict(zip(ch_names, pos))

    nasion, lpa, rpa = None, None, None
    if fid_names is not None:
        nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, lpa=lpa, rpa=rpa)

# def generate_cartesian_coord(theta,phi,r=1):
#     x = r*np.sin(phi)* np.cos(theta)
#     y = r*np.sin(phi)* np.sin(theta)
#     z = r*np.cos(phi)
#     return [x,y,z]
# x,y,z = generate_cartesian_coord(data["Theta"].values,data["Phi"].values)
# print("x : ",x)
# print("y : ",y)
# print("z : ",z)
# # channel_names = data["Site"].values[:-3]
# data["x"] = x
# data["y"] = y
# data["z"] = z
# print(data)
#
from mne.defaults import HEAD_SIZE_DEFAULT

file_name = "ABMx10.txt"

montage = _read_theta_phi_in_degrees(fname=file_name, head_size=HEAD_SIZE_DEFAULT,
                                     fid_names=['Nz', 'LPA', 'RPA'],
                                     add_fiducials=False)

print(montage)