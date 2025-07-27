from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import spaudiopy as spa

from utils import sph2cart


def plot_points_on_sphere(az: ArrayLike, el: ArrayLike):
    """
    Plot azimuth and elevation points on a unit sphere.

    Parameters
    ----------
    az : ArrayLike
        Array of azimuth angles (in degrees).
    el : ArrayLike
        Array of elevation angles (in degrees).

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    if len(az) != len(el):
        az_grid, el_grid = np.meshgrid(az, el)
    else:
        az_grid, el_grid = az, el

    # Flatten and convert to Cartesian
    new_grid = sph2cart(az_grid.ravel(), el_grid.ravel(),
                        np.ones_like(az_grid.ravel()))
    x, y, z = new_grid.T

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')

    # Sphere outline (optional)
    u, v = np.mgrid[-np.pi:np.pi:60j, -np.pi / 2:np.pi / 2:30j]
    xs = np.cos(v) * np.cos(u)
    ys = np.cos(v) * np.sin(u)
    zs = np.sin(v)
    ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.7)

    ax.set_title("Azimuth-Elevation Grid on Unit Sphere")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    plt.tight_layout()
    plt.show()
