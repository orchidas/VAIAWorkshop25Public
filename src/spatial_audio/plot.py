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


def plot_spherical_harmonics(orders_list: List):
    """
    Plot spherical harmonic functions on the sphere for all orders in orders_list.

    Parameters
    ----------
    orders_list : list of int
        List of spherical harmonic orders to plot.

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    # Generate a spherical grid
    n_theta = 200
    n_phi = 100
    theta = np.linspace(0, 2 * np.pi, n_phi)  # azimuth [0, 2π]
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_theta)  # colatitude [0, π]
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert to cartesian for plotting - size is n_theta x n_phi
    x = np.cos(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.cos(phi_grid)
    z = np.sin(phi_grid)

    # Loop through SH orders 1 to 4
    for order in orders_list:
        n_coeffs = (order + 1)**2
        # flatten the grid for sh evaluation
        Y = spa.sph.sh_matrix(order, theta_grid.ravel(),
                              np.pi / 2 - phi_grid.ravel(),
                              'real')  # shape (N_pts, (N+1)^2)
        fig = plt.figure(figsize=(6, 3 * n_coeffs))

        for i in range(n_coeffs):
            coeff = Y[:, i].reshape(theta_grid.shape)
            coeff /= np.max(np.abs(coeff))  # normalize between +-1

            # radius of sphere
            r = 1
            # to plot the sphere
            X = r * x
            Y_ = r * y
            Z = r * z

            ax = fig.add_subplot(n_coeffs, 1, i + 1, projection='3d')
            ax.plot_surface(
                X,
                Y_,
                Z,
                facecolors=plt.cm.seismic(
                    (coeff + 1) /
                    2),  # this maps coeffs to 0 to 1 from -1 to 1
                rstride=1,
                cstride=1,
                antialiased=True,
                linewidth=0,
                edgecolor=None,
            )
            ax.set_title(f"Spherical Harmonic: order {order}, index {i}")
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()
