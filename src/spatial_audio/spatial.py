from abc import ABC, abstractmethod
from typing import Tuple
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
from scipy.spatial import ConvexHull
import spaudiopy as spa
from tqdm import tqdm

from utils import cart2sph, sph2cart, unpack_coordinates


def convert_A2B_format_tetramic(rirs_Aformat: NDArray) -> NDArray:
    """
    Convert A format tetramic RIRs to B-format using SN3D normalisation and ACN ordering.

    Parameters
    ----------
    rirs_Aformat : NDArray
        RIRs in A-format, of shape (num_time_samples, num_channels).

    Returns
    -------
    NDArray
        RIRs in B format of shape (num_time_samples, num_channels) [w, y, z, x] ordering.
    """
    # Assume 4 unit vectors for tetrahedral mic (each row is [x, y, z])
    dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    #### WRITE YOUR CODE HERE ####
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.acos(z)
    phi = np.atan2(y, x)

    # Create SN3D-normalized real SH basis functions (ACN order)
    # Order: [Y_0^0, Y_1^-1, Y_1^0, Y_1^1] => [W, Y, Z, X]
    Y_00 = 1 / np.sqrt(4 * np.pi) * np.ones_like(theta)
    Y_1m1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y_10 = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    Y_11 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)

    # Stack SH functions into shape (num_mic_dirs, num_channels)
    Y = np.column_stack((Y_00, Y_1m1, Y_10, Y_11))

    # Invert to get A → B transform
    Y_inv = np.linalg.pinv(Y)

    # Multiply wth inverted matrix with A-format RIRs to get B-format RIRs of
    # shape (num_time_samples, num_channels). Use einsum
    rirs_Bformat = (Y_inv @ rirs_Aformat.T).T

    # Return B-format RIRs of shape: (num_time_samples, num_channels) in ACN/SN3D
    return rirs_Bformat