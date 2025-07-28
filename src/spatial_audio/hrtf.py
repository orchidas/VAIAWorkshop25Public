from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft
import spaudiopy as spa
from tqdm import tqdm

from utils import cart2sph


@dataclass
class HRIRSet:
    """
    Dataclass for storing HRIR data and associated metadata.

    Attributes
    ----------
    fs : int
        Sample rate.
    num_rotations : int
        Number of data points in listener view.
    ir_len_samps : int
        Length of the HRIRs.
    hrir_data : NDArray
        HRIR data of shape (num_measurements, 2, num_time_samps).
    listener_view : NDArray
        Listener view coordinates in spherical coordinate system.
    listener_view_type : str
        Whether the listener view is in cartesian or spherical coordinates.
    source_view : Optional[NDArray], optional
        Source view coordinates in spherical coordinate system.
    """

    fs: int  # sample rate
    num_rotations: int  # number of data points in listener view
    ir_len_samps: int  # length of the HRIRs
    hrir_data: NDArray  # HRIR data of shape (num_measurements, 2, num_time_samps)
    listener_view: NDArray  # listener view coordinates in spherical coord system
    listener_view_type: str  # is the listener view in cartesian or spherical coordinates?
    source_view: Optional[
        NDArray] = None  # source view coordinates in spherical coord system

    def __post_init__(self):
        # ensure that listener view is specified in spherical coordinates
        if self.listener_view_type == "cartesian":
            self.listener_view = cart2sph(self.listener_view[:, 0],
                                          self.listener_view[:, 1],
                                          self.listener_view[:, 2])
            self.listener_view_type = "spherical"

        # following SOFA conventions
        if np.any(self.listener_view[:, 0] > 180):
            self.listener_view[:, 0] = (self.listener_view[:, 0] +
                                        180) % 360 - 180

        if np.any(self.listener_view[:, 1] > 90):
            self.listener_view[:, 1] -= 90

        # normalise for peak value 1
        self.hrir_data /= np.max(np.abs(self.hrir_data))


class HRIRInterpolator:
    """
    Class for interpolating HRIRs on a denser grid.
    """

    def __init__(self, hrir_data: HRIRSet):
        """
        Initialize the HRIRInterpolator.

        Parameters
        ----------
        hrir_data : HRIRSet
            HRIR reader object for the original dataset.
        """
        self.hrir_set = hrir_data

    def bilinear_interpolation(self, new_az_res: float,
                               new_el_res: float) -> HRIRSet:
        """
        Bilinear interpolation of HRIRs by finding the 4 nearest neighbours. Only works with an equiangular grid.

        Parameters
        ----------
        new_az_res : float
            Resolution of the azimuth angles.
        new_el_res : float
            Resolution of the elevation angles.

        Returns
        -------
        HRIRSet
            New HRIRSet with the interpolated HRIRs for the angles in the new grid.
        """

        def get_index(e_idx: int, a_idx: int):
            """Get the 4 HRIRs from the flattened data"""
            return e_idx * num_az + a_idx

        #### WRITE YOUR CODE HERE ####

        # create new grid of azimuth angles, in SOFA these range from [-180, +180] degrees
        # call the variable az_angles

        # create new grid of elevation angles, in SOFA these range from [-90 +90] degrees
        # call the variable el_angles

        # create a 2D meshgrid with both azimuth and elevation angles
        new_az_grid, new_el_grid = np.meshgrid(az_angles, el_angles)
        az_query = new_az_grid.ravel()
        el_query = new_el_grid.ravel()

        # stack them to create a new spherical grid
        new_grid_sph = np.stack(
            [az_query, el_query, np.ones_like(az_query)], axis=-1)

        # get original grid
        og_grid = self.hrir_set.listener_view
        az_grid = np.unique(og_grid[:, 0])
        el_grid = np.unique(og_grid[:, 1])
        num_az = len(az_grid)
        num_el = len(el_grid)

        hrirs_interp = []

        for az_new, el_new in tqdm(zip(az_query, el_query),
                                   total=len(az_query)):

            # Find indices of HRIRs in original dataset closest to az_new, el_new
            az_idx = np.searchsorted(az_grid, az_new) - 1
            az_idx = np.clip(az_idx, 0, num_az - 2)

            el_idx = np.searchsorted(el_grid, el_new) - 1
            el_idx = np.clip(el_idx, 0, num_el - 2)

            #### WRITE YOUR CODE HERE ####

            # find theta_grid and phi_grid

            # find c_theta and c_phi

            # get the interpolation weights

            # get the four nearest HRIRs (use get_index() function)

            # find the interpolated HRIR and append it to hrirs_interp

        # Stack all interpolated HRIRs into numpy array
        hrirs_interp = np.stack(
            hrirs_interp, axis=0)  # shape: (num_new_points, time_samples, 2)

        # create new HRIRSet
        new_hrir_set = HRIRSet(self.hrir_set.fs,
                               new_grid_sph.shape[0],
                               hrirs_interp.shape[-1],
                               hrirs_interp,
                               new_grid_sph,
                               "spherical",
                               source_view=self.hrir_set.source_view)
        return new_hrir_set
