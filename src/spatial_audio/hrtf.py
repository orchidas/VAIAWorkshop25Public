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

    def get_spherical_harmonic_representation(self,
                                              ambi_order: int) -> NDArray:
        """
        Get the spherical harmonic representation of the HRTFs using specified ambisonics order.

        Parameters
        ----------
        ambi_order : int
            Spherical harmonics order.

        Returns
        -------
        NDArray
            The HRIRs in the SH domain of shape (num_ambi_channels, 2, num_time_samples).
        """
        # 1. Compute HRTFs from time-domain HRIRs
        fft_size = 2**int(np.ceil(np.log2(self.ir_len_samps)))

        #### WRITE YOUR CODE HERE ####

        # Get the FFT of the HRIRs and save it as the variable hrtfs

        # Create the spherical grid
        incidence_az = np.deg2rad(self.listener_view[..., 0])
        # zenith angle is different from elevation angle
        incidence_zen = np.deg2rad(90 -
                                   self.listener_view[..., 1])  # zenith angle

        # Get quadrature weights and create a diagonal matrix out of them, call it W.

        # Get spherical harmonic matrix, Y, using incidence_az, incidence_zen - shape (num_dirs, num_sh_channels)

        # Calculate (WY)^\dagger W

        # Multiply (WY)^\dagger W with hrtfs to get output of shape num_sh_channels, 2, num_freq_bins

        # Take inverse FFT to get SH domain HRIR of shape: (num_sh_channels, 2, num_time_samples) and return it

        return


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
        az_angles = np.arange(-180, 180, new_az_res)

        # create new grid of elevation angles, in SOFA these range from [-90 +90] degrees
        # call the variable el_angles
        el_angles = np.arange(-90, 90, new_el_res)

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
            theta_grid = np.abs(az_grid[az_idx] - az_grid[az_idx + 1])
            phi_grid = np.abs(el_grid[el_idx] - el_grid[el_idx + 1])

            # find c_theta and c_phi
            c_theta = np.mod(az_new, theta_grid) / theta_grid
            c_phi = np.mod(el_new, phi_grid) / phi_grid

            # get the interpolation weights
            w_A = (1 - c_theta) * (1 - c_phi)
            w_B = c_theta * (1 - c_phi)
            w_C = c_theta * c_phi
            w_D = (1 - c_theta) * c_phi

            # get the four nearest HRIRs (use get_index() function)
            h_A = self.hrir_set.hrir_data[get_index(el_idx, az_idx + 1)]
            h_B = self.hrir_set.hrir_data[get_index(el_idx, az_idx)]
            h_C = self.hrir_set.hrir_data[get_index(el_idx + 1, az_idx)]
            h_D = self.hrir_set.hrir_data[get_index(el_idx + 1, az_idx + 1)]

            # find the interpolated HRIR and append it to hrirs_interp
            interp = w_A * h_A + w_B * h_B + w_C * h_C + w_D * h_D
            hrirs_interp.append(interp)

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
