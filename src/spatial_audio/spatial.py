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
    Y_inv = np.linalg.inv(Y)

    # Multiply wth inverted matrix with A-format RIRs to get B-format RIRs of
    # shape (num_time_samples, num_channels). Use einsum
    rirs_Bformat = (Y_inv @ rirs_Aformat.T).T

    # Return B-format RIRs of shape: (num_time_samples, num_channels) in ACN/SN3D
    return rirs_Bformat


def convert_srir_to_brir(srirs: NDArray, hrir_sh: NDArray,
                         head_orientations: ArrayLike) -> NDArray:
    """
    Convert SRIRs to BRIRs for specific orientations using spherical harmonic transformation.

    Parameters
    ----------
    srirs : NDArray
        SRIRs of shape (num_pos, num_ambi_channels, num_time_samp).
    hrir_sh : NDArray
        HRIRs encoded in SH domain of shape (num_ambi_channels, 2, num_time_samps).
    head_orientations : ArrayLike
        Head orientations of shape (num_ori, 2) in degrees.

    Returns
    -------
    NDArray
        BRIRs of shape (num_pos, num_ori, num_time_samples, 2).
    """
    ambi_order = int(np.sqrt(srirs.shape[1] - 1))
    num_receivers = srirs.shape[0]
    num_freq_bins = 2**int(np.ceil(np.log2(srirs.shape[-1])))

    # take FFT of SRIRs - size is num_receivers x num_ambi_channels x num_time_samples
    ambi_rtfs = rfft(srirs, num_freq_bins, axis=-1)

    # take FFT of SH-HRIRs these are of shape num_ambi_channels x 2 x num_freq_samples
    ambi_hrtfs = rfft(hrir_sh, n=num_freq_bins, axis=-1)
    logger.info("Done calculating FFTs")

    num_orientations = head_orientations.shape[0]
    brirs = np.zeros((num_receivers, num_orientations, num_freq_bins, 2))

    #### WRITE YOUR CODE HERE ####

    # loop through receiver positions
    for rec_pos_idx in tqdm(range(num_receivers)):

        # get current SRIR FFT = shape is num_ambi_channels x num_freqs
        cur_ambi_rtf = ambi_rtfs[rec_pos_idx, ...]

        # loop through head orientations
        for ori_idx in range(num_orientations):

            # get current head orientation
            cur_head_orientation = head_orientations[ori_idx, :]

            # get rotation matrix in the opposite direction - size num_freq_bins x num_ambi_channels
            cur_rotation_matrix = spa.sph.sh_rotation_matrix(
                ambi_order,
                np.radians(-cur_head_orientation[0]),
                np.radians(-cur_head_orientation[1]),
                0,
                sh_type='real')

            # get current rotated SRIR
            rotated_ambi_rtf = cur_rotation_matrix @ cur_ambi_rtf

            # get the binaural room transfer function by conjugating
            # freq-domain SRIRs and multiplying them with SH-HRTFs
            cur_brtf = np.einsum('nrf, nf -> fr', np.conj(ambi_hrtfs),
                                 rotated_ambi_rtf)

            # get the BRIR by taking an inverse FFT and save it to current
            # receiver position and orientation index
            cur_brir = irfft(cur_brtf, n=num_freq_bins, axis=0)
            brirs[rec_pos_idx, ori_idx, ...] = cur_brir

    return brirs


#####################################################################


class VBAP(ABC):
    """
    Base class to do 3D Vector based amplitude panning for a given loudspeaker grid.
    """

    def __init__(self, num_loudspeakers: int, emitter_positions: NDArray):
        """
        Parameters
        ----------
        num_loudspeakers : int
            Number of loudspeakers in the layout.
        emitter_positions : NDArray
            Positions of the loudspeakers (2D or 3D), must be (num_loudspeakers, ndim).
        """
        self.num_loudspeakers = num_loudspeakers
        self.emitter_positions = emitter_positions

    @abstractmethod
    def process(self, target_dir: NDArray) -> NDArray:
        """
        Abstract method for processing.

        Parameters
        ----------
        target_dir : NDArray
            Target direction vector.

        Returns
        -------
        NDArray
            Output gain vector for loudspeakers.
        """
        pass


class VBAP_3D(VBAP):
    """
    Class for doing 3D panning with VBAP with an arbitrary number of loudspeakers.
    """

    def __init__(self, num_loudspeakers: int, emitter_positions: NDArray):
        """
        Parameters
        ----------
        num_loudspeakers : int
            Number of loudspeakers in the layout.
        emitter_positions : NDArray
            Positions of the loudspeakers (2D or 3D) in Cartesian coordinates, must be (num_loudspeakers, ndim).
        """
        super().__init__(num_loudspeakers, emitter_positions)
        self.find_adjacent_triplets()
        # logger.debug(self.loudspeaker_triplets)
        self.find_triangle_inverse()
        # logger.debug(self.matrix_inverse)
        self.loudspeaker_gains = np.zeros(num_loudspeakers)
        # are there loudspeakers below the ear level
        self.find_speakers_below_ear_level()

    def find_speakers_below_ear_level(self):
        """Check if any of the loudspeakers are below the ear level"""
        # find the elevation of all loudspeakers
        sph_coord = cart2sph(self.emitter_positions[:, 0],
                             self.emitter_positions[:, 1],
                             self.emitter_positions[:, 2])
        el = sph_coord[:, 1]
        self.is_speaker_below_ear_level = el < 0.0

    def find_adjacent_triplets(self):
        """Find all adjacent triplets of loudspeakers in setup using a convex hull"""
        # points must be (npoints, ndim)
        convex_hull = ConvexHull(self.emitter_positions)
        all_triplets = convex_hull.simplices
        self.loudspeaker_triplets = []

        # a triangle cannot be made of speakers on the floor
        for triplet in all_triplets:
            if np.linalg.det(self.emitter_positions[triplet]) != 0:
                self.loudspeaker_triplets.append(triplet)

    def find_triangle_inverse(self):
        """Find the inverse matrix formed from an active triangle"""
        self.matrix_inverse = {}
        for triplet in self.loudspeaker_triplets:
            self.matrix_inverse[tuple(triplet)] = np.linalg.inv(
                self.emitter_positions[triplet])

    def find_active_triangle(self,
                             target_dir: ArrayLike) -> Tuple[NDArray, NDArray]:
        """
        Find active triangle of loudspeakers for a target direction vector.

        Parameters
        ----------
        target_dir : NDArray
            Target direction vector in 3D cartesian coordinates.

        Returns
        -------
        tuple of NDArray, NDArray
            The indices of the active loudspeaker triplet, and their corresponding gains.
        """
        # we calculate gains for all simplices of measurements
        k = 0
        gains_all = np.zeros((len(self.loudspeaker_triplets), 3), dtype=float)
        for triplet in self.loudspeaker_triplets:
            gains_all[k, :] = np.matmul(self.matrix_inverse[tuple(triplet)].T,
                                        target_dir)
            k += 1

        # catch any unstable gains
        unstable_gains = gains_all > 1e10
        if np.any(unstable_gains):
            gains_all[unstable_gains] = -1.0

        # choose triangle with largest min coefficient
        # normally all but one will be negative
        min_coeff_per_simplex = np.min(gains_all, axis=1)
        matching_simplex = np.argmax(min_coeff_per_simplex, axis=0)
        indices = self.loudspeaker_triplets[matching_simplex]
        return indices, gains_all[matching_simplex, :]

    def process(self, target_dir: NDArray) -> NDArray:
        """
        Find loudspeaker gains corresponding to the target direction vector.

        Parameters
        ----------
        target_dir : NDArray
            Target direction vector in 3D cartesian coordinates.

        Returns
        -------
        NDArray
            The gain vector for each loudspeaker in the setup.
        """
        # if the target direction vector points down, then we will have to invert a
        # singular matrix of triplet speakers on the floor, which is not possible.
        # A hack to fix that is, if there is no loudspeaker below ear level,
        # then force the elevation of the target vector to be slightly above ear level
        if not np.all(self.is_speaker_below_ear_level):
            az, el, dist = unpack_coordinates(
                cart2sph(target_dir[:, 0], target_dir[:, 1], target_dir[:, 2]))
            el[el < 0.0] = 1e-3
            target_dir = sph2cart(az, el, dist)

        num_target_dir = target_dir.shape[0]
        self.loudspeaker_gains = np.zeros(
            (self.num_loudspeakers, num_target_dir))

        for i in range(num_target_dir):
            loudspeaker_idx, gains = self.find_active_triangle(target_dir[i])
            scaled_gains = gains / np.linalg.norm(gains)

            for k in range(len(loudspeaker_idx)):
                self.loudspeaker_gains[loudspeaker_idx[k], i] = scaled_gains[k]

        return self.loudspeaker_gains
