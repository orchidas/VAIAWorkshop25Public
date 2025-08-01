from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfft
from scipy.signal import istft, stft
from tqdm import tqdm

from utils import DataLogger, sph2cart

from .spatial import VBAP_3D


@dataclass
class SIRRParameters:
    """
    SIRR parameters per time frame.

    Attributes
    ----------
    intensity_vector : NDArray
        Intensity vector for each time frame.
    diffuseness_metric : ArrayLike
        Diffuseness metric for each time frame.
    azimuth : ArrayLike
        Azimuth angles for each time frame.
    elevation : ArrayLike
        Elevation angles for each time frame.
    """

    intensity_vector: NDArray
    diffuseness_metric: ArrayLike
    azimuth: ArrayLike
    elevation: ArrayLike


class SIRR:

    def __init__(
        self,
        srir: NDArray,
        fs: float,
        loudspeaker_positions: NDArray,
        loudspeaker_position_coord: str = "cartesian",
        ambi_order: int = 1,
        win_size: int = 2**9,
        hop_size: int = 2**8,
        fft_size: int = 2**10,
    ):
        """
        Class to implement Spatial Impulse Response Rendering of SRIRs.

        Parameters
        ----------
        srir : NDArray
            B-format SRIR of shape (num_time_samples, num_channels).
        fs : float
            Sample rate.
        loudspeaker_positions : NDArray
            Virtual loudspeaker positions for rendering, of shape (num_loudspeakers, 3).
        loudspeaker_position_coord : str, optional
            Whether coordinates are in spherical or cartesian coordinates (default is "cartesian").
        ambi_order : int, optional
            Ambisonics order of the SRIRs (default is 1).
        win_size : int, optional
            Window size for STFT (default is 2**9).
        hop_size : int, optional
            Hop size for STFT (default is 2**8).
        fft_size : int, optional
            FFT size for STFT (default is 2**10).
        """
        self.srir = srir
        (self.ir_length, num_chans) = self.srir.shape
        self.sample_rate = fs
        self.ambi_order = ambi_order
        self.num_chans = (self.ambi_order + 1)**2
        assert self.num_chans == num_chans, "Ambisonics order mismatch!"

        # initialise STFT parameters
        self._init_stft(win_size, hop_size, fft_size)

        # initialise output loudspeaker layput and VBAP
        self._init_vbap(loudspeaker_positions, loudspeaker_position_coord)

        # initialise decorrelators
        self._init_decorrelation()

        # to log DoAs
        self.history = DataLogger(max_len=2000)

    @property
    def impedance(self) -> float:
        """
        Characteristic impedance of air.

        Returns
        -------
        float
            The characteristic impedance of air (415 Pa s/m).
        """
        return 415.0

    @property
    def unit_vectors(self) -> NDArray:
        """
        Unit vectors in 3D cartesian coordinates.

        Returns
        -------
        NDArray
            List of unit vectors in 3D cartesian coordinates.
        """
        # Assume ACN channel ordering
        unit_vectors = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                                  dtype=np.complex64)
        return [vec / np.linalg.norm(vec) for vec in unit_vectors]

    def _init_vbap(self, loudspeaker_positions: NDArray,
                   loudspeaker_position_coord: str):
        # ensure everything is in cartesian coordinates for VBAP
        self.loudspeaker_positions = loudspeaker_positions
        if loudspeaker_position_coord == "spherical":
            self.loudspeaker_positions = sph2cart(
                self.loudspeaker_positions[:, 0],
                self.loudspeaker_positions[:, 1],
                self.loudspeaker_positions[:, 2])
        self.num_loudspeakers = self.loudspeaker_positions.shape[0]
        self.vbap = VBAP_3D(self.num_loudspeakers, self.loudspeaker_positions)

    def _init_stft(self, win_size: int, hop_size: int, fft_size: int):
        """
        Initialise STFT of the SRIR.

        Parameters
        ----------
        win_size : int
            Window size for STFT.
        hop_size : int
            Hop size for STFT.
        fft_size : int
            FFT size for STFT.
        """
        self.win_size = win_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        # Calculate STFT per channel
        freqs, time_frames, srir_stft = zip(*(stft(srir_ch,
                                                   fs=self.sample_rate,
                                                   nperseg=self.win_size,
                                                   noverlap=self.hop_size,
                                                   nfft=self.fft_size)
                                              for srir_ch in self.srir.T))

        # srir_stft is now a tuple of STFTs, one for each channel
        # Each srir_stft[i] has shape (n_freq_bins, n_time_frames)
        # shape: (n_channels, n_freq_bins, n_time_frames)
        self.srir_stft = np.stack(srir_stft, axis=0)

        self.freqs = freqs[0]
        self.time_frames = time_frames[0]
        self.num_time_frames = len(self.time_frames)
        self.num_freq_bins = len(self.freqs)

    def _init_decorrelation(self):
        """
        Initialise noise sequences used for decorrelation.
        """
        self.noise = np.random.randn(self.num_loudspeakers,
                                     2 * self.num_freq_bins - 1)
        self.noise_fft = rfft(self.noise,
                              n=2 * self.num_freq_bins - 1,
                              axis=-1)

    def calculate_parameters(self,
                             cur_stft_frame: NDArray,
                             _eps: float = 1e-9) -> SIRRParameters:
        """
        Calculate the directional vectors and diffuseness metric from the current STFT frame.

        Parameters
        ----------
        cur_stft_frame : NDArray
            Current STFT frame of shape (num_channels, num_freq_bins).
        _eps : float, optional
            Small epsilon to avoid division by zero (default is 1e-9).

        Returns
        -------
        SIRRParameters
            Dataclass containing directional and diffuseness parameters.
        """
        #### WRITE YOUR CODE HERE ####

        # calculate the velocity vector from the X,Y,Z channels of B-format RIRs
        X_t = np.zeros((self.num_freq_bins, 3), dtype=np.complex64)

        for k in range(self.num_chans - 1):
            X_t += np.einsum('f, c -> fc', cur_stft_frame[k + 1, :],
                             self.unit_vectors[k].squeeze())

        # calculate the intensity vector from the velocity vector and W channel
        inner_product = np.real(
            np.einsum('f, fc -> fc', np.conj(cur_stft_frame[0, :]), X_t))

        intensity = (np.sqrt(2) / self.impedance) * inner_product

        # calculate diffuseness metric
        diffuseness = 1 - (
            (np.sqrt(2) * np.linalg.norm(inner_product, axis=-1)) /
            (np.abs(cur_stft_frame[0, :])**2 +
             0.5 * np.linalg.norm(X_t, axis=-1)**2))

        # calculate azimuth and elevation from the diffuseness metric
        azimuth = np.arctan2(-intensity[:, 1], -intensity[:, 0])
        elevation = np.arctan2(
            -intensity[:, 2], np.sqrt(intensity[:, 0]**2 + intensity[:, 1]**2))

        # return an object of type SIRRParameters
        return SIRRParameters(intensity, diffuseness, azimuth, elevation)

    def process_frame(self, cur_stft_frame: NDArray) -> NDArray:
        """
        Process the current STFT frame by using SMOOTHED SIRR PARAMETERS.
        self.smoothed_parameters is of type SIRRParameters and has the
        paramters smoothed over time.

        Parameters
        ----------
        cur_stft_frame : NDArray
            Current STFT frame of shape (num_channels, num_freq_bins).

        Returns
        -------
        NDArray
            The signals for the loudspeaker setup of shape (num_loudspeakers, num_freq_bins).
        """
        #### WRITE YOUR CODE HERE ####

        cur_output_frame = np.zeros(
            (self.num_loudspeakers, self.num_freq_bins), dtype=np.complex64)

        # decompose into directional part = sqrt(1 - smoothed_diffuseness_metric) * W
        directional_part = np.sqrt(
            (1 -
             self.smoothed_params.diffuseness_metric)) * cur_stft_frame[0, ...]

        # process directional part
        cur_output_frame = self.process_directional_part(directional_part)

        # decompose into diffuse part = smoothed_diffuseness_metric * W**2
        diffuse_part = self.smoothed_params.diffuseness_metric * np.power(
            cur_stft_frame[0, ...], 2)

        # process diffuse part
        cur_output_frame += self.process_diffuse_part(diffuse_part)

        # add directional and diffuse parts and return output
        return cur_output_frame

    def process_directional_part(self, directional_part: NDArray) -> NDArray:
        """
        Process directional part.

        Parameters
        ----------
        directional_part : NDArray
            Directional part of the omni signal, of shape (num_freq_bins,).

        Returns
        -------
        NDArray
            Directional signals for each loudspeaker of shape (num_loudspeakers, num_freq_bins).
        """
        #### WRITE YOUR CODE HERE ####

        # get the target direction of the directional component in cartesian
        # coordinates from the smoothed DoAs
        target_dir = sph2cart(self.smoothed_params.azimuth,
                              self.smoothed_params.elevation,
                              np.ones(self.num_freq_bins),
                              degrees=False)

        # get loudspeaker_gains using VBAP by calling self.vbap.process()
        # shape is  (num_loudspeakers, num_freq_bins)
        loudspeaker_gains = self.vbap.process(target_dir).astype(np.complex64)

        # get the directional signal for each loudspeaker by
        # multiplying directional_part with loudspeaker_gains
        # shape is (num_loudspeakers, num_freq_bins)
        processed_directional_part = np.einsum('nf, f -> nf',
                                               loudspeaker_gains,
                                               directional_part)

        # return directional part for all loudspeakers
        return processed_directional_part

    def process_diffuse_part(self, diffuse_part: NDArray) -> NDArray:
        """
        Process diffuse part.

        Parameters
        ----------
        diffuse_part : NDArray
            Diffuse part of the omni signal, of shape (num_freq_bins,).

        Returns
        -------
        NDArray
            Diffuse signals for each loudspeaker of shape (num_loudspeakers, num_freq_bins).
        """
        # re-randomize phase
        self._init_decorrelation()
        # perform phase randomization
        target_mag = np.tile(np.abs(diffuse_part), (self.num_loudspeakers, 1))
        target_phase = np.angle(self.noise_fft)

        processed_diffuse_part = target_mag * np.exp(1j * target_phase)
        return processed_diffuse_part

    def smooth_parameters(self,
                          cur_params: SIRRParameters,
                          smooth_factor: float = 0.95):
        """
        Smooth parameters across time frames.

        Parameters
        ----------
        cur_params : SIRRParameters
            Current SIRR parameters.
        smooth_factor : float, optional
            Smoothing factor (default is 0.95).
        """
        for field in fields(cur_params):
            name = field.name
            current_value = getattr(cur_params, name)
            previous_smoothed = getattr(self.smoothed_params, name)
            smoothed_value = smooth_factor * previous_smoothed + (
                1 - smooth_factor) * current_value
            setattr(self.smoothed_params, name, smoothed_value)

        # log data for visualisation
        self.history.log("azimuth", self.smoothed_params.azimuth)
        self.history.log("elevation", self.smoothed_params.elevation)
        self.history.log("diffuseness",
                         self.smoothed_params.diffuseness_metric)

    def process(self) -> NDArray:
        """
        Process the STFT of the SRIR frame by frame.

        Returns
        -------
        NDArray
            Output signal in the time domain of shape (num_loudspeakers, ir_length).
        """
        # initialise output signal
        self.output_signal = np.zeros(
            (self.num_loudspeakers, self.num_freq_bins, self.num_time_frames),
            dtype=np.complex64)
        for k in tqdm(range(len(self.time_frames))):
            # get current frame
            cur_frame = self.srir_stft[..., k]
            # calculate directional and diffuseness parameter for current frame
            cur_params = self.calculate_parameters(cur_frame)
            if k == 0:
                self.smoothed_params = cur_params
            else:
                # smooth the current parameters
                self.smooth_parameters(cur_params)
            # process current frame
            self.output_signal[..., k] = self.process_frame(cur_frame)

        # take inverse STFT of output signals to get one RIR per loudspeaker
        rir_loudspeakers = []
        for i in range(self.num_loudspeakers):
            _, cur_output_channel = istft(
                self.output_signal[i],
                fs=self.sample_rate,
                nperseg=self.win_size,
                noverlap=self.hop_size,
                nfft=self.fft_size,
            )
            rir_loudspeakers.append(cur_output_channel)

        # Stack into array: shape (num_loudspeakers, num_time_samples)
        rir_loudspeakers = np.stack(rir_loudspeakers)
        return rir_loudspeakers

    def save_history(self, save_path: str) -> None:
        """
        Save history to a .mat file.

        Parameters
        ----------
        save_path : str
            Directory to save file in.
        """
        self.history.to_file(f'{save_path}/history.mat')
