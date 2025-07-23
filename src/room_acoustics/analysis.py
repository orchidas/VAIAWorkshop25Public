import numpy as np
from numpy.typing import NDArray
from scipy.signal import spectrogram
from scipy.stats import linregress
from utils import (
    discard_last_n_percent,
    filterbank,
    ms_to_samps,
)


def schroeder_backward_int(
    x: NDArray,
    energy_norm: bool = True,
    subtract_noise: bool = False,
    noise_level: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """
    Compute the backward integration of the squared impulse response (Schroeder integration).

    Parameters
    ----------
    x : NDArray
        Input 1D array (impulse response or energy decay curve).
    energy_norm : bool, optional
        If True, normalize the output to its maximum value (default: True).
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal (default: False).
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True (default: 0.0).

    Returns
    -------
    tuple of NDArray
        Tuple containing the backward integrated and normalized array, and the normalization value(s) used.
    """
    # Flip the input array to prepare for backward integration
    out = np.flip(x, axis=-1)
    # Subtract noise power from the squared signal if requested 
    if subtract_noise:
        out_sqrd = out ** 2 - noise_level ** 2
    else:
        out_sqrd = out ** 2
    # Compute cumulative sum (integration) over the reversed array
    out = np.cumsum(out_sqrd, axis=-1)
    # Flip the result back to original order
    out = np.flip(out, axis=-1)

    # Normalize the energy if requested
    if energy_norm:
        norm_vals = np.max(out, keepdims=True, axis=-1)  # per channel
    else:
        norm_vals = np.ones_like(out)

    return out / norm_vals, norm_vals


def compute_edc(
    x: NDArray,
    use_filterbank: bool = False,
    compensate_fbnk_energy: bool = True,
    n_fractions: int = 1,
    f_min: int = 63,
    f_max: int = 16000,
    fs:int = 48000,
    energy_norm: bool = True,
    subtract_noise: bool = False,
    noise_level: float = 0.0,
) -> NDArray:
    """
    Compute the Energy Decay Curve (EDC) in dB from an input signal.

    Parameters
    ----------
    x : NDArray
        Input 1D array (impulse response or energy decay curve).
    use_filterbank : bool, optional
        If True, apply a fractional octave filterbank to the signal (default: False).
    compensate_fbnk_energy : bool, optional
        If True, compensate for energy loss in filtering (default: True).
    n_fractions : int, optional
        Number of fractions per octave for the filterbank (default: 1).
    f_min : int, optional
        Minimum frequency of the filterbank (default: 63).
    f_max : int, optional
        Maximum frequency of the filterbank (default: 16000).
    fs : int, optional
        Sampling rate of the signal (default: 48000).
    energy_norm : bool, optional
        If True, normalize the output to its maximum value (default: True).
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal (default: False).
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True (default: 0.0).

    Returns
    -------
    NDArray
        The energy decay curve in dB.
    """
    # Remove filtering artifacts (last 5 permille)
    out = discard_last_n_percent(x, 0.5)
    if use_filterbank:
        # Use filterbank to compute EDCs
        out = filterbank(out, n_fractions, f_min=f_min, f_max=f_max, sample_rate=fs, compensate_energy=compensate_fbnk_energy)[0]
    # Compute EDCs using Schroeder backward integration
    out = schroeder_backward_int(out, energy_norm, subtract_noise, noise_level)[0]
    # Convert to dB scale
    out = 10 * np.log10(out)

    return out


def estimate_rt60(
        edc_db: NDArray, 
        time: NDArray, 
        decay_start_db: float = -5, 
        decay_end_db:float = -65
    ) -> tuple[float, float, float, NDArray]:
    """
    Estimate the reverberation time (RT60) from an Energy Decay Curve (EDC) using linear regression.

    Parameters
    ----------
    edc_db : NDArray
        Energy decay curve in dB.
    time : NDArray
        Time vector corresponding to the EDC samples.
    decay_start_db : float, optional
        Starting decay level in dB for the linear fit (default: -5).
    decay_end_db : float, optional
        Ending decay level in dB for the linear fit (default: -65).

    Returns
    -------
    tuple of float, float, float, NDArray
        Tuple containing:
        - rt60 : float
            Estimated RT60 in seconds
        - slope : float
            Slope of the linear fit in dB/s
        - intercept : float
            Y-intercept of the linear fit
        - valid_range : NDArray
            Boolean array indicating the samples used for the fit
    """
    # Select the range of EDC values between decay_start_db and decay_end_db and save it in valid_range
    valid_range = (edc_db < decay_start_db) & (edc_db > decay_end_db)
    # Perform linear regression with scipy.stats's linregress on the selected range to estimate decay slope and intercept
    slope, intercept, *_ = linregress(time[valid_range], edc_db[valid_range])
    # Calculate RT60 as the time required for a 60 dB decay
    rt60 = -60 / slope
    return rt60, slope, intercept, valid_range


def compute_edr(
        x: NDArray,
        energy_norm: bool = True,
        subtract_noise: bool = False,
        noise_level: float = 0.0,
    ) -> NDArray:
    """
    Compute the Energy Decay Relief (EDR) in dB from an input signal using a filterbank.

    Parameters
    ----------
    x : NDArray
        Input 1D array (impulse response or energy decay curve).
    energy_norm : bool, optional
        If True, normalize the output to its maximum value (default: True).
    subtract_noise : bool, optional
        If True, subtract the squared noise level from the squared signal (default: False).
    noise_level : float, optional
        The noise level to subtract if subtract_noise is True (default: 0.0).

    Returns
    -------
    NDArray
        The energy decay relief in dB.
    """
    # Remove filtering artifacts (last 5 permille)
    out = discard_last_n_percent(x, 0.5)
    # Compute the Short-Time Fourier Transform (STFT) magnitude
    _, _, stft_mag = spectrogram(out, nperseg=1028, noverlap=int(1028*0.75), mode='magnitude')
    # Apply Schroeder backward integration to each time-frequency bin
    out = schroeder_backward_int(stft_mag, energy_norm, subtract_noise, noise_level)[0]
    # Convert energy to decibel (dB) scale, adding a small offset to avoid log(0)
    out = 10 * np.log10(out + 1e-32)

    return out


def normalized_echo_density(
        rir: NDArray,
        fs: float,
        window_length_ms: float = 30,
        use_local_avg: bool = True):
    """
    Compute the normalized echo density profile as defined by Abel.

    Parameters
    ----------
    rir : NDArray
        Room impulse response.
    fs : float
        Sampling rate in Hz.
    window_length_ms : float, optional
        Window length in milliseconds (default: 30).
    use_local_avg : bool, optional
        If True, use local average for weighted standard deviation (default: True).

    Returns
    -------
    np.ndarray
        Normalized echo density profile.
    """
    def weighted_std(signal: NDArray, window_func: NDArray, use_local_avg: bool):
        """
        Return the weighted standard deviation of a signal.

        Parameters
        ----------
        signal : NDArray
            Input signal.
        window_func : NDArray
            Window function for weighting.
        use_local_avg : bool
            If True, use local average for variance calculation.

        Returns
        -------
        float
            Weighted standard deviation.
        """
        if use_local_avg:
            average = np.average(signal, weights=window_func)
            variance = np.average((signal - average)**2, weights=window_func)
        else:
            variance = np.average((signal)**2, weights=window_func)
        return np.sqrt(variance)
    # erfc(1/√2)
    ERFC = 0.3173

    window_length_samps = ms_to_samps(window_length_ms, fs)
    # Ensure window length is odd for symmetric windowing
    if not window_length_samps % 2:
        window_length_samps += 1
    half_window = int((window_length_samps - 1) / 2)

    # Pad the RIR to handle windowing at the edges
    padded_rir = np.pad(rir, ((half_window, half_window)))

    # Prepare output array and window function
    output = np.zeros(len(rir) + 2 * half_window)
    window_func = np.hanning(window_length_samps)
    window_func = window_func / sum(window_func)
    # Slide window across RIR and compute normalized echo density
    for cursor in range(len(rir)):
        # Extract the current frame from the padded RIR
        frame = padded_rir[cursor:cursor + window_length_samps]
        # Compute the weighted standard deviation of the frame    
        std = weighted_std(frame, window_func, use_local_avg)
        # Count the number of samples above the weighted standard deviation, weighted by the window function
        count = ((np.abs(frame) > std) * window_func).sum()
        # Normalize the count by the ERFC constant and store it in the output array
        output[cursor] = (1 / ERFC) * count
    # Remove padding to match original RIR length
    ned = output[:-window_length_samps]
    return ned


def rt2slope(rt: float, fs: int) -> float:
    """
    Convert reverberation time (RT60) to slope in dB/s.

    Parameters
    ----------
    rt : float
        Reverberation time in seconds.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    float
        Slope in dB/s.
    """
    if rt <= 0:
        raise ValueError("RT60 must be a positive value.")
    return -60 / rt / fs