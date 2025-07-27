import numpy as np 
from numpy.typing import NDArray, ArrayLike
from typing import Union 


def sine_sweep(fs: int, f1: float = 20, f2: float = None, T: float = 1.0) -> NDArray:
    """
    Generate a logarithmic sine sweep (chirp) signal.
    
    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    f1 : float, optional
        Starting frequency in Hz. If None, defaults to 20 Hz.
    f2 : float, optional
        Ending frequency in Hz. If None, defaults to Nyquist frequency.
    T : float, optional
        Duration of the sweep in seconds. If None, defaults to 1.0 seconds.
    
    Returns
    -------
    NDArray
        Generated logarithmic sine sweep signal.
    
    Notes
    -----
    The logarithmic chirp is generated using the formula:
    s(t) = sin((ω₁ * T / log(ω₂/ω₁)) * (exp((t/T) * log(ω₂/ω₁)) - 1))
    
    This type of sweep provides equal energy per octave, making it useful for
    acoustic measurements and system identification.
    """
    if f2 is None:
        f2 = fs / 2
    omega1 = 2 * np.pi * f1    
    omega2 = 2 * np.pi * f2

    t = np.linspace(0, T, int(fs*T))
    log_ratio = np.log(omega2 / omega1)
    exponential_term = np.exp((t / T) * log_ratio) - 1
    y = np.sin((omega1 * T / log_ratio) * exponential_term)
    return y


def decay_kernel(
    t_values: Union[float, ArrayLike],
    time: ArrayLike,
    fs: float,
    normalize_envelope: bool = False,
    add_noise: bool = False,
) -> NDArray:
    """
    Generate a decay kernel for the exponential envelope. Accepts only one frequency band at a time.

    Parameters
    ----------
    t_values : float or ArrayLike
        The T60 values. Should have shape (B, K) where B is the number of frequency bands and K is the number of RIRs.
    time : ArrayLike
        Time vector of length T.
    fs : float
        Sampling rate.
    normalize_envelope : bool, optional
        Whether to normalize the energy to account for Schroeder integration (default is False).
    add_noise : bool, optional
        Whether to add noise to the decay kernel (use only if modeling a single slope, default is False).

    Returns
    -------
    NDArray
        Exponential decay kernel of shape (B, T, K), or with noise if `add_noise` is True.

    Notes
    -----
    The kernel is computed as exp(-t/tau), where tau is derived from T60.
    If `add_noise` is True, a linearly decaying noise component is concatenated.
    """
    assert len(t_values.shape) <= 2, "t_values should should have max 2 dimensions"
    if len(t_values.shape) == 1:
        t_values = np.expand_dims(t_values, axis=0)

    # calculate the decay time constant tau from T60 - save them in a variable called tau_vals
    tau_vals = np.log(10**6) / t_values

    # calculate the exponential decay kernel
    exponential = np.exp(-np.einsum("bk,t->btk", tau_vals, time))

    # normalise the kernel to have unit energy
    if normalize_envelope:
        exponential = np.einsum("ntb, nb -> ntb", exponential,
                                np.sqrt((1 - np.exp(-2 * tau_vals / fs))))
    # construct the decay kernel
    if add_noise:
        # calculate noise
        ir_len = len(time)

        # generate the kernel for the noise, which should be a linearly decaying signal from ir_len to 0
        noise = np.linspace(ir_len, 0, ir_len) 
        noise = np.expand_dims(noise, axis=(0, -1))
        noise = np.tile(
            noise, (exponential.shape[0], 1, 1))  # repeat noise along all rirs
        
        # concatenate it to the exponential decay kernels
        exponential = np.concatenate((exponential, noise), axis=-1)

    return exponential

