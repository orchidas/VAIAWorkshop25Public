
import matplotlib.pyplot as plt
import numpy as np


def plot_time_domain(x, fs, title="Time Domain Signal", xlabel="Time (s)", ylabel="Amplitude"):
    """
    Plot the time-domain signal.

    Parameters
    ----------
    x : array_like
        The signal to plot.
    fs : int or float
        The sampling frequency of the signal in Hz.
    title : str, optional
        The title of the plot (default is "Time Domain Signal").
    xlabel : str, optional
        The label for the x-axis (default is "Time (s)").
    ylabel : str, optional
        The label for the y-axis (default is "Amplitude").

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    time = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, time[-1])
    plt.grid()
    plt.show()


def plot_spectrogram(x, fs: int, title="Spectrogram", n_fft=1024, hop_length=None, clim = [-100, 0], cmap="viridis"):
    """
    Plot the spectrogram of a signal.

    Parameters
    ----------
    x : array_like
        The input signal.
    fs : int or float
        The sampling frequency of the signal in Hz.
    title : str, optional
        The title of the plot (default is "Spectrogram").
    n_fft : int, optional
        Number of FFT points (default: 1024).
    hop_length : int, optional
        Number of samples between successive frames (default: n_fft // 4).
    clim : list of float, optional
        Color limits for the spectrogram in dB (default: [-100, 0]).
    cmap : str, optional
        Colormap for the spectrogram (default: "viridis").

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    plt.figure(figsize=(10, 4))
    S, freqs, bins, im = plt.specgram(x, NFFT=n_fft, Fs=fs, noverlap=n_fft - hop_length, cmap=cmap)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(im).set_label('Intensity [dB]')
    plt.clim(clim[0], clim[1])
    plt.tight_layout()
    plt.show()
