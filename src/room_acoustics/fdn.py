import random
import torch
import numpy as np 
import scipy as sp

from utils import db2lin

from typing import List
from collections import OrderedDict
from numpy.typing import NDArray
from room_acoustics.analysis import rt2slope

# FLAMO imports
from flamo.processor import dsp, system

class FeedbackDelayNetwork:
    def __init__(self, sample_rate: int, delay_lengths: List, feedback_matrix_type: str, t60: float = 1.0):
        """
        Initialize the Feedback Delay Network with delay times and feedback matrix.

        Parameters
        ----------
        sample_rate : int
            The sample rate (in Hz).
        delay_lengths : list of int
            List of delay lengths in samples.
        feedback_matrix_type : str
            The type of feedback matrix to use. Options: 'identity', 'orthogonal', 'Hadamard', 'Householder', 'circulant'.
        t60 : float, optional
            Reverberation time in seconds (default is 1.0).
        """
        self.sample_rate = sample_rate
        self.N = len(delay_lengths)
        self.delay_lengths = delay_lengths
        self.t60 = t60
        self.feedback_matrix = self.get_feedback_matrix(feedback_matrix_type)
        self.input_gains = np.random.randn(1, self.N)  # Default input gains for each delay line
        self.output_gains = np.random.randn(self.N, 1) 
       
        # initialize the buffers
        self.delay_buffers = [np.zeros(length) for length in delay_lengths]
        self.write_indices = [0] * self.N
        self.output = np.zeros(0)

    def get_feedback_matrix(self, feedback_matrix_type: str) -> NDArray:
        """
        Generate the feedback matrix based on the specified type.

        Parameters
        ----------
        feedback_matrix_type : str
            The type of feedback matrix to generate.

        Returns
        -------
        NDArray
            The generated feedback matrix.
        """
        # convert to all lower case
        feedback_matrix_type = feedback_matrix_type.lower()

        if feedback_matrix_type == 'identity':
            Q =  np.eye(self.N)
        elif feedback_matrix_type == 'random':
            # this is one way to generate a random orthogonal matrix based on QR decomposition
            A = np.random.randn(self.N, self.N)
            Q, R = np.linalg.qr(A)
            Q = np.matmul(Q, np.diag(np.sign(np.diag(R)))) 
        elif feedback_matrix_type == 'hadamard':
            Q = sp.linalg.hadamard(self.N) / np.sqrt(self.N)  # Normalize Hadamard matrix
        elif feedback_matrix_type == 'householder':
            u = np.random.randn(self.N, 1)
            u = u / np.linalg.norm(u)
            Q = np.eye(self.N) - 2 * np.multiply(u, u.T) 
        elif feedback_matrix_type == 'circulant':
            v = np.random.randn(self.N)
            R = np.fft.fft(v)
            R = R / np.abs(R)
            r = np.fft.ifft(R).reshape(-1, 1)  # Ensure r is a column vector
            rnd_sign = 1 if random.random() < 0.5 else -1

            if rnd_sign == 1:
                r2 = np.roll(np.flip(r), 1)
                Q = sp.linalg.toeplitz(r2, r)
            elif rnd_sign == -1:
                r2 = np.roll(r, 1)
                C = sp.linalg.toeplitz(r2, np.flip(r))
                Q = np.fliplr(C)
            else:
                raise ValueError('Not defined')
        else:
            raise ValueError("Invalid feedback matrix type specified.")
        
        self.feedback_matrix = Q

        # apply attenuation 
        gamma = np.power(db2lin(rt2slope(self.t60, self.sample_rate)), np.array(self.delay_lengths))

        Gamma = np.diag(gamma)
        return np.matmul(Gamma, self.feedback_matrix)
    
    def process(self, input_signal: NDArray) -> NDArray:    
        """
        Process the input signal through the Feedback Delay Network.

        Parameters
        ----------
        input_signal : NDArray
            The input audio signal (1D array).

        Returns
        -------
        NDArray
            The processed output signal after passing through the FDN.
        """
        output_signal = []
        
        # process each sample individually
        for sample in input_signal:
            # read output from the delay lines
            feedback_input = np.array([
                self.delay_buffers[i][self.write_indices[i]] for i in range(self.N)
            ])
            # compute the new input ´delay_input´ to the delay lines 
            delay_input = self.input_gains * sample + np.matmul(self.feedback_matrix, feedback_input)
            
            for i in range(self.N):
                # store ´delay_input´ in the delay buffers
                self.delay_buffers[i][self.write_indices[i]] = delay_input[0, i]
                # update the write index for each delay line
                self.write_indices[i] = (self.write_indices[i] + 1) % len(self.delay_buffers[i])
                
            # compute the output sample by multiplying the feedback input with the output gains
            # you can the "append" method to store the output samples
            output_signal.append(np.matmul(feedback_input, self.output_gains).item())

        self.output = np.array(output_signal)
        return self.output
    

class DifferentiableFeedbackDelayNetwork:
    r"""
    Class for creating a differentiable Feedback Delay Network (FDN) model.
    """
    def __init__(self, delay_lengths, fs, nfft, onset_time=0.01, alias_decay_db=5):

        self.delay_lengths = torch.tensor(delay_lengths)
        self.N = len(delay_lengths)
        self.fs = fs
        self.nfft = nfft
        self.onset_time = onset_time  # onset delay in seconds
        self.alias_decay_db = alias_decay_db 
        self.set_fdn()
        self.input_layer = dsp.FFTAntiAlias(self.nfft, alias_decay_db=alias_decay_db) 
        self.output_layer = dsp.iFFTAntiAlias(nfft=nfft, alias_decay_db=alias_decay_db)
        self.model = self.get_shell()

    def set_fdn(self):
        ### WRITE YOUR CODE HERE ### 
        # hereunder you have a list of the modules you should use to build the FDN
        # complete the code by filling the parameters of each module

        direct_gain = dsp.Gain(

        )
        
        onset_delay = dsp.Delay(

        )
        # Input gain
        input_gain = dsp.Gain(

        )
        # Output gain
        output_gain = dsp.Gain(

        )

        # Feedback path with orthogonal matrix
        mixing_matrix = dsp.Matrix(
            
        )

        # (NON LEARNABLE) Parallel delay lines
        delays = dsp.parallelDelay(
            size=(self.N,),
            max_len=self.delay_lengths.max(),
            nfft=self.nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db, 
        )
        delays.assign_value(delays.sample2s(self.delay_lengths))

        # Attenuation (GEQ) module
        attenuation = dsp.parallelGEQ(
            size=(self.N,),
            octave_interval=1,
            nfft=self.nfft,
            fs=self.fs,
            requires_grad=True,
            alias_decay_db=self.alias_decay_db,
        )
        # We need to map the attenuation to a decibel scale
        attenuation.map = lambda x: 20 * torch.log10(torch.sigmoid(x))

        # Create the feedback loop
        feedback = system.Series(
            OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
        )
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        # Create the upper branch of the FDN
        branchA = system.Series(
            OrderedDict(
                {
                    "input_gain": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gain": output_gain,
                }
            )
        )

        # Create the lower branch of the FDN with the direct path
        branchB= system.Series(
            OrderedDict(
                {
                    "onset_delay": onset_delay,
                    "direct_gain": direct_gain,
                }
            )
        )

        self.fdn = system.Parallel(brA=branchA, brB=branchB)
        
    def get_shell(self):
        return system.Shell(
            core=self.fdn, input_layer=self.input_layer, output_layer=self.output_layer
        )
    def normalize_late_energy(
        self,
        target_energy=1,
    ):
        """
        Energy normalization done in the frequency domain.

        Note
        ----
        The energy computed from the frequency response is not the same as the energy of the impulse response.
        Read more at https://pytorch.org/docs/stable/generated/torch.fft.rfft.html

        Parameters
        ----------
        target_energy : float, optional
            Target energy for normalization (default is 1).

        Returns
        -------
        float
            The energy of the FDN after normalization.
        """

        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))

        # apply energy normalization on input and output gains only
        with torch.no_grad():
            core = self.model.get_core()
            core.branchA.input_gain.assign_value(
                torch.div(
                    core.branchA.input_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            core.branchA.output_gain.assign_value(
                torch.div(
                    core.branchA.output_gain.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            self.model.set_core(core)

        # recompute the energy of the FDN
        H = self.model.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))
        return energy_H
