"""
H1mpulResper - Calculate the Impulse Response from Raw Measurement Data
Description: Produces the time domain impulse response by calculating the H1 
    estimator using the raw measurement data, plus correction factors. The 
    resulting impulse response has units of Volts measured/Volts stimulus.
    
    The estimated time domain response, and the coherence function versus
    frequency, are plotted on the screen.

Author: Charlie Laub
Date: October 2026
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.fft
from scipy import signal


def read_wav_file(file_path):
    """Reads a WAV file and returns sample rate and data as float64."""
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data.astype(np.float64)


def write_wav_file(file_path, sample_rate, data_float32):
    """Writes time-domain data as FLOAT32 WAV file without normalization."""
    wavfile.write(file_path, sample_rate, data_float32)
    print(f"The calculated impulse response was saved to '{file_path}' at {sample_rate} Hz in FLOAT32 format.\n")


def calculate_stimulus_Vrms(x, v_max, stimulus_gain):
    """
    Calculates the RMS voltage of the middle 50% of the signal array x,
    excluding the tapered ends.
    """
    n = len(x)
    
    # Define the boundaries for the middle 50% (discarding outer 25% on each end)
    start_idx = int(0.25 * n)
    end_idx = int(0.75 * n)
    
    # Extract the central region
    x_core = x[start_idx:end_idx]
    
    # Calculate the Root Mean Square (RMS) voltage at audio interface output
    rms_voltage = v_max * np.sqrt(np.mean(x_core**2))
    
    # Scale by stimulus gain to get voltage at loudspeaker terminals
    if stimulus_gain is not None:
        rms_voltage *= stimulus_gain

    # Print the result
    print(f"The stimulus voltage applied to the loudspeaker was {rms_voltage:.4f} Vrms.\n")
    
    #return rms_voltage


def compute_h1_time_domain(measurement, stimulus):
    len_m = len(measurement)
    len_s = len(stimulus)
    
    # Proper FFT length for linear convolution / deconvolution
    native_n = len_m + len_s - 1
    n = scipy.fft.next_fast_len(native_n)

    # Frequency domain calculations
    X = np.fft.rfft(stimulus, n=n)
    Y = np.fft.rfft(measurement, n=n)

    # Cross-spectral density and Auto-spectral density
    S_xy = np.conj(X) * Y
    S_xx = np.abs(X) ** 2

    # Regularization
    eps = 1e-12 * np.max(S_xx) if np.max(S_xx) > 0 else 1e-12
    H1_freq = S_xy / (S_xx + eps)

    # Convert back to time-domain impulse response
    h1_time = np.fft.irfft(H1_freq, n=n)

    # Optional: Trim back to expected impulse response length if needed
    return h1_time[:len_m].astype(np.float32)


def plot_results(h1_time, sample_rate, C_f, Cxy):
    """
    Plots the H1 time-domain impulse response (upper plot) 
    and the magnitude squared coherence function (lower plot).
    """
    # 1. Compute the time axis for the impulse response
    n = len(h1_time)
    time_axis = np.arange(n) / sample_rate

    # 2. Set up the figure with 2 vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle('System Identification Analysis', fontsize=14, fontweight='bold')

    # Upper Plot: H1 Estimated Impulse Response (Time Domain)
    ax1.plot(time_axis, h1_time, label="$h_1(t)$ Impulse Response", color="navy")
    ax1.set_title("H1 Estimator - Time Domain Response")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Lower Plot: Coherence Function (Frequency Domain)
    ax2.semilogx(C_f, Cxy, label="Magnitude Squared Coherence", color="darkred", linewidth=1.5)
    ax2.set_title("Coherence Function")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Coherence Gamma")
    ax2.set_ylim(-0.05, 1.05)  # Strict mathematical bounds for coherence
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # Adjust layout to prevent overlapping text and render
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate time-domain H1 estimator from acoustic measurement and stimulus WAV files."
    )
    parser.add_argument(
        "-m", "--mic_file",
        type=str,
        required=True,
        help="Path to the WAV file containing the microphone signal."
    )
    parser.add_argument(
        "-s", "--stimulus_file",
        type=str,
        required=True,
        help="Path to the WAV file containing the stimulus signal."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="impulse_H1.wav",
        help="Filepath where the H1 estimator WAV file will be saved (default: impulse_H1.wav)."
    )
    parser.add_argument(
        "--vmax_mic_input", 
        type=float, 
        required=True,
        help="RMS voltage that produces 0 dBFS on the input channel where the microphone signal was recorded"
    )
    parser.add_argument(
        "--vmax_stimulus_input", 
        type=float, 
        required=True,
        help="RMS voltage that produces 0 dBFS on the input channel where the stimulus was recorded"
    )
    parser.add_argument("--stimulus_gain", 
        type=float, 
        required=True,
        help="Stimulus gain in units of Volts applied to loudspeaker per Volt at the audio interface input."
    )
    args = parser.parse_args()

    # Read input WAV files
    fs_measurement, y = read_wav_file(args.mic_file)
    fs_stimulus, x = read_wav_file(args.stimulus_file)

    if fs_measurement != fs_stimulus:
        raise ValueError(f"Sampling rates do not match: {fs_measurement} Hz vs {fs_stimulus} Hz")

    print(f"\n\nNow processing measurement data to produce the impulse response...\n\n")

    # Calculate and report the RMS Voltage of the stimulus signal
    calculate_stimulus_Vrms( x, args.vmax_stimulus_input, args.stimulus_gain )

    # Compute H1 purely in the time domain (inputs/outputs are time-domain)
    h1_time = compute_h1_time_domain(measurement=y, stimulus=x)

    # correct H1 by the ratio of the mic channel and reference channel voltage maximxums
    # this converts the units into Volts measured by the Mic per Volt of stimulus
    h1_time *= args.vmax_mic_input / args.vmax_stimulus_input
    print(f"The H1 estimator produces data with the following units:\n   Volts produced by the microphone per Volt stimulus at the loudspeaker.\n")

    # Write H1 to file
    write_wav_file(args.output, fs_measurement, h1_time)

    # generate and save the coherence function
    #   x: stimulus, y: response, fs: sampling frequency
    # Calculate an optimal nperseg for Welch's method based on signal length.
    #   With 50% overlap, number of segments is roughly (2 * length) / nperseg
    target_segments = 10
    ideal_nperseg = (2 * len(y)) / target_segments
    # Find the nearest power of 2 for FFT efficiency
    power_of_2 = 2 ** np.round(np.log2(ideal_nperseg))
    # Clamp to practical bounds (e.g., min 2048 for resolution, max 32768 to ensure enough averaging)
    nperseg_value = int(np.clip(power_of_2, 2048, 32768))
    C_f, Cxy = signal.coherence(x, y, fs=fs_measurement, nperseg=nperseg_value)

    # Plot time-domain H1 estimator and the coherence function
    plot_results(h1_time=h1_time, sample_rate=fs_measurement, C_f=C_f, Cxy=Cxy)


if __name__ == "__main__":
    main()