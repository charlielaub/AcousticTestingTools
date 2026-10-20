"""
H1mpulResper - Calculate the Impulse Response from Raw Measurement Data
Description: Produces the time domain impulse response by calculating the H1 
    estimator using the raw measurement data, plus correction factors. The 
    resulting impulse response has units of Volts measured/Volts stimulus.

Author: Charlie Laub
Date: October 2026
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


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
    """
    Computes the time-domain H1 impulse response.
    All frequency-domain operations and spectral variables are kept internal.
    """
    # Ensure equal length by padding with zeros
    n = max(len(measurement), len(stimulus))

    # Frequency domain calculations (internal to compute_h1_time_domain)
    X = np.fft.rfft(stimulus, n=n)
    Y = np.fft.rfft(measurement, n=n)

    # Cross-spectral density (S_xy) and Auto-spectral density (S_xx)
    S_xy = np.conj(X) * Y
    S_xx = np.abs(X) ** 2

    # Regularization to prevent division by zero
    eps = 1e-12 * np.max(S_xx) if np.max(S_xx) > 0 else 1e-12
    H1_freq = S_xy / (S_xx + eps)

    # Convert back to time-domain impulse response h1(t)
    h1_time = np.fft.irfft(H1_freq, n=n)

    return h1_time.astype(np.float32)


def plot_h1_response(h1_time, sample_rate):
    """Plots the time-domain impulse response."""
    n = len(h1_time)
    time_axis = np.arange(n) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, h1_time, label="$h_1(t)$ Impulse Response", color="navy")
    plt.title("H1 Estimator - Time Domain Response")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
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
        default="h1_output.wav",
        help="Filepath where the H1 estimator WAV file will be saved (default: h1_output.wav)."
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
        default=None,
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

    # Write output file in main
    write_wav_file(args.output, fs_measurement, h1_time)

    # Plot time-domain result
    plot_h1_response(h1_time, fs_measurement)


if __name__ == "__main__":
    main()