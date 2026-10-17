"""
ImpAlign - Impulse Response Time Alignment
Description: This program aligns impulse responses collected using with the 
    Measure, Move-Move, Measure technique.   

Author: Charlie Laub
Date: October 2026
"""

import numpy as np                    #the numpy module must be installed
import matplotlib.pyplot as plt       #the matplotlib module must be installed
import scipy.io.wavfile as wav        #the scipy module must be installed
from scipy.signal import correlate
import argparse, textwrap

def read_wav_file(file_path):
    """
    Reads a WAV file, forces it to floating-point representation,
    and converts multi-channel audio to mono by selecting the first channel.
    """
    fs, data = wav.read(file_path)
    
    # Handle multi-channel (stereo) files by taking the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
        
    # Convert integer types to float64 for high-precision math
    if data.dtype.kind != 'f':
        data = data.astype(np.float64)
        
    return fs, data

def calculate_subsample_offset(data_master, data_slave, fs, max_window_us=500.0):
    """
    Computes cross-correlation restricted to a specific time window,
    protects boundaries, and applies a quadratic fit to find sub-sample alignment.
    """
    # Calculate max lag samples based on sampling rate
    max_lag_samples = int(np.ceil(max_window_us * 1e-6 * fs))
    
    # Compute full raw cross-correlation
    corr = correlate(data_master, data_slave, mode='full')
    lags = np.arange(-len(data_slave) + 1, len(data_master))
    
    # Mask out indices outside our target window
    valid_window_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_window = corr[valid_window_mask]
    lags_window = lags[valid_window_mask]
    
    # Find peak index relative to the restricted window slice
    peak_idx_in_window = np.argmax(np.abs(corr_window))
    coarse_lag = lags_window[peak_idx_in_window]
    
    # Boundary Protection: Safely extract neighbor indices from global array
    global_peak_idx = int(np.where(lags == coarse_lag)[0][0])
    
    # Extract three points surrounding the peak for quadratic interpolation
    y1 = corr[global_peak_idx - 1]
    y2 = corr[global_peak_idx]
    y3 = corr[global_peak_idx + 1]
    
    # Vertex interpolation formula
    fine_offset = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
    exact_lag_samples = coarse_lag + fine_offset
    
    # Return metrics along with correlation data needed for visualization
    return exact_lag_samples, coarse_lag, fine_offset, max_lag_samples, lags, corr, global_peak_idx

def apply_fourier_shift(data_slave, exact_lag_samples):
    """
    Applies a precise sub-sample time shift in the frequency domain.
    """
    N = len(data_slave)
    freqs = np.fft.fftfreq(N)
    slave_fft = np.fft.fft(data_slave)
    
    # Phase rotation matrix based on Fourier Shift Theorem
    shift_vector = np.exp(-2j * np.pi * freqs * exact_lag_samples)
    corrected_data = np.fft.ifft(slave_fft * shift_vector).real
    return corrected_data

def plot_parabolic_fit(lags, corr, global_peak_idx, coarse_lag, fine_offset):
    """
    Generates a matplotlib figure visualizing the true correlation points 
    against the fitted parabolic curve down to sub-sample resolution.
    """
    # Extract the 3 points used for fitting
    idx_points = [global_peak_idx - 1, global_peak_idx, global_peak_idx + 1]
    x_points = lags[idx_points]
    y_points = corr[idx_points]
    
    # Reconstruct the parabola: y = ax^2 + bx + c relative to the coarse peak index
    # Shifting origin to 'coarse_lag' maps points to x = [-1, 0, 1]
    y1, y2, y3 = y_points
    a = 0.5 * (y1 - 2 * y2 + y3)
    b = 0.5 * (y3 - y1)
    c = y2
    
    # Create high-resolution evaluation points for plotting the curve smooth
    x_fit_rel = np.linspace(-1.5, 1.5, 200)
    y_fit = a * (x_fit_rel**2) + b * x_fit_rel + c
    x_fit_abs = x_fit_rel + coarse_lag
    
    exact_lag = coarse_lag + fine_offset
    peak_value = a * (fine_offset**2) + b * fine_offset + c
    
    # Update global config to specify 600 DPI instead of default
    plt.rcParams['savefig.dpi'] = 600 

    # 1. Base figure creation
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(x_fit_abs, y_fit, 'g--', label='Quadratic Fit (Parabola)')
    plt.scatter(x_points, y_points, color='red', zorder=5, label='Discrete Sample Lags')
    plt.axvline(x=coarse_lag, color='gray', linestyle=':', label=f'Coarse Peak ({coarse_lag})')
    plt.scatter([exact_lag], [peak_value], color='blue', marker='*', s=150, zorder=6, 
                label=f'Sub-sample Peak ({exact_lag:.4f})')

    plt.title('Cross-Correlation Peak & Parabolic Interpolation')
    plt.xlabel('Lag (Samples)')
    plt.ylabel('Correlation Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    # 2. FIX: Restrict tight_layout so it leaves a blank padding region at the bottom
    # rect format is [left, bottom, right, top] from 0 to 1. 
    #plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(
        left=0.15,    # Padding on the left edge
        right=0.93,   # Padding on the right edge
        top=0.9,     # Padding at the top (leaves room for title)
        bottom=0.21   # Extra deep padding at the bottom (leaves room for labels & text note)
    )

    # 3. Inject the text safely into that bottom padding region
    plt.figtext(
        0.5, 0.05,  # 50% across horizontal window, 3% up from the absolute bottom boundary
        f"To close the window, click on the kill button |X| located in the top right corner.",
        ha="center", 
        fontsize=12, 
        color="blue" 
    )

    plt.show()

def preserve_and_save_wav(output_path, fs, original_dtype, processed_data):
    """
    Dynamically checks the original data type and converts the floating-point 
    processed data back safely to prevent empty or silent outputs.
    """
    if np.issubdtype(original_dtype, np.integer):
        # Original was integer (e.g., 16-bit PCM)
        # Find the type boundaries (e.g., -32768 to 32767 for int16)
        info = np.iinfo(original_dtype)
        final_signal = np.clip(processed_data, info.min, info.max).astype(original_dtype)
    else:
        # Original was float (e.g., 32-bit float ranging from -1.0 to 1.0)
        final_signal = np.clip(processed_data, -1.0, 1.0).astype(original_dtype)
        
    wav.write(output_path, fs, final_signal)


def main():
    parser = argparse.ArgumentParser(
        description="""
Description:
    This program works with impulse response data generated with the Measure, Move-Move, Measure technique.
    Impulse data from measurement 1 should be supplied in infile1, and from measurement 2 in infile2. 
    The program calculates the Cross-Correlation between the infiles and finds the most likely time offset.
    It then optimizes the offset to sub-sample resolution, and recalculates the impulse response supplied in
    infile2 including the offset. The corrected data from infile2 is saved to the outfile. A plot is presented
    showing the optimized time offset. Both infiles must be WAV files with data in the same format (e.g. 
    FLOAT32). This format will be use when saving the outfile.""",
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # File arguments
    parser.add_argument(
        "--infile1", "--f1", type=str, required=True, help="Path and Filename (*.wav) for the Measurement 1 impulse."
    )
    parser.add_argument(
        "--infile2", "--f2", type=str, required=True, help="Path and Filename (*.wav) for the Measurement 2 impulse."
    )
    parser.add_argument(
        "--outfile", "--of", type=str, required=True, help="Path and Filename (*.wav) for the aligned file2 data."
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # Define input and output files
    master_file = args.infile1
    slave_file = args.infile2
    output_file = args.outfile
    
    try:
        print("Reading WAV files...")
        # We need to peek at the original dtype to handle output scaling correctly
        _, raw_slave_data = wav.read(slave_file)
        original_dtype = raw_slave_data.dtype
        
        fs_master, data_master = read_wav_file(master_file)
        fs_slave, data_slave = read_wav_file(slave_file)
        
        if fs_master != fs_slave:
            raise ValueError(f"Sampling rate mismatch: Master={fs_master}Hz, Slave={fs_slave}Hz.")
        
        print(f"Analyzing alignment (Search window restricted to +/-500 microseconds)...")
        exact_lag, coarse_lag, fine_offset, max_lag, lags, corr, peak_idx = \
            calculate_subsample_offset(data_master, data_slave, fs_master, max_window_us=500.0)
            
        time_offset_us = (exact_lag / fs_master) * 1e6
        print(f"\n--- Alignment Analysis ---")
        print(f"Original Data Type : {original_dtype}")
        print(f"Search Constraints : +/-{max_lag} samples")
        print(f"Coarse Sample Lag  : {coarse_lag} samples")
        print(f"Sub-sample Delta   : {fine_offset:+.4f} samples")
        print(f"Calculated Offset  : {time_offset_us:.3f} microseconds")
        
        print("\nCorrecting alignment via Fourier phase shift...")
        corrected_data = apply_fourier_shift(data_slave, exact_lag)
        
        print(f"Writing corrected file to '{output_file}'...")
        # Call the new dynamic type-safe saver
        preserve_and_save_wav(output_file, fs_master, original_dtype, corrected_data)
        print("File operation successful.")
        
        #plot the data unless --no-plot was specified
        if not args.no_plot:
            print("\nRendering sub-sample fit visualization...")
            plot_parabolic_fit(lags, corr, peak_idx, coarse_lag, fine_offset)
        
    except FileNotFoundError as e:
        print(f"Error: Could not locate file. Ensure your paths are correct. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")


if __name__ == '__main__':
    main()


