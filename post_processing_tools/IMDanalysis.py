import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d

def extract_multitone_metrics(stim_file, mic_file, output_base, smooth_sigma=8):
    """
    Reads the cyclic IMD stimulus and measurement WAV files, extracts the 
    fundamental tones and the distortion hash, applies smoothing, and exports 
    the results to a high-res plot and CSV file.
    """
    print(f"Loading Stimulus: {stim_file}")
    print(f"Loading Measurement: {mic_file}")
    
    # 1. Read the audio files
    fs_stim, stim_data = wavfile.read(stim_file)
    fs_mic, mic_data = wavfile.read(mic_file)
    
    if fs_stim != fs_mic:
        raise ValueError("Sample rates of stimulus and measurement do not match.")
    if len(stim_data) != len(mic_data):
        raise ValueError("Buffer lengths of stimulus and measurement do not match.")
        
    N = len(stim_data)
    fs = fs_stim
    
    # 2. Strict Rectangular FFT (No windowing for cyclic signals)
    fft_stim = np.fft.rfft(stim_data)
    fft_mic = np.fft.rfft(mic_data)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    
    # Standard amplitude scaling for real FFT
    scale_factor = 2.0 / N
    
    # Convert to dBFS (adding 1e-12 to prevent log10(0) domain errors)
    mag_stim_db = 20 * np.log10(np.abs(fft_stim) * scale_factor + 1e-12)
    mag_mic_db = 20 * np.log10(np.abs(fft_mic) * scale_factor + 1e-12)
    
    # 3. Masking: Identify the precise integer bins containing fundamental tones
    # We look for any bin in the ideal stimulus that is within 50 dB of the peak
    peak_stim_db = np.max(mag_stim_db)
    active_bins = np.where(mag_stim_db > (peak_stim_db - 50.0))[0]
    
    # Ensure DC (0 Hz) is ignored
    active_bins = active_bins[active_bins > 0]
    
    # Create the inverted mask for the empty (Distortion/Noise) bins
    empty_mask = np.ones(len(freqs), dtype=bool)
    empty_mask[active_bins] = False
    empty_mask[0] = False # Ignore DC
    
    # 4. Segregate the Data
    fund_freqs = freqs[active_bins]
    fund_mags = mag_mic_db[active_bins]
    
    empty_freqs = freqs[empty_mask]
    empty_mags = mag_mic_db[empty_mask]
    
    print(f"Detected {len(fund_freqs)} fundamental tones.")
    
    # 5. Envelope Smoothing (Fractional-Octave Approximation)
    # Generate a fixed logarithmic frequency grid (100 points per octave) for standard plotting
    f_out = np.logspace(np.log10(20), np.log10(min(20000, fs/2)), 1000)
    
    # Interpolate the discrete fundamental tones to form a continuous frequency response curve
    fund_interp = np.interp(f_out, fund_freqs, fund_mags)
    # A light smoothing on the fundamental curve to make it look like a standard FR
    fund_env = gaussian_filter1d(fund_interp, sigma=2) 
    
    # Interpolate the dense distortion hash onto the log grid
    td_interp = np.interp(f_out, empty_freqs, empty_mags)
    # Apply a heavier Gaussian filter in the log-frequency domain to create the TD envelope
    td_env = gaussian_filter1d(td_interp, sigma=smooth_sigma)
    
    # 6. Calculate Signal-to-Distortion Ratio (SDR)
    sdr = fund_env - td_env
    
    # 7. Export CSV Data (for Excel, REW, VACS)
    csv_file = f"{output_base}_processed.csv"
    df = pd.DataFrame({
        'Frequency_Hz': f_out,
        'Fundamental_Response_dBFS': fund_env,
        'Total_Distortion_dBFS': td_env,
        'Signal_to_Distortion_Ratio_dB': sdr
    })
    df.to_csv(csv_file, index=False, float_format='%.3f')
    print(f"Exported continuous curve data to: {csv_file}")
    
    # 8. Plotting with Commercial Aesthetics
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot the smoothed Total Distortion (TD) envelope and fill beneath it
    ax1.plot(f_out, td_env, color='crimson', linewidth=2, label='TD (Total Distortion) Envelope')
    ax1.fill_between(f_out, -140, td_env, color='crimson', alpha=0.15)
    
    # Plot the smoothed Fundamental Response envelope
    ax1.plot(f_out, fund_env, color='navy', linewidth=2, label='Fundamental Response')
    
    # Overlay the discrete fundamental tones as "Pins"
    ax1.scatter(fund_freqs, fund_mags, color='deepskyblue', edgecolors='navy', 
                s=40, zorder=5, label='Discrete Fundamental Tones')
    
    # Formatting the primary Y-axis (Magnitude)
    ax1.set_xscale('log')
    ax1.set_xlim(20, 20000)
    ax1.set_ylim(np.min(td_env) - 10, np.max(fund_mags) + 15)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Magnitude (dBFS)', fontsize=12, fontweight='bold')
    ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.6, alpha=0.7)
    ax1.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Add a secondary Y-axis strictly for the SDR curve
    ax2 = ax1.twinx()
    ax2.plot(f_out, sdr, color='forestgreen', linewidth=1.5, linestyle='--', label='Signal-to-Distortion Ratio (SDR)')
    ax2.set_ylabel('SDR / Dynamic Range (dB)', color='forestgreen', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='forestgreen')
    ax2.set_ylim(0, max(100, np.max(sdr) + 10))
    
    # Consolidate legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', ncol=2, fontsize=10, 
               bbox_to_anchor=(0.5, -0.2), frameon=True, shadow=True)
    
    plt.title('Multi-Tone Intermodulation Distortion (IMD) Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 9. High-Resolution Print Export
    png_file = f"{output_base}_IMD_Plot.png"
    plt.savefig(png_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Exported 600 DPI plot to: {png_file}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Multi-Tone IMD Measurement Data")
    parser.add_argument("--stim", type=str, required=True, help="Path to the Ideal Stimulus WAV file")
    parser.add_argument("--mic", type=str, required=True, help="Path to the TSA Measurement WAV file")
    parser.add_argument("--out", type=str, required=True, help="Base name for the output files (CSV and PNG)")
    parser.add_argument("--smooth", type=float, default=8.0, help="Smoothing intensity for the distortion hash envelope (default: 8.0)")
    
    args = parser.parse_args()
    
    extract_multitone_metrics(args.stim, args.mic, args.out, smooth_sigma=args.smooth)