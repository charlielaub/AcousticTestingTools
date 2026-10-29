"""
Time2Freq - Convert the Time Domain Response to the Frequency Domain Response 
Description: Calculates the time domain frequency response and the harmonic 
    levels using the Farina method, plus correction factors. Can produce 
    relative SPL levels, or the SPL level in units of dB re 20 uPa @ 2.83V/1m. 
    Both the regular phase and minimum phase responses are made available.
    The H1 estimated impulse response is used to produce correct phase data.

Author: Charlie Laub
Date: October 2026
"""


    
import argparse
import os
import sys
import numpy as np
import scipy.signal
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
import math
import struct
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

def parse_arguments():
    """
    Parses command-line arguments for input WAV file, base filename for outputs,
    optional microphone calibration, sensitivity parameters, and amplifier gain.
    """
    parser = argparse.ArgumentParser(description="An interactive program to process loudspeaker measurements collected using the program DaDooRunRun. Exports frequency domain response data and plots.")
    parser.add_argument("-m","--measurement_datafile", 
            type=str,
            required=True,
            help="Path to the WAV file containing the microphone measurement data."
    )
    parser.add_argument("-i","--impulse_datafile", 
            type=str,
            required=True,
            help="Path to the WAV file containing the impulse response data."
    )
    parser.add_argument("-f","--farina_filterfile", 
            type=str,
            required=True,
            help="Path to the WAV file containing the Farina inverse filter."
    )
    parser.add_argument("-b", "--basename", 
            type=str, 
            default="response",
            help="Base file name for output files (default: response)."
    )
    parser.add_argument("-c","--mic_cal_file", 
            type=str, 
            default=None,
            help="Path to microphone calibration file (*.mic or *.cal)."
    )
    parser.add_argument("-k","--spl_correction_factor", 
            type=float, 
            default=None,
            help="Lumped SPL correction factor."
    )
    return parser.parse_args()




def compute_frequency_response(
    audio_data,
    sample_rate,
    gate_end_idx,
    tukey_alpha
):
    """
    Applies a Tukey window, computes a standardized normalized FFT, and returns
    frequencies, calibrated magnitude (dBSPL), phase vectors, and delta frequency.
    """
    gate_end_idx = max(10, min(gate_end_idx, len(audio_data)))

    # Apply Tukey window locally to the active segment
    win = scipy.signal.get_window(('tukey', tukey_alpha), gate_end_idx)
    windowed_data = audio_data[:gate_end_idx].astype(np.float32) * win

    # Determine optimal power-of-2 FFT length
    desired_len = max(2048, 2 * len(windowed_data))
    n_fft = 1 << math.ceil(math.log2(desired_len))

    # Standard real FFT execution
    fft_result = np.fft.rfft(windowed_data, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    mag = np.abs(fft_result)

    # Coherent gain normalization for the windowed signal
    window_sum = np.sum(win)
    if window_sum > 0:
        # Correct scaling for real FFT (rfft) preserving physical peak amplitude
        mag_normalized = mag / (window_sum / 2.0)
    else:
        mag_normalized = mag

    mag_db = 20.0 * np.log10(np.maximum(mag_normalized, 1e-12))
    
    phase_deg = np.angle(fft_result, deg=True)

    # Compute Minimum Phase response on the active windowed data
    try:
        min_phase_signal = scipy.signal.minimum_phase(
            windowed_data, method='homomorphic', half=False
        )
        min_fft_result = np.fft.rfft(min_phase_signal, n=n_fft)
        min_phase_deg = np.angle(min_fft_result, deg=True)
    except Exception:
        min_phase_deg = phase_deg.copy()

    delta_frequency = float(sample_rate / n_fft)
    valid_idx = freqs <= (sample_rate / 2.0)
    
    return (
        freqs[valid_idx],
        mag_db[valid_idx],
        phase_deg[valid_idx],
        min_phase_deg[valid_idx],
        delta_frequency,
        fft_result[valid_idx] # Also pass raw complex spectrum out for post-rotation phase fixes
    )





def check_harmonic_window_overlaps(full_ir, inv_filter, sample_rate, start_freq, gate_start_idx, gate_end_idx, max_harmonic=5):
    """
    Checks if the given window length causes overlapping windows between 
    adjacent harmonics in a Farina method impulse response.
    """
    duration = len(inv_filter) / sample_rate
    nyquist = 0.5 * sample_rate
    end_freq = 0.99 * nyquist
    ln_ratio = np.log(end_freq / start_freq)
    
    linear_peak_idx = np.argmax(np.abs(full_ir))
    peak_indices = {}
    
    # 1. Calculate peak indices for all harmonics
    for k in range(1, max_harmonic + 1):
        if k == 1:
            sample_offset = 0
        else:
            delta_t = (duration / ln_ratio) * np.log(k)
            sample_offset = int(round(delta_t * sample_rate))
        
        peak_indices[k] = linear_peak_idx - sample_offset
        
    # 2. Initialize overlap tracking array
    overlaps = np.zeros(max_harmonic + 1, dtype=int)
    warnings_list = []
    
    # Calculate the actual active window length from user gates
    window_len = gate_end_idx - gate_start_idx
    
    # 3. Check for overlaps between adjacent harmonics
    for k in range(2, max_harmonic + 1):
        current_peak = peak_indices[k]
        
        # Correct window end relative to each harmonic's unique peak
        # (Assuming the window spans from gate_start_idx relative offset to window_len)
        window_end_sample = current_peak + window_len
        prev_peak = peak_indices[k - 1] # The lower-order harmonic to its right
        
        if window_end_sample > prev_peak:
            overlap_samples = window_end_sample - prev_peak
            overlaps[k] = overlap_samples
            warnings_list.append(
                f"Overlap Warning: Harmonic {k} window extends past the peak of "
                f"Harmonic {k-1} by {overlap_samples} samples."
            )
        else:
            overlaps[k] = 0
            
    # 4. Append a single recommendation at the end if warnings were found
    if warnings_list:
        warnings_list.append(
            "Recommendation: Consider reducing the window length or increasing the sweep duration."
        )
        
    return peak_indices, warnings_list, overlaps



def extract_harmonic_segment(full_ir, sample_rate, duration, start_freq, end_freq, timezero_sample_idx, n):
    """
    Extracts the Nth harmonic segment using Teemu Ikonen's midpoint-boundary method 
    and returns both the segment and its starting index, anchored to timezero_sample_idx.
    """
    order_t1 = n + 0.5
    order_t2 = (n - 1) + 0.5 if n > 1 else 1.5
    
    t1 = -duration * np.log(order_t1) / np.log(end_freq / start_freq)
    t2 = -duration * np.log(order_t2) / np.log(end_freq / start_freq)
    
    start_idx = timezero_sample_idx + int(round(t1 * sample_rate))
    end_idx = timezero_sample_idx + int(round(t2 * sample_rate))
    
    start_idx, end_idx = min(start_idx, end_idx), max(start_idx, end_idx)
    start_idx = max(0, min(start_idx, len(full_ir)))
    end_idx = max(0, min(end_idx, len(full_ir)))
    
    segment = full_ir[start_idx:end_idx]
    return segment, start_idx




def run_farina_analysis(
    recorded_audio,
    sample_rate,
    inv_filter,
    start_freq,
    max_valid_freq,
    Fgrid_density,
    gate_start_idx,
    gate_end_idx,
    timezero_sample_idx,
    tukey_alpha,
    max_harmonic=5
):
    """
    function run_farina_analysis:
    Uses the Farina method to extract linear and nonlinear distortion 
    responses from a recorded exponential sine sweep, supporting either 
    a raw linear frequency grid (when Fgrid_density = 0) or a logarithmic 
    energy-averaged grid on Fgrid_density frequencies per octave.

    Parameters:
    -----------
    recorded_audio : numpy.ndarray
        The raw audio data recorded from the measurement microphone.
    sample_rate : int
        The sampling rate of the audio data in Hz.
    inv_filter : numpy.ndarray
        The pre-generated inverse filter matching the stimulus sweep.
    start_freq : float
        The starting frequency of the logarithmic sweep in Hz.
    max_valid_freq : float
        The maximum valid frequency limit for the analysis in Hz.
    Fgrid_density : float or int
        Frequency grid density in points per octave. Set to 0 for a raw linear FFT grid.
    gate_start_idx : int
        The absolute sample index where the response window begins.
    gate_end_idx : int
        The absolute sample index where the response window ends.
    timezero_sample_idx : int
        The absolute sample index corresponding to the time=0 position.
    tukey_alpha : float
        The shape parameter for the Tukey window applied to each impulse slice.
    max_harmonic : int, optional
        The maximum harmonic order to extract and analyze (default is 5).

    Returns:
    --------
    dict
        A dictionary containing frequency response data, phase, 
        minimum phase, and THD percentage.
    """ 
    duration = len(inv_filter) / sample_rate
    nyquist = 0.5 * sample_rate

    # 1. Perform Fast Deconvolution
    full_ir = scipy.signal.fftconvolve(recorded_audio, inv_filter, mode='full')

    # Calculate active window length based on absolute indices
    window_len = gate_end_idx - gate_start_idx
    
    peak_indices, warnings_list, overlaps = check_harmonic_window_overlaps(
        full_ir=full_ir,
        inv_filter=inv_filter,
        sample_rate=sample_rate,
        start_freq=start_freq,
        gate_start_idx=gate_start_idx,
        gate_end_idx=gate_end_idx,
        max_harmonic=max_harmonic
    )

    if warnings_list:
        print("Farina Harmonic Window Warning:")
        for warning in warnings_list:
            print(f"  -> {warning}")

    harmonic_results = {}

    # 2. Extract and process each harmonic cleanly
    for k in range(1, max_harmonic + 1):
        if k == 1:
            start_idx = gate_start_idx
            end_idx = gate_end_idx
            segment = full_ir[start_idx:end_idx].copy()
        else:
            segment, start_idx = extract_harmonic_segment(
                full_ir, sample_rate, duration, start_freq, 0.99 * nyquist, timezero_sample_idx, k
            )

        freqs, mag_db, phase_deg, min_phase_deg, delta_f, raw_fft = compute_frequency_response(
            audio_data=segment,
            sample_rate=sample_rate,
            gate_end_idx=len(segment),
            tukey_alpha=tukey_alpha
        )

        # Add Phase Rotation relative to timezero_sample_idx
        time_delay = (start_idx - timezero_sample_idx) / sample_rate
        phase_rotation = np.exp(1j * -2.0 * np.pi * freqs * time_delay)
        aligned_fft = raw_fft * phase_rotation
        
        corrected_phase = np.angle(aligned_fft, deg=True)
        corrected_phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(corrected_phase)))
        corrected_min_phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(min_phase_deg)))

        corrected_freqs = freqs / k
        corrected_mag_db = mag_db.copy()
        
        if k > 1:
            corrected_mag_db += 20.0 * np.log10(k)

        # --- FREQUENCY GRID SELECTION ---
        if float(Fgrid_density) == 0.0:
            # Retain raw linear FFT grid from DC to Nyquist
            log_freqs = corrected_freqs
            log_mag_db = corrected_mag_db
            log_phase = corrected_phase_unwrapped
            log_min_phase = corrected_min_phase_unwrapped
        else:
            # --- LOGARITHMIC FREQUENCY GRID & ENERGY AVERAGING ---
            window_width = window_len / sample_rate
            f_window_limit = 1.0 / window_width if window_width > 0 else 0.0
            
            f_start = max(start_freq, f_window_limit, 10.0)
            harm_max_valid_freq = max_valid_freq / k
            f_max = min(harm_max_valid_freq, 0.92 * nyquist / k)
            
            if f_start < f_max:
                n_octaves = np.log2(f_max / f_start)
                num_points = int(np.ceil(n_octaves * float(Fgrid_density))) + 1
                log_freqs = f_start * (2.0 ** (np.arange(num_points) / float(Fgrid_density)))
                log_freqs = log_freqs[log_freqs <= f_max]
                
                log_mag_db = np.zeros_like(log_freqs)
                half_octave_factor = 2.0 ** (1.0 / (2.0 * float(Fgrid_density)))
                
                for i, fc in enumerate(log_freqs):
                    f_low = fc / half_octave_factor
                    f_high = fc * half_octave_factor
                    bin_mask = (corrected_freqs >= f_low) & (corrected_freqs < f_high)
                    
                    if np.any(bin_mask):
                        # Energy averaging using np.mean to normalize for expanding bandwidth
                        p_lin = 10.0 ** (corrected_mag_db[bin_mask] / 10.0)
                        log_mag_db[i] = 10.0 * np.log10(np.mean(p_lin))
                    else:
                        log_mag_db[i] = np.interp(fc, corrected_freqs, corrected_mag_db)
                        
                log_phase = np.interp(log_freqs, corrected_freqs, corrected_phase_unwrapped)
                log_min_phase = np.interp(log_freqs, corrected_freqs, corrected_min_phase_unwrapped)
            else:
                log_freqs = np.array([f_start])
                log_mag_db = np.array([0.0])
                log_phase = np.array([0.0])
                log_min_phase = np.array([0.0])
        # -----------------------------------------------------

        # --- ADDITIONAL FREQUENCY MASKING FOR k >= 1 ---
        # Mask out any frequencies exceeding the valid harmonic limit (sets invalid bins to NaN)
        max_k_freq = min(max_valid_freq / k, 0.92 * nyquist / k)
        invalid_mask = log_freqs > max_k_freq
        if np.any(invalid_mask):
            log_mag_db[invalid_mask] = np.nan
            log_phase[invalid_mask] = np.nan
            log_min_phase[invalid_mask] = np.nan

        harmonic_results[k] = {
            'freqs': log_freqs,
            'mag_db': log_mag_db,
            'phase': log_phase,
            'min_phase': log_min_phase,
            'delta_frequency': delta_f
        }

    # 3. Low-Frequency Boundary Truncation Matrix for THD Calculations & Grid Unification
    if 1 in harmonic_results and len(harmonic_results) > 1:
        f_ref = harmonic_results[1]['freqs']
        h1_mag = harmonic_results[1]['mag_db']
        sum_sq_harmonics_linear = np.zeros_like(f_ref)
        
        valid_thd_mask = f_ref >= (2.0 * start_freq)
        
        for k in range(2, max_harmonic + 1):
            if k in harmonic_results:
                h_data = harmonic_results[k]
                
                mag_k_interp = np.interp(f_ref, h_data['freqs'], h_data['mag_db'], left=np.nan, right=np.nan)
                phase_k_interp = np.interp(f_ref, h_data['freqs'], h_data['phase'], left=np.nan, right=np.nan)
                min_phase_k_interp = np.interp(f_ref, h_data['freqs'], h_data['min_phase'], left=np.nan, right=np.nan)
                
                linear_ratio = 10.0 ** ((mag_k_interp - h1_mag) / 20.0)
                
                max_harmonic_freq = sample_rate / (2.0 * k)
                linear_ratio[f_ref > max_harmonic_freq] = 0.0
                mag_k_interp[f_ref > max_harmonic_freq] = np.nan
                
                sum_sq_harmonics_linear += np.nan_to_num(linear_ratio ** 2, nan=0.0)
                
                harmonic_results[k]['freqs'] = f_ref
                harmonic_results[k]['mag_db'] = mag_k_interp
                harmonic_results[k]['phase'] = phase_k_interp
                harmonic_results[k]['min_phase'] = min_phase_k_interp
                
        thd_linear = np.sqrt(sum_sq_harmonics_linear)
        thd_percent = thd_linear * 100.0
        
        thd_percent[~valid_thd_mask] = np.nan
        harmonic_results['thd_percent'] = thd_percent
        
        harmonic_results['freqs'] = f_ref
        harmonic_results['mag_db'] = h1_mag
        harmonic_results['phase'] = harmonic_results[1]['phase']
        harmonic_results['min_phase'] = harmonic_results[1]['min_phase']
        
    return harmonic_results




def read_wav_file_plus_info(filename):
  """Reads a 32-bit float mono WAV file and extracts the audio data,
        the sample rate, and the following metadata into a dictionary:
                             parameter <> chunk identifier:
                      sweep start freq <> f0sw
                        sweep end freq <> f1sw
                           repetitions <> reps
                            chirp rate <> rchr
                        internal level <> levl
  hf_window start F for inverse filter <> fhfw
                             timestamp <> date
  """
  with open(filename, "rb") as f:
    # 1. Verify RIFF header
    riff = f.read(4)
    if riff != b"RIFF":
      raise ValueError("Not a valid RIFF WAV file.")

    file_size = struct.unpack("<I", f.read(4))[0]
    wave = f.read(4)
    if wave != b"WAVE":
      raise ValueError("Not a valid WAVE file.")

    sample_rate = None
    audio_data = []
    custom_metadata = {}

    # 2. Traverse file chunks iteratively
    while True:
      chunk_id = f.read(4)
      if not chunk_id or len(chunk_id) < 4:
        break

      chunk_size_bytes = f.read(4)
      if not chunk_size_bytes or len(chunk_size_bytes) < 4:
        break
      chunk_size = struct.unpack("<I", chunk_size_bytes)[0]

      if chunk_id == b"fmt ":
        fmt_data = f.read(chunk_size)
        # Extract audio format, number of channels, and sample rate
        audio_format, num_channels, sample_rate = struct.unpack(
            "<HHI", fmt_data[:8]
        )
        if audio_format != 3:  # 3 indicates IEEE float format
          raise ValueError(
              "WAV file is not in 32-bit IEEE float format (format tag 3)."
          )

      elif chunk_id == b"data":
        data_bytes = f.read(chunk_size)
        num_samples = chunk_size // 4
        audio_data = list(struct.unpack(f"<{num_samples}f", data_bytes))

      elif chunk_id == b"xDAT":
        xdat_data = f.read(chunk_size)
        offset = 0
        # Parse 8-byte blocks: 4 bytes tag + 4 bytes value
        while offset < chunk_size:
          tag = xdat_data[offset : offset + 4].decode("utf-8", errors="ignore")
          offset += 4

          if tag in ("date", "reps"):
            # Timestamp and repetitions are stored as uint32 integers
            val = struct.unpack("<I", xdat_data[offset : offset + 4])[0]
          else:
            # All other parameters are stored as floats
            val = struct.unpack("<f", xdat_data[offset : offset + 4])[0]
          offset += 4

          custom_metadata[tag] = val
      else:
        # Skip unknown chunks
        f.seek(chunk_size, 1)

      # Handle RIFF word-alignment padding (chunks must be even-sized bytes)
      if chunk_size % 2 != 0:
        f.seek(1, 1)

    return sample_rate, audio_data, custom_metadata




def Is_Consistent_Data(metadata_dicts, tolerance=1e-5):
  """Validates a list of pre-loaded custom metadata dictionaries to ensure
  key parameters match across the set.
                         key parameter <> chunk identifier:
                      sweep start freq <> f0sw
                        sweep end freq <> f1sw
                           repetitions <> reps
                            chirp rate <> rchr
                        internal level <> levl
  hf_window start F for inverse filter <> fhfw
  """
  if not metadata_dicts or len(metadata_dicts) < 2:
    return True  # 0 or 1 dictionary automatically matches

  # Establish the first dictionary as the baseline reference
  baseline = metadata_dicts[0]
  keys_to_check = ["f0sw", "f1sw", "reps", "rchr", "levl", "fhfw"]

  mismatches_found = False

  # Compare subsequent dictionaries against the baseline
  for idx, meta in enumerate(metadata_dicts[1:], start=2):
    for key in keys_to_check:
      val_base = baseline.get(key)
      val_curr = meta.get(key)

      # Check if a key is missing entirely
      if val_base is None or val_curr is None:
        print(f"Error: Missing key '{key}' in dictionary 1 or dictionary {idx}")
        mismatches_found = True
        continue

      # Compare based on whether values are numeric (float/int)
      if isinstance(val_base, (int, float)) and isinstance(
          val_curr, (int, float)
      ):
        # Use math.isclose if either value is a float
        if isinstance(val_base, float) or isinstance(val_curr, float):
          if not math.isclose(
              float(val_base),
              float(val_curr),
              rel_tol=tolerance,
              abs_tol=tolerance,
          ):
            print(f"Mismatch in parameter '{key}':")
            print(f"  - Baseline (Dict 1): {val_base}")
            print(f"  - Dict {idx}: {val_curr}")
            mismatches_found = True
        else:
          # Exact comparison for integers
          if val_base != val_curr:
            print(f"Mismatch in parameter '{key}':")
            print(f"  - Baseline (Dict 1): {val_base}")
            print(f"  - Dict {idx}: {val_curr}")
            mismatches_found = True
      else:
        # Fallback exact comparison for non-numeric types
        if val_base != val_curr:
          print(f"Mismatch in parameter '{key}':")
          print(f"  - Baseline (Dict 1): {val_base}")
          print(f"  - Dict {idx}: {val_curr}")
          mismatches_found = True

  if mismatches_found:
    print("\nParameter mismatch detected across dictionaries.")
    return False
  else:
    print(
        "\nAll dictionary parameters match successfully."
    )
    return True



def read_mic_cal_file(filename, freqs):
    """
    Reads a microphone calibration file (*.mic or *.cal) containing frequency (Hz) 
    and correction (dB), and interpolates corrections to match the given FFT frequencies.
    """
    cal_freqs = []
    cal_db = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('*', '#', ';')):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        f_val = float(parts[0])
                        db_val = float(parts[1])
                        cal_freqs.append(f_val)
                        cal_db.append(db_val)
                    except ValueError:
                        continue
        if not cal_freqs:
            print("Warning: Calibration file contained no valid data. No compensation applied.")
            return np.zeros_like(freqs)
            
        cal_freqs = np.array(cal_freqs)
        cal_db = np.array(cal_db)
        corrections = np.interp(freqs, cal_freqs, cal_db, left=cal_db[0], right=cal_db[-1])
        return corrections
    except Exception as e:
        print(f"Error reading microphone calibration file: {e}")
        return np.zeros_like(freqs)




def write_frd_file(filename, freqs, mag_db, phase_deg):
    """
    Saves frequency domain data to an FRD format text file.
    - Frequency: Fixed float with 1 decimal place (Hz)
    - Magnitude: 2 decimal places (dB)
    - Phase: 1 decimal place (degrees)
    - Automatically skips any NaN or infinite values.
    """
    try:
        # Ensure inputs are numpy arrays for safe masking
        freqs = np.asarray(freqs)
        mag_db = np.asarray(mag_db)
        phase_deg = np.asarray(phase_deg)
        
        # Create a mask that keeps only valid, finite numbers across all arrays
        valid_mask = np.isfinite(freqs) & np.isfinite(mag_db) & np.isfinite(phase_deg)
        
        with open(filename, 'w') as f:
            f.write("* Frequency Response Data (FRD)\n")
            f.write("* Columns: Frequency (Hz), Magnitude (dB), Phase (deg)\n")
            
            # Iterate only over the filtered, valid data points
            for freq, mag, phase in zip(freqs[valid_mask], mag_db[valid_mask], phase_deg[valid_mask]):
                freq_str = f"{freq:.1f}"
                mag_str = f"{mag:.2f}"
                phase_str = f"{phase:.1f}"
                
                f.write(f"{freq_str}\t{mag_str}\t{phase_str}\n")
                
        print(f"   ...saved {filename}")
    except Exception as e:
        print(f"WARNING: The following error was encountered when attempting to save file {filename}: {e}")



def show_notification(title, message):
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    
    # Bring the dialog to the front (helpful if working with other GUI frameworks)
    root.attributes('-topmost', True)
    
    # Pop up the modal message box (this pauses execution until OK is clicked)
    messagebox.showinfo(title, message)
    
    # Clean up the hidden root window
    root.destroy()



# ============================================================================ #
#                                                                              #
#                        ****   MAIN FUNCTION   ****                           # 
#                                                                              #
# ============================================================================ #

def main():
    # PROGRAM CONSTANTS
    MAX_HD_ORDER = 11

    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Read the raw microphone data, Farina inverse filter, and stimulus data
        sample_rate, mic_data, mic_metadata = read_wav_file_plus_info(args.measurement_datafile)
        FF_sample_rate, inv_filter, FF_metadata = read_wav_file_plus_info(args.farina_filterfile)
        H1_sample_rate, h1_impulse, impulse_metadata = read_wav_file_plus_info(args.impulse_datafile)
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
    
    if not ( H1_sample_rate == FF_sample_rate == sample_rate ):
        print(f"ERROR: The sample rates of the measurement, Farina filter, and impulse response do not agree:")
        print(f"   Measurement data SR = '{sample_rate}' Hz")
        print(f"   Farina filter SR = '{FF_sample_rate}' Hz")
        print(f"   Impulse Response SR = '{H1_sample_rate}' Hz")
        print(f"All sample rates must be identical. Exiting.")
        return

    """
    # Run a validation check on the metadata from each input file
    all_metadata = [mic_metadata, FF_metadata, impulse_metadata]
    if not Is_Consistent_Data(all_metadata):
        print("ERROR: parameter mismatch between the measurement, Farina filter, and impulse response data.")    
        print("The parameters used during the measurement must be identical for all files!")
        # Print out required measurement parameters
        print(f"\n------------------------ Measurement Parameters: -------------------------------")
        print(f"             Start Frequency (f0sw): {mic_metadata.get("f0sw", 0.0)} Hz")
        print(f"               End Frequency (f1sw): {mic_metadata.get("f1sw", 0.0)} Hz")
        print(f"                  Chirp Rate (rchr): {mic_metadata.get("rchr", 0.0)} seconds/decade Hz")
        print(f"                 Repetitions (reps): {mic_metadata.get("reps", 1)}")
        print(f"              Internal Level (levl): {mic_metadata.get("levl", -10.0)} dB")
        print(f"             HF Window Start (fhfw): {mic_metadata.get("fhfw", 0.0)} Hz")
        print(f"--------------------------------------------------------------------------------\n\n")
        print("Exiting Program.")
        return
    else:
        measurement_start_freq = mic_metadata.get("f0sw", 0.0)
        measurement_timestamp = datetime.fromtimestamp(mic_metadata.get("date", 0))
        # Print out validated parameters
        print(f"\n------------------------ Measurement Parameters: -------------------------------")
        print(f"             Start Frequency (f0sw): {mic_metadata.get("f0sw", 0.0)} Hz")
        print(f"               End Frequency (f1sw): {mic_metadata.get("f1sw", 0.0)} Hz")
        print(f"                  Chirp Rate (rchr): {mic_metadata.get("rchr", 0.0)} seconds/decade Hz")
        print(f"                 Repetitions (reps): {mic_metadata.get("reps", 1)}")
        print(f"              Internal Level (levl): {mic_metadata.get("levl", -10.0)} dB")
        print(f"             HF Window Start (fhfw): {mic_metadata.get("fhfw", 0.0)} Hz")
        print(f"                   Timestamp (date): {measurement_timestamp.strftime("%Y-%m-%d %H:%M:%S")}")
        print(f"--------------------------------------------------------------------------------\n")
    """    

    #set these parmaeters manually during debugging  only ! < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < <
    measurement_start_freq = 10.0
    HF_window_startF = 0.91 * 0.5 * sample_rate

    # 1. Compute the full impulse response using Farina deconvolution
    print("Computing the time domain impulse response using the Farina method...")
    full_ir = scipy.signal.fftconvolve(mic_data, inv_filter, mode='full')
    
    print("Using the imported impulse response to correct the Farina time axis...")
    # Determine the sample index in full_ir corresponding to time = 0
    # Define the search bounds (middle 20% of full_ir) to isolate the main impulse
    n_samples = len(full_ir)
    window_start = int(0.4 * n_samples)
    window_end = int(0.6 * n_samples)
    farina_segment = full_ir[window_start:window_end]

    # Cross-correlate the Farina segment with the H1 impulse to find the relative shift
    correlation = scipy.signal.correlate(farina_segment, h1_impulse, mode='full')
    lags = scipy.signal.correlation_lags(len(farina_segment), len(h1_impulse), mode='full')
    optimal_lag = lags[np.argmax(correlation)]

    # t0_sample_index maps index 0 (time = 0) of the H1 reference to the Farina dataset,
    # preserving the true propagation delay/latency.
    t0_sample_index = window_start + optimal_lag
    print(f"The corrected time=0 sample index in full_ir occurs at sample {t0_sample_index} of {n_samples} total samples.")

    # Construct the corrected time axis where t = 0 matches the measurement start
    time_axis = (np.arange(len(full_ir)) - t0_sample_index) / sample_rate * 1000.0

    total_duration_ms = (len(full_ir) / sample_rate) * 1000.0
    print(f"Sample Rate: {sample_rate} Hz | Duration of time data: {total_duration_ms/1000.0:.5f} seconds ({total_duration_ms:.2f} msec).\n")
    
    # Determine y-axis label for magnitude based on whether spl correction factor was supplied
    if (
        args.spl_correction_factor is not None
        and args.mic_cal_file is not None
    ):
        mag_ylabel = f"SPL [dB re 20 uPa @ 2.83V/1m" 
    else:
        mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"
    
    # Setup Figure and Subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23, hspace=0.4)
    
    # Plot Time Domain Impulse Response Data with the corrected time axis
    ax_time.plot(time_axis, full_ir, color='b', lw=1, label='IR')
    ax_time.set_title("Impulse Response (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_xlabel("Time [msec]")
    ax_time.grid(True)
    
    # Set initial x-axis view to show from slightly before time=0 
    # through the propagation delay and main impulse peak
    ax_time.set_xlim(-5.0, 50.0)  # Adjust window as needed for your setup's distance
    
    # Initial gate indices
    initial_start_time_ms = 0.0
    initial_start_idx = t0_sample_index
    initial_gate_time_ms = 20.0
    initial_gate_idx = t0_sample_index + int( initial_gate_time_ms * sample_rate / 1000.0 )

    # Add lines for both start and end gates
    start_line = ax_time.axvline(x=initial_start_time_ms, color='orange', linestyle='--', lw=1.5, label='Window Start')
    gate_line  = ax_time.axvline(x=initial_gate_time_ms,  color='g',     linestyle='--', lw=1.5, label='Window End')
    t0_line  = ax_time.axvline(x=0.0,  color='black',     linestyle='--', lw=1.5, label='time=0')
    
    # State variables
    state = {
        'gate_idx': initial_gate_idx,            
        'gate_start_idx': initial_start_idx,
        'user_has_set_gate': False,    
        'selecting_gate_step': 0,               
        'temp_start_time_ms': 0.0,              
        'selecting_gate': False,
        'raw_harmonic_results': None,
        'smoothed_harmonic_results': None,
        'selected_harmonic': 'H1',
        'hd_max': 5,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': False,
        'tukey_alpha': 0.1,
        'resolution': None,
        'smoothing_octaves': 24,
        'last_smoothing_octaves': 24,
        'HF_window_startF': HF_window_startF
    }
    
    initial_win = scipy.signal.get_window(('tukey', state['tukey_alpha']), initial_gate_idx)
    ylim_initial = ax_time.get_ylim()
    initial_y_top = ylim_initial[1] if ylim_initial[1] != 0 else 1.0
    window_line, = ax_time.plot(time_axis[:initial_gate_idx] * 1000, initial_win * (0.95 * initial_y_top), color='r', lw=1.5, label='Tukey Window')
    
    ax_time.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')

    # Get the current handles and labels from the axis
    handles, labels = ax_time.get_legend_handles_labels()

    # Recreate the legend while skipping the first entry (index 0)
    ax_time.legend(handles[1:], labels[1:],loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')

    # Configure Frequency Domain Plot
    ax_freq.set_title("Frequency Domain Data")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel(mag_ylabel)
    ax_freq.set_xscale('log')
    ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set up initial x-axis:
    ax_freq.set_xlim(20, 20000)
    
    # Define initial custom frequency ticks
    custom_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax_freq.set_xticks(custom_ticks)
    
    # Custom frequency formatter for clean labels (e.g., 1k, 10k, 20k)
    def freq_formatter(x, pos):
        if x >= 1000:
            return f"{int(x // 1000)}k" if x % 1000 == 0 else f"{x / 1000:g}k"
        else:
            return f"{int(x) if x == int(x) else x}"
            
    ax_freq.xaxis.set_major_formatter(ticker.FuncFormatter(freq_formatter))
    
    # Optional: Clean minor ticks handled automatically
    ax_freq.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))

    #initialize old_xlim
    old_xlim = ax_freq.get_xlim()

    
    # Button Axes
    btn_y = 0.03
    btn_h = 0.04
    
    ax_btn_gate         = plt.axes([0.03,  btn_y, 0.14, btn_h])
    ax_btn_calc         = plt.axes([0.19,  btn_y, 0.16, btn_h])
    ax_btn_view         = plt.axes([0.37,  btn_y, 0.18, btn_h])
    ax_btn_min          = plt.axes([0.57,  btn_y, 0.22, btn_h])
    ax_btn_save_exit    = plt.axes([0.81,  btn_y, 0.15, btn_h])
    
    btn_gate = Button(ax_btn_gate, 'Place Gate', color='aliceblue', hovercolor='palegreen')
    btn_calc = Button(ax_btn_calc, 'Compute/Clear', color='aliceblue', hovercolor='palegreen')
    btn_view = Button(ax_btn_view, 'Toggle Mag/Phase', color='aliceblue', hovercolor='palegreen')
    btn_min  = Button(ax_btn_min, 'Toggle Phase-MinPhase', color='aliceblue', hovercolor='palegreen')
    btn_save_exit = Button(ax_btn_save_exit, 'Save & Exit', color='aliceblue', hovercolor='palegreen')
    
    # Row widgets position specifiers: [left, bottom, width, height]
    # Tukey width
    ax_box_tukey = plt.axes([0.13, 0.10, 0.05, 0.03])
    textbox_tukey = TextBox(ax_box_tukey, 'TUKEY alpha: ', initial=str(state['tukey_alpha']), textalignment='center')

    # HD Max
    ax_box_hd_max = plt.axes([0.28, 0.10, 0.03, 0.03])
    textbox_hd_max = TextBox(ax_box_hd_max, 'HD MAX: ', initial=str(state['hd_max']), textalignment='center')

    # Smoothing Octaves
    ax_box_sm_octaves = plt.axes([0.51, 0.10, 0.03, 0.03])
    textbox_sm_octaves = TextBox(ax_box_sm_octaves, 'OCTAVE SMOOTHING 1/: ', initial=str(state['smoothing_octaves']), textalignment='center')

    # Phase Type
    ax_box_phase_type = plt.axes([0.66, 0.10, 0.13, 0.03])
    initial_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
    phase_textbox = TextBox(ax_box_phase_type, 'PHASE TYPE: ', initial=initial_text, textalignment='center')
    phase_textbox.set_active(False)

    # Dataset Toggles
    ax_btn_prev = plt.axes([0.82, 0.10, 0.035, 0.03])
    ax_box_dataset = plt.axes([0.860, 0.10, 0.05, 0.03])
    ax_btn_next = plt.axes([0.915, 0.10, 0.035, 0.03])

    btn_prev = Button(ax_btn_prev, r'$\blacktriangleleft$', color='aliceblue', hovercolor='palegreen')
    textbox_dataset = TextBox(ax_box_dataset, '', initial=state['selected_harmonic'], textalignment='center')
    textbox_dataset.set_active(False)
    btn_next = Button(ax_btn_next, r'$\blacktriangleright$', color='aliceblue', hovercolor='palegreen')

    def get_harmonic_list():
        h_list = [f"H{i}" for i in range(1, state['hd_max'] + 1)]
        h_list.extend(['ALL', 'THD'])
        return h_list

    def update_tukey_visual(event=None):
        ylim = ax_time.get_ylim()
        y_top = ylim[1] if ylim[1] != 0 else 1.0
        
        start_idx = max(0, state.get('gate_start_idx', 0))
        end_idx = state['gate_idx']
        
        if start_idx < end_idx:
            win_len = end_idx - start_idx + 1
            win = scipy.signal.get_window(('tukey', state['tukey_alpha']), win_len)
            
            # Fix 1: time_axis is already in milliseconds, remove the extra * 1000 multiplier
            window_line.set_xdata(time_axis[start_idx:end_idx + 1])
            window_line.set_ydata(win * (0.95 * y_top))
            
            # Fix 2: Correctly map absolute sample indices back to the shifted ms time axis
            start_time_ms = ((start_idx - t0_sample_index) / sample_rate) * 1000.0
            end_time_ms = ((end_idx - t0_sample_index) / sample_rate) * 1000.0
            
            start_line.set_xdata([start_time_ms, start_time_ms])
            gate_line.set_xdata([end_time_ms, end_time_ms])
            
        fig.canvas.draw_idle()

    ax_time.callbacks.connect('ylim_changed', update_tukey_visual)
    ax_time.callbacks.connect('xlim_changed', update_tukey_visual)

    def submit_tukey_alpha(text):
        try:
            val = float(text)
            if 0.001 <= val <= 0.3:
                state['tukey_alpha'] = val
                print(f"Tukey alpha updated to: {state['tukey_alpha']}")
                update_tukey_visual()
            else:
                user_message = f"Error: A Tukey alpha of ({val}) is out of bounds. Must be between 0.001 and 0.3."
                print(user_message)
                show_notification("Error", user_message)
                textbox_tukey.set_val(str(state['tukey_alpha']))
        except ValueError:
            textbox_tukey.set_val(str(state['tukey_alpha']))

    textbox_tukey.on_submit(submit_tukey_alpha)

    def submit_hd_max(text):
        try:
            val = int(text)
            if 2 <= val <= MAX_HD_ORDER:
                if val == state['hd_max']:
                    return
                state['hd_max'] = val
                user_message = f"A new HD MAX of {state['hd_max']} was provided.\n" 
                user_message += "Press Compute/Clear to clear the data and then again to show the new results."
                print(user_message)
                show_notification("Error", user_message)
                state['selected_harmonic'] = 'H1'
                textbox_dataset.set_val('H1')
                if state['smoothed_harmonic_results'] is not None:
                    refresh_frequency_plot()
            else:
                user_message = f"Error: valid HD MAX values are 2 through '{MAX_HD_ORDER}'."
                print(user_message)
                show_notification("Error", user_message)
                textbox_hd_max.set_val(str(state['hd_max']))
        except ValueError:
            textbox_hd_max.set_val(str(state['hd_max']))

    textbox_hd_max.on_submit(submit_hd_max)

    def submit_smoothing(text):
        try:
            cleaned = text.strip().lower()
            if cleaned in ('none', 'off', '0', ''):
                val = 0
            else:
                val = int(text)
                
            # Safely ensure type consistency for comparison
            current_last = int(state['last_smoothing_octaves']) if state['last_smoothing_octaves'] is not None else 0
            
            if val == current_last:
                return
                
            if 0 <= val <= 48:
                user_message = "A new smoothing level was provided.\nPress Compute/Clear to clear the data and then again to show the new "
                state['smoothing_octaves'] = val
                
                if val == 0:
                    user_message += "unsmoothed data."
                else:
                    user_message += f"data smoothed to 1/{val} octave."
                
                # FIX: Moved print and notification OUTSIDE the inner if/else so they run for val == 0 too
                print(user_message)
                show_notification("Tip", user_message)
                
                state['last_smoothing_octaves'] = val 
            else:
                user_message = "You entered an invalid value. Valid octave smoothing values run from 0 (raw FFT bins) to 48 points per octave."
                print(user_message)
                show_notification("Error", user_message)
                current_val = state['smoothing_octaves']
                textbox_sm_octaves.set_val(str(current_val) if current_val is not None else "None")
                
        except ValueError:
            print("Error: Invalid smoothing value entered. Please enter an integer.")
            current_val = state['smoothing_octaves']
            textbox_sm_octaves.set_val(str(current_val) if current_val is not None else "None")

    textbox_sm_octaves.on_submit(submit_smoothing)


    def on_click_time_plot(event):
        if state['selecting_gate_step'] > 0 and event.inaxes == ax_time:
            clicked_time_ms = event.xdata
            if clicked_time_ms is not None:
                # Convert the clicked time (relative to the new t=0 point) back to an absolute sample index
                clicked_sample = t0_sample_index + int((clicked_time_ms / 1000.0) * sample_rate)
                
                if state['selecting_gate_step'] == 1:
                    state['temp_start_time_ms'] = clicked_time_ms
                    print(f"   The end of the window taper was placed at: {clicked_time_ms:.2f} ms.")
                    print(f"Step 2 of 2: Now click again to set the Window end time(Reflection limit).")
                    state['selecting_gate_step'] = 2
                    
                elif state['selecting_gate_step'] == 2:
                    # Convert the saved start time back to an absolute sample index using t0_sample_index
                    clicked_onset_sample = t0_sample_index + int((state['temp_start_time_ms'] / 1000.0) * sample_rate)
                    end_sample = clicked_sample
                    
                    if end_sample > clicked_onset_sample:
                        temp_window_len = end_sample - clicked_onset_sample
                        taper_len = int(temp_window_len * state['tukey_alpha'] / 2.0)
                        true_start_sample = max(0, clicked_onset_sample - taper_len)
                        
                        state['gate_start_idx'] = true_start_sample
                        state['gate_idx'] = end_sample
                        
                        # Convert absolute sample indices back to the new shifted time axis (in ms)
                        actual_start_ms = ((true_start_sample - t0_sample_index) / sample_rate) * 1000.0
                        clicked_onset_ms = state['temp_start_time_ms']
                        actual_end_ms = ((end_sample - t0_sample_index) / sample_rate) * 1000.0
                        
                        print(f"   The end of the window was placed at: {clicked_time_ms:.2f} ms.\n")
                        print(f"Window Updated!")
                        print(f"  - Window starts at: {actual_start_ms:.2f} ms")
                        print(f"  - Window ends at: {actual_end_ms:.2f} ms")
                        print(f"  - Time Resolution for this window: {1000.0/(actual_end_ms-actual_start_ms):.3g} Hz\n")
                        
                        start_line.set_xdata([actual_start_ms, actual_start_ms])
                        gate_line.set_xdata([actual_end_ms, actual_end_ms])
                        update_tukey_visual()
                        state['user_has_set_gate'] = True
                    else:
                        print("Error: Window end must be after the window start. Try again.")
                    
                    state['selecting_gate_step'] = 0
                    btn_gate.label.set_color('black')
                    ax_btn_gate.set_facecolor('aliceblue')
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click_time_plot)

    def handle_gate_button(event):
        if state['selecting_gate_step'] == 0:
            if state['user_has_set_gate'] == False:
                #first time to click on the Place Gate tool, so give instructions via a pop-up
                user_message = "Follow these steps to set the window:\n"
                user_message += "Click on the time plot "
                user_message += "where the impulse first leaves the baseline. Then "
                user_message += "click again at a later time point to set the end "
                user_message += "time for the Tukey window. Adjust the Tukey alpha "
                user_message += "parameter to change the width of the tapers. This "
                user_message += "message will not be shown again."
                show_notification("INSTRUCTIONS", user_message)
            state['selecting_gate_step'] = 1
            btn_gate.label.set_color('red')
            ax_btn_gate.set_facecolor('#ffcccc')
            print("Setting a new time domain window:")
            print("Step 1 of 2: Click on the time plot where the impulse first leaves the baseline...")
        else:
            state['selecting_gate_step'] = 0
            btn_gate.label.set_color('black')
            ax_btn_gate.set_facecolor('aliceblue')
            print("Gate selection cancelled.")
        fig.canvas.draw_idle()
        
    def refresh_frequency_plot():
        if state['smoothed_harmonic_results'] is None:
            return

        current_sel = state['selected_harmonic']
        current_phase = state['showing_phase']

        results = state['smoothed_harmonic_results']
        sel = current_sel
        
        # 1. Capture current X-axis limit (to preserve frequency zoom across views)
        try:
            old_xlim = ax_freq.get_xlim()
        except Exception:
            old_xlim = None

        # Clear the axis and set up grid / scale
        ax_freq.clear()
        ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_freq.set_xscale('log')

        # 2. Extract data and mask out NaNs from mag_db to get valid x-values for xmax
        freqs = np.array(results[1]['freqs'])
        mag_db = np.array(results[1]['mag_db'])

        valid_mask = ~np.isnan(mag_db)
        x_masked = freqs[valid_mask]
        xmax = max(x_masked)
        xmax = min(xmax, state['HF_window_startF'])

        # 3. Get the minimum resolution frequency from state & generate master ticks
        res_min = state["resolution"]

        start_decade = int(np.floor(np.log10(res_min))) - 1
        end_decade = int(np.ceil(np.log10(xmax)))
        decades = np.arange(start_decade, end_decade + 1)

        multipliers = [1, 2, 3, 5]
        master_ticks = []
        for d in decades:
            for m in multipliers:
                master_ticks.append(m * (10 ** d))

        # Lower bound: largest master tick <= res_min (tick just below data start)
        valid_lower_ticks = [t for t in master_ticks if t <= res_min]
        xmin_temp = max(valid_lower_ticks) if valid_lower_ticks else res_min

        # Upper bound: smallest master tick >= xmax (next tick *above* last data point)
        valid_upper_ticks = [t for t in master_ticks if t >= xmax]
        xmax_temp = min(valid_upper_ticks) if valid_upper_ticks else xmax

        # Filter active ticks inclusively between the lower and upper buffer ticks
        active_ticks = [t for t in master_ticks if xmin_temp <= t <= xmax_temp]

        # 4. Apply custom ticks and formatting first
        ax_freq.set_xticks(active_ticks)
        
        # Major ticks: Use smart 'k' formatting for thousands, standard numbers below
        def freq_formatter(x, pos):
            if x >= 1000:
                return f"{int(x // 1000)}k" if x % 1000 == 0 else f"{x / 1000:g}k"
            else:
                return f"{int(x) if x == int(x) else x}"
                
        ax_freq.xaxis.set_major_formatter(ticker.FuncFormatter(freq_formatter))
        
        # Define a smart dynamic formatter for minor ticks
        def smart_minor_formatter(x, pos):
            xlim = ax_freq.get_xlim()
            if xlim[0] > 0 and xlim[1] > 0:
                span_decades = np.log10(xlim[1] / xlim[0])
            else:
                span_decades = 3.0  
            
            if span_decades > 1.5:
                return ""
            
            # Apply 'k' formatting to minor ticks too if they exceed 1000
            if x >= 1000:
                return f"{int(x // 1000)}k" if x % 1000 == 0 else f"{x / 1000:g}k"
            else:
                return f"{x:g}"

        ax_freq.xaxis.set_minor_formatter(ticker.FuncFormatter(smart_minor_formatter))
        ax_freq.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))

        # 5. Handle X-axis limits (preserve user zoom or snap neatly to custom ticks)
        is_default_limit = (not old_xlim) or (old_xlim == (0.0, 1.0)) or (abs(old_xlim[0] - 20.0) < 0.1 and abs(old_xlim[1] - 20000.0) < 1.0)

        if not is_default_limit:
            # User has actively zoomed in! Respect their zoom range.
            ax_freq.set_xlim(old_xlim)
        else:
            # Default state: Snap to your custom active ticks range
            if active_ticks:
                xmin_val = float(active_ticks[0])
                xmax_val = float(active_ticks[-1])
                ax_freq.set_xlim(xmin_val, xmax_val)
            else:
                ax_freq.set_xlim(20.0, 20000.0)

        # Define k_idx so it is available for BOTH phase and magnitude views
        k_idx = int(sel[1:]) if len(sel) > 1 and sel[1:].isdigit() else 1

        if current_phase:
            # --- PHASE VIEW BRANCH ---
            if sel == 'ALL' or sel == 'THD':
                ax_freq.set_ylabel("Phase (degrees)")
                ax_freq.set_title("Phase View (Blank for ALL/THD)")
                state['freq_line_plot'] = []
            else:
                if k_idx <= state['hd_max']:
                    raw_phase = results[k_idx]['min_phase'] if state['using_min_phase'] else results[k_idx]['phase']
                    
                    # Wrap phase into the [-180, 180] degree range
                    current_phase_data = (raw_phase + 180) % 360 - 180

                    title_label = f"Harmonic {k_idx} - {'Minimum Phase' if state['using_min_phase'] else 'Phase'}"
                    line, = ax_freq.plot(results[k_idx]['freqs'], current_phase_data, color='g', lw=1.2)
                    ax_freq.set_ylabel("Phase (degrees)")
                    ax_freq.set_title(title_label)
                    state['freq_line_plot'] = [line]
        else:
            # --- MAGNITUDE VIEW BRANCH ---
            if sel == 'ALL':
                freqs = results[1]['freqs']
                line1, = ax_freq.plot(freqs, results[1]['mag_db'], color='r', lw=1.2, label='H1 (Fund)')
                lines = [line1]
                colors = ['blue', 'tomato', 'green', 'black', 'aqua', 'blueviolet', 'peru', 'deeppink']
                for idx, k in enumerate(range(2, state['hd_max'] + 1)):
                    if k in results:
                        l, = ax_freq.plot(results[k]['freqs'], results[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='-', label=f'H{k}')
                        lines.append(l)
                ax_freq.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')
                ax_freq.set_ylabel(mag_ylabel)
                ax_freq.set_title("Frequency Domain Data - All Harmonics Magnitude")
                state['freq_line_plot'] = lines
            elif sel == 'THD':
                freqs = results[1]['freqs']
                thd_data = results['thd_percent']
                line, = ax_freq.plot(freqs, thd_data, color='blue', lw=1.2)
                
                # Set the plot y-axis minimum to zero 
                ax_freq.set_ylim(bottom=0)
                
                valid_max = np.nanmax(thd_data) if not np.isnan(thd_data).all() else 10.0
                upper_limit = max( min(10.0, valid_max), 1.0)
                
                ax_freq.set_ylim(top=upper_limit)
                
                ax_freq.set_ylabel("THD (%)")
                ax_freq.set_title("Total Harmonic Distortion (THD)")
                state['freq_line_plot'] = [line]
            else:
                if k_idx <= state['hd_max']:
                    if k_idx > 1 and 1 in results:
                        h1_interp = np.interp(results[k_idx]['freqs'], results[1]['freqs'], results[1]['mag_db'])
                        mag_data = results[k_idx]['mag_db'] - h1_interp
                        ylabel_str = "Level Relative to H1 (dB)"
                        title_str = f"Harmonic {k_idx} - Magnitude Relative to Fundamental"
                    else:
                        mag_data = results[k_idx]['mag_db']
                        ylabel_str = mag_ylabel
                        title_str = f"Frequency Domain Data - Magnitude (Harmonic {k_idx})"

                    line, = ax_freq.plot(results[k_idx]['freqs'], mag_data, color='r', lw=1.2)
                    ax_freq.set_ylabel(ylabel_str)
                    ax_freq.set_title(title_str)
                    state['freq_line_plot'] = [line]

            ax_freq.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')

        # 6. Refresh canvas
        fig.canvas.draw_idle()

    def handle_calc_button(event):
        if state['user_has_set_gate'] == False:
            user_message = "You must place the start and end gates to define the window "
            user_message += "at least once before calculating the results. Click on the "
            user_message += "Place Gate button. Click on the time plot to place the gate points."
            print(user_message)
            show_notification("Error", user_message)
            return
            
        if state['smoothed_harmonic_results'] is not None:
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_ylabel(mag_ylabel)
            ax_freq.set_title("Frequency Domain Data (Cleared)")
            ax_freq.set_xlabel('Frequency (Hz)')

            # Set up initial x-axis:
            ax_freq.set_xlim(20, 20000)
            
            # Define initial custom frequency ticks
            custom_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            ax_freq.set_xticks(custom_ticks)
            
            # Custom frequency formatter for clean labels (e.g., 1k, 10k, 20k)
            def freq_formatter(x, pos):
                if x >= 1000:
                    return f"{int(x // 1000)}k" if x % 1000 == 0 else f"{x / 1000:g}k"
                else:
                    return f"{int(x) if x == int(x) else x}"
                    
            ax_freq.xaxis.set_major_formatter(ticker.FuncFormatter(freq_formatter))
            
            # Optional: Clean minor ticks handled automatically
            ax_freq.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))

            fig.canvas.draw_idle()
            state['smoothed_harmonic_results'] = None
            state['resolution'] = None
            print("All calculated data cleared.\n")
        else:
            print(f"Computing Farina Method response up to HD {state['hd_max']}...")
            
            results = run_farina_analysis(
                recorded_audio=mic_data,
                sample_rate=sample_rate,
                inv_filter=inv_filter,
                start_freq=measurement_start_freq,
                max_valid_freq=HF_window_startF,
                Fgrid_density=state['smoothing_octaves'],
                gate_start_idx=state['gate_start_idx'],
                gate_end_idx=state['gate_idx'],
                timezero_sample_idx=t0_sample_index,
                tukey_alpha=state['tukey_alpha'],
                max_harmonic=state['hd_max']
            )
            
            # Apply SPL corrections:
            mic_cal_file = getattr(args, 'mic_cal_file', None)
            K = getattr(args, 'spl_correction_factor', None)

            # Convert the linear system factor K into a decibel offset
            if K is not None and K > 0:
                spl_offset_db = 20 * np.log10(K)
            else:
                spl_offset_db = 0.0

            # Loop through all k harmonics stored in the dictionary
            for k, harmonic_data in results.items():
                # Skip top-level summary keys (like 'freqs', 'thd_percent', etc.)
                if not isinstance(k, int):
                    continue
                    
                freqs = harmonic_data['freqs']
                
                # Ensure mag_db is treated as a NumPy array so math operations work safely
                mag_db = np.asarray(harmonic_data['mag_db'], dtype=float)
                
                # 1. Apply the system SPL correction factor
                mag_db += spl_offset_db
                
                # 2. Apply the microphone calibration corrections if a file was provided
                if mic_cal_file is not None:
                    cal_corrections = read_mic_cal_file(mic_cal_file, freqs)
                    mag_db -= np.asarray(cal_corrections, dtype=float)
                    
                # Update the dictionary with the fully calibrated magnitude data
                harmonic_data['mag_db'] = mag_db            
            
            #store the results in the state variable
            state['smoothed_harmonic_results'] = results
            
            window_duration = (state['gate_idx'] - state['gate_start_idx']) / sample_rate
            state['resolution'] = 1.0 / window_duration

            refresh_frequency_plot()
            print(f"The Farina analysis has completed and new data is available.\n")

            # --- FIX THE HOME BUTTON STACK HERE ---
            toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
            if toolbar is not None:
                # Clear out the old empty startup history stack
                if hasattr(toolbar, '_views'):
                    toolbar._views.clear()
                    toolbar._positions.clear()
                    toolbar.push_current()  # Pushes the current valid data limits as the new Home baseline!
                elif hasattr(toolbar, '_nav_stack'):
                    # For newer matplotlib versions using NavigationStack
                    toolbar._nav_stack.clear()
                    toolbar.push_current()

    def handle_view_button(event):
        if state['smoothed_harmonic_results'] is not None:
            state['showing_phase'] = not state['showing_phase']
            refresh_frequency_plot()
            update_phase_indicator()
            
            # Reset toolbar nav stack for the new view
            toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
            if toolbar is not None:
                if hasattr(toolbar, '_views'):
                    toolbar._views.clear()
                    toolbar._positions.clear()
                    toolbar.push_current()
                elif hasattr(toolbar, '_nav_stack'):
                    toolbar._nav_stack.clear()
                    toolbar.push_current()
        else:
            print("No frequency data computed yet. Click 'Compute/Clear' first.")

    def update_phase_indicator():
        new_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
        phase_textbox.set_val(new_text)

    def handle_min_phase_button(event):
        state['using_min_phase'] = not state['using_min_phase']
        update_phase_indicator()
        if state['smoothed_harmonic_results'] is not None and state['showing_phase']:
            refresh_frequency_plot()

    def handle_cycle_left(event):
        h_list = get_harmonic_list()
        try:
            curr_idx = h_list.index(state['selected_harmonic'])
        except ValueError:
            curr_idx = 0
        new_idx = (curr_idx - 1) % len(h_list)
        state['selected_harmonic'] = h_list[new_idx]
        textbox_dataset.set_val(state['selected_harmonic'])
        
        # If user cycles into ALL or THD while in phase view, automatically switch back to magnitude view
        if state['selected_harmonic'] in ('ALL', 'THD') and state['showing_phase']:
            state['showing_phase'] = False
            print("Note: Phase view is not available for ALL or THD. Switched back to Magnitude view.")

        if state['smoothed_harmonic_results'] is not None:
            refresh_frequency_plot()
            
            # Reset toolbar nav stack so "Back" doesn't jump to previous harmonics
            toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
            if toolbar is not None:
                if hasattr(toolbar, '_views'):
                    toolbar._views.clear()
                    toolbar._positions.clear()
                    toolbar.push_current()
                elif hasattr(toolbar, '_nav_stack'):
                    toolbar._nav_stack.clear()
                    toolbar.push_current()

    def handle_cycle_right(event):
        h_list = get_harmonic_list()
        try:
            curr_idx = h_list.index(state['selected_harmonic'])
        except ValueError:
            curr_idx = 0
        new_idx = (curr_idx + 1) % len(h_list)
        state['selected_harmonic'] = h_list[new_idx]
        textbox_dataset.set_val(state['selected_harmonic'])
        
        # If user cycles into ALL or THD while in phase view, automatically switch back to magnitude view
        if state['selected_harmonic'] in ('ALL', 'THD') and state['showing_phase']:
            state['showing_phase'] = False
            print("Note: Phase view is not available for ALL or THD. Switched back to Magnitude view.")

        if state['smoothed_harmonic_results'] is not None:
            refresh_frequency_plot()
            
            # Reset toolbar nav stack so "Back" doesn't jump to previous harmonics
            toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
            if toolbar is not None:
                if hasattr(toolbar, '_views'):
                    toolbar._views.clear()
                    toolbar._positions.clear()
                    toolbar.push_current()
                elif hasattr(toolbar, '_nav_stack'):
                    toolbar._nav_stack.clear()
                    toolbar.push_current()

    def handle_save_exit_button(event):
        if state['smoothed_harmonic_results'] is None:
            print("WARNING: There is no data to save. Run the calculation first before saving.")
            return
            
        # Ensure the destination directory exists if base_name includes a path
        import os
        save_dir = os.path.dirname(args.basename)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        if state['using_min_phase']:
            print(f"Saving FRD data files (minimum phase response)...")
        else:
            print(f"Saving FRD data files...")
        res = state['smoothed_harmonic_results']
        base_name = args.basename  
        
        for k in range(1, state['hd_max'] + 1):
            if k in res:
                h_data = res[k]
                phase_to_save = h_data['min_phase'] if state['using_min_phase'] else h_data['phase']
                filename_k = f"{base_name}_HD{k}.frd"
                write_frd_file(filename_k, h_data['freqs'], h_data['mag_db'], phase_to_save)

        print("\nSaving plots at 600 DPI in png file format...")
        
        # Dynamically compute active ticks matching the on-screen UI safely
        freqs = np.array(res[1]['freqs'])
        mag_db = np.array(res[1]['mag_db'])
        valid_mask = ~np.isnan(mag_db)
        x_masked = freqs[valid_mask]
        xmax = max(x_masked) if len(x_masked) > 0 else 20000.0
        
        res_min = state.get("resolution")
        if res_min is None or res_min <= 0:
            res_min = 20.0
            
        try:
            start_decade = int(np.floor(np.log10(res_min))) - 1
            end_decade = int(np.ceil(np.log10(xmax))) + 1  # Headroom for upper bound
            decades = np.arange(start_decade, end_decade + 1)
            
            # Use multipliers including 3 to catch 30k instead of jumping to 50k
            multipliers = [1, 2, 3, 5]
            master_ticks = []
            for d in decades:
                for m in multipliers:
                    t = m * (10 ** d)
                    if t > 0:
                        master_ticks.append(t)
                        
            # Lower bound: largest master tick <= res_min
            valid_lower_ticks = [t for t in master_ticks if t <= res_min]
            xmin_temp = max(valid_lower_ticks) if valid_lower_ticks else res_min
            
            # Upper bound: smallest master tick >= xmax (prevents data clipping)
            valid_upper_ticks = [t for t in master_ticks if t >= xmax]
            xmax_temp = min(valid_upper_ticks) if valid_upper_ticks else xmax
            
            active_ticks = [t for t in master_ticks if xmin_temp <= t <= xmax_temp]
        except Exception:
            active_ticks = [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000]

        def apply_custom_scaling_and_ticks(ax):
            ax.set_xscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Ensure all ticks are strictly positive for log scale
            positive_ticks = [t for t in active_ticks if t > 0]
            if positive_ticks:
                ax.set_xticks(positive_ticks)
                ax.set_xlim(float(positive_ticks[0]), float(positive_ticks[-1]))
            else:
                ax.set_xlim(20.0, 20000.0)
                
            # Custom frequency formatter for clean labels (e.g., 1k, 10k, 30k)
            def freq_formatter(x, pos):
                if x >= 1000:
                    return f"{int(x // 1000)}k" if x % 1000 == 0 else f"{x / 1000:g}k"
                else:
                    return f"{int(x) if x == int(x) else x}"

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(freq_formatter))
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))

        # Plot A: Fundamental Magnitude (_FREQ.png)
        fig_freq, ax_f = plt.subplots(figsize=(11, 5))
        ax_f.plot(res[1]['freqs'], res[1]['mag_db'], color='r', lw=1.2)
        ax_f.set_title("Fundamental Frequency Response - Magnitude")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel(mag_ylabel)
        apply_custom_scaling_and_ticks(ax_f)
        fig_freq.savefig(f"{base_name}_FREQ.png", dpi=600, bbox_inches='tight')
        plt.close(fig_freq)

        # Plot B: Fundamental Phase (_PHASE.png)
        fig_phase, ax_p = plt.subplots(figsize=(11, 5))
        phase_raw = res[1]['min_phase'] if state['using_min_phase'] else res[1]['phase']
        phase_to_plot = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(phase_raw))))
        phase_title = "Fundamental Minimum Phase (Wrapped)" if state['using_min_phase'] else "Fundamental Phase (Wrapped)"
        ax_p.plot(res[1]['freqs'], phase_to_plot, color='g', lw=1.2)
        ax_p.set_title(phase_title)
        ax_p.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')
        ax_p.set_ylabel("Phase (degrees)")
        ax_p.set_ylim(-180, 180)
        ax_p.set_yticks([-180, -90, 0, 90, 180])
        apply_custom_scaling_and_ticks(ax_p)
        fig_phase.savefig(f"{base_name}_PHASE.png", dpi=600, bbox_inches='tight')
        plt.close(fig_phase)

        # Plot C: All Harmonics Magnitude (_HD.png)
        fig_hd, ax_h = plt.subplots(figsize=(11, 5))
        ax_h.plot(freqs, res[1]['mag_db'], color='r', lw=1.2, label='H1 (Fund)')
        colors = ['blue', 'tomato', 'green', 'black', 'aqua', 'blueviolet', 'peru', 'deeppink']
        for idx, k in enumerate(range(2, state['hd_max'] + 1)):
            if k in res:
                ax_h.plot(res[k]['freqs'], res[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='-', label=f'H{k}')
        ax_h.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')
        ax_h.set_title("Harmonic Distortion - All Harmonics")
        ax_h.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')
        ax_h.set_ylabel(mag_ylabel)
        apply_custom_scaling_and_ticks(ax_h)
        fig_hd.savefig(f"{base_name}_HD.png", dpi=600, bbox_inches='tight')
        plt.close(fig_hd)

        # Plot D: Total Harmonic Distortion (_THD.png)
        fig_thd, ax_t = plt.subplots(figsize=(11, 5))
        ax_t.plot(res[1]['freqs'], res['thd_percent'], color='blue', lw=1.2)
        ax_t.set_title("Total Harmonic Distortion (THD)")
        ax_t.set_xlabel("Frequency (Hz)")
        ax_t.set_ylabel("THD (%)")
        apply_custom_scaling_and_ticks(ax_t)
        fig_thd.savefig(f"{base_name}_THD.png", dpi=600, bbox_inches='tight')
        plt.close(fig_thd)

        print("\nCompleted all EXIT related actions. Now closing the application.")
        plt.close('all')
        sys.exit(0)

    btn_gate.on_clicked(handle_gate_button)
    btn_calc.on_clicked(handle_calc_button)
    btn_view.on_clicked(handle_view_button)
    btn_min.on_clicked(handle_min_phase_button)
    btn_save_exit.on_clicked(handle_save_exit_button)
    btn_prev.on_clicked(handle_cycle_left)
    btn_next.on_clicked(handle_cycle_right)
    
    # Access the toolbar via the canvas manager
    toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None

    if toolbar is not None:
        # For Qt Backends (PyQt / PySide)
        if hasattr(toolbar, 'actions'):
            for action in toolbar.actions():
                text = action.text().lower()
                tooltip = action.toolTip().lower()
                if 'home' in text or 'reset' in text or 'home' in tooltip or 'reset' in tooltip:
                    toolbar.removeAction(action)
                    
        # For Tkinter (TkAgg) Backend
        elif hasattr(toolbar, '_buttons'):
            if 'Home' in toolbar._buttons:
                toolbar._buttons['Home'].pack_forget() # Hide the home button widget

    # Trigger default zoom tool if desired
    if toolbar is not None:
        toolbar.zoom()

    plt.show()


if __name__ == "__main__":
    main()


