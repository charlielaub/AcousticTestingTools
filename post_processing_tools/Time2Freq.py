"""
Time2Freq - Convert the Time Domain Response to the Frequency Domain Response 
Description: Calculates the time domain frequency response and the harmonic 
    levels using the Farina method, plus correction factors. Can produce 
    relative SPL levels, or the SPL level in units of dB re 20 uPa @ 2.83V/1m. 
    Both the regular phase and minimum phase responses are made available.

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
from matplotlib.ticker import LogLocator
import math


def parse_arguments():
    """
    Parses command-line arguments for input WAV file, base filename for outputs,
    optional microphone calibration, sensitivity parameters, and amplifier gain.
    """
    parser = argparse.ArgumentParser(description="Process loudspeaker impulse response WAV file and export data/plots (Interactive GUI).")
    parser.add_argument("-m","--mic_data", 
            type=str,
            required=True,
            help="Path to the microphone data WAV file."
    )
    parser.add_argument("-f","--farina_filter", 
            type=str,
            required=True,
            help="Path to the Farina inverse filter WAV file."
    )
    parser.add_argument("--start_f", 
            type=float, 
            required=True,
            help="Start frequency of stimulus sweep in Hz."
    )
    parser.add_argument("-o", "--output", 
            type=str, 
            default="response",
            help="Base filename for output files (default: response)."
    )
    parser.add_argument("-c","--mic_cal", 
            type=str, 
            default=None,
            help="Path to microphone calibration file (*.mic or *.cal)."
    )
    parser.add_argument("-s","--mic_sensitivity", 
            type=float, 
            default=None,
            help="Microphone sensitivity in milliVolts per Pascal (mV/Pa)."
    )
    parser.add_argument("-v","--v_stimulus", 
            type=float, 
            default=2.83,
            help="RMS Voltage reported in SPL units (default: 2.83V)."
    )
    parser.add_argument("-d","--distance", 
            type=float, 
            default=None,
            help="Distance between mic and source in meters."
    )
    return parser.parse_args()




def compute_frequency_response(
    audio_data,
    sample_rate,
    gate_end_idx,
    tukey_alpha,
    mic_cal_file,
    mic_sensitivity,
    v_stimulus,
    distance,
):
    """
    Applies a Tukey window, computes a standardized normalized FFT,
    applies acoustic hardware correction factors, and returns frequencies, 
    calibrated magnitude (dBSPL), phase vectors, and delta frequency.
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

    # --- Apply Real-World Acoustic Calibration Matrices ---
    if mic_sensitivity is not None and mic_sensitivity > 0:
        sensitivity_v_pa = mic_sensitivity / 1000.0  # mV/Pa to V/Pa
        p_ref = 20e-6  # 20 uPa reference pressure
        spl_offset = -20.0 * np.log10(sensitivity_v_pa * p_ref)
        mag_db += spl_offset

    if mic_cal_file is not None:
        # Assumes your read_mic_cal_file helper function is accessible globally
        cal_corrections = read_mic_cal_file(mic_cal_file, freqs)
        mag_db -= cal_corrections

    # Scale for stimulus RMS voltage
    mag_db -= 20.0 * np.log10(v_stimulus)

    # Scale for free-field acoustic measurement distance
    if distance is not None:
        mag_db += 20.0 * np.log10(distance)

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




def nth_harmonic_time(n, duration, f1, f2):
    """Computes the relative time offset of the n-th harmonic (or fractional order)
    using Teemu Ikonen's Farina sweep formulation.
    """
    return -duration * np.log(n) / np.log(f2 / f1)



def extract_harmonic_segment(full_ir, sample_rate, duration, start_freq, end_freq, linear_peak_idx, n):
    """
    Extracts the Nth harmonic segment using Teemu Ikonen's midpoint-boundary method 
    and returns both the segment and its starting index.
    """
    order_t1 = n + 0.5
    order_t2 = (n - 1) + 0.5 if n > 1 else 1.5
    
    t1 = -duration * np.log(order_t1) / np.log(end_freq / start_freq)
    t2 = -duration * np.log(order_t2) / np.log(end_freq / start_freq)
    
    start_idx = linear_peak_idx + int(round(t1 * sample_rate))
    end_idx = linear_peak_idx + int(round(t2 * sample_rate))
    
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
        gate_start_offset,  # Changed to relative sample offsets (e.g., -100)
        gate_end_offset,    # Changed to relative sample offsets (e.g., 1000)
        tukey_alpha,
        mic_cal_file,
        mic_sensitivity,
        v_stimulus,
        distance,
        max_harmonic=5
    ):
    """
          function run_farina_analysis:
          Uses the Farina method to extract linear and nonlinear distortion 
          responses from a recorded exponential sine sweep.

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
          gate_start_idx : int
              The sample index where the user-defined time-domain reflection window begins.
          gate_end_idx : int
              The sample index where the user-defined time-domain reflection window ends.
          tukey_alpha : float
                  The shape parameter for the Tukey window applied to each impulse slice.
          mic_cal_file : str or None
              Path to the microphone calibration file, or None if unused.
          mic_sensitivity : float or None
              Microphone sensitivity in milliVolts per Pascal (mV/Pa), or None if unused.
          v_stimulus : float
              The RMS voltage of the playback stimulus.
          distance : float or None
              The measurement distance in meters, or None if unused.
          max_harmonic : int, optional
              The maximum harmonic order to extract and analyze (default is 5).

          Returns:
          --------
          dict
              A dictionary containing frequency response data (frequencies, magnitude in dB, 
              standard phase, minimum phase, and frequency resolution) for each harmonic 
              index `k`, as well as the total harmonic distortion percentage ("thd_percent").
    """ 
    duration = len(inv_filter) / sample_rate
    nyquist = 0.5 * sample_rate
    end_freq = 0.99 * nyquist

    # 1. Perform Fast Deconvolution
    full_ir = scipy.signal.fftconvolve(recorded_audio, inv_filter, mode='full')
    linear_peak_idx = np.argmax(np.abs(full_ir))

    harmonic_results = {}

    # 2. Extract and process each harmonic cleanly
    for k in range(1, max_harmonic + 1):
        if k == 1:
            # Linear fundamental gate anchored relative to the dynamic peak idx
            start_idx = linear_peak_idx + gate_start_offset
            end_idx = linear_peak_idx + gate_end_offset
            segment = full_ir[start_idx:end_idx].copy()
        else:
            # FIXED: Use Ikonen's midpoint-boundary method to automatically and 
            # rigorously scale the extraction window width inversely with order (1/k)
            segment, start_idx = extract_harmonic_segment(
                full_ir, sample_rate, duration, start_freq, end_freq, linear_peak_idx, k
            )

        # FIXED: Pass calibration parameters uniformly across all harmonics
        freqs, mag_db, phase_deg, min_phase_deg, delta_f, raw_fft = compute_frequency_response(
            audio_data=segment,
            sample_rate=sample_rate,
            gate_end_idx=len(segment),
            tukey_alpha=tukey_alpha,
            mic_cal_file=mic_cal_file,
            mic_sensitivity=mic_sensitivity,
            v_stimulus=v_stimulus,
            distance=distance
        )

        # 1. Apply the Phase Matrix Rotation on the raw, uncompressed grid first
        time_delay = (start_idx - linear_peak_idx) / sample_rate
        phase_rotation = np.exp(1j * 2.0 * np.pi * freqs * time_delay)
        aligned_fft = raw_fft * phase_rotation
        
        # 2. Re-extract unwrapped phase maps cleanly from the aligned spectrum
        corrected_phase = np.angle(aligned_fft, deg=True)
        corrected_phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(corrected_phase)))
        corrected_min_phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(min_phase_deg)))

        # 3. Apply Farina Core Adjustments: Compress axis and add octave energy density scale
        corrected_freqs = freqs / k
        corrected_mag_db = mag_db.copy()
        
        if k > 1:
            corrected_mag_db += 20.0 * np.log10(k)

        # 4. Mask out empty high frequency range where the harmonic physically stops
        valid_idx = corrected_freqs <= (sample_rate / (2.0 * k))

        harmonic_results[k] = {
            'freqs': corrected_freqs[valid_idx],
            'mag_db': corrected_mag_db[valid_idx],
            'phase': corrected_phase_unwrapped[valid_idx],
            'min_phase': corrected_min_phase_unwrapped[valid_idx],
            'delta_frequency': delta_f
        }

    # 3. Low-Frequency Boundary Truncation Matrix for THD Calculations
    if 1 in harmonic_results and len(harmonic_results) > 1:
        f_ref = harmonic_results[1]['freqs']
        h1_mag = harmonic_results[1]['mag_db']
        sum_sq_harmonics_linear = np.zeros_like(f_ref)
        
        valid_thd_mask = f_ref >= (2.0 * start_freq)
        
        for k in range(2, max_harmonic + 1):
            if k in harmonic_results:
                # Interpolate the harmonic curves onto the common fundamental frequency grid
                mag_k_interp = np.interp(f_ref, harmonic_results[k]['freqs'], harmonic_results[k]['mag_db'])
                
                # Compute proportional distance and convert to standard linear fraction ratio
                linear_ratio = 10.0 ** ((mag_k_interp - h1_mag) / 20.0)
                
                # FIXED: Mask out high frequencies where this specific harmonic order physically stops existing
                max_harmonic_freq = sample_rate / (2.0 * k)
                linear_ratio[f_ref > max_harmonic_freq] = 0.0
                
                sum_sq_harmonics_linear += linear_ratio ** 2
                
        thd_linear = np.sqrt(sum_sq_harmonics_linear)
        thd_percent = thd_linear * 100.0
        
        thd_percent[~valid_thd_mask] = np.nan
        harmonic_results['thd_percent'] = thd_percent
        
        harmonic_results['freqs'] = f_ref
        harmonic_results['mag_db'] = h1_mag
        harmonic_results['phase'] = harmonic_results[1]['phase']
        harmonic_results['min_phase'] = harmonic_results[1]['min_phase']
        
    return harmonic_results








def read_wav_file(filename):
    """
    Reads a mono WAV file and returns the sample rate and raw audio data.
    """
    sample_rate, data = wav.read(filename)
    
    # Handle stereo files by taking the first channel if necessary
    if data.ndim > 1:
        data = data[:, 0]
        
    return sample_rate, data



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
    """
    try:
        with open(filename, 'w') as f:
            f.write("* Frequency Response Data (FRD)\n")
            f.write("* Columns: Frequency (Hz), Magnitude (dB), Phase (deg)\n")
            
            for freq, mag, phase in zip(freqs, mag_db, phase_deg):
                freq_str = f"{freq:.1f}"
                mag_str = f"{mag:.2f}"
                phase_str = f"{phase:.1f}"
                
                f.write(f"{freq_str}\t{mag_str}\t{phase_str}\n")
                
        print(f"Successfully saved FRD data to {filename}")
    except Exception as e:
        print(f"Error saving FRD file: {e}")




def fractional_octave_smoothing(freqs, mag_db, fraction=3):
    """
    Applies fractional octave smoothing to a frequency response magnitude curve.
    
    Parameters:
    -----------
    freqs : numpy.ndarray
        Frequency array in Hz (must be strictly positive).
    mag_db : numpy.ndarray
        Magnitude array in dB.
    fraction : int or float, optional
        Fractional octave denominator (e.g., 3 for 1/3 octave, 12 for 1/12 octave).
        
    Returns:
    --------
    numpy.ndarray
        Smoothed magnitude array in dB.
    """
    freqs = np.asarray(freqs, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    
    smoothed_mag_db = np.empty_like(mag_db)
    
    # Octave bandwidth factor constant: 2^(1 / (2 * fraction))
    b = 2.0 ** (1.0 / (2.0 * fraction))
    
    for i, f0 in enumerate(freqs):
        if f0 <= 0:
            smoothed_mag_db[i] = mag_db[i]
            continue
            
        # Define lower and upper frequency bounds for the smoothing band
        f_lower = f0 / b
        f_upper = f0 * b
        
        # Find all frequency points falling within the fractional band
        mask = (freqs >= f_lower) & (freqs <= f_upper)
        
        if np.any(mask):
            # Average the dB values within the band
            smoothed_mag_db[i] = np.mean(mag_db[mask])
        else:
            smoothed_mag_db[i] = mag_db[i]
            
    return smoothed_mag_db



def generate_smoothed_results(harmonic_results, smoothing_fraction):
    """
    Takes the raw harmonic_results dictionary and returns a new dictionary 
    with smoothed magnitude and THD curves at the specified fraction.
    """
    if smoothing_fraction is None:
        # If no smoothing is requested, just return a copy of the raw results
        return harmonic_results.copy()
        
    smoothed_results = {}
    
    for k, data in harmonic_results.items():
        if k == 'thd_percent':
            # Smooth the THD curve directly
            smoothed_results['thd_percent'] = fractional_octave_smoothing(
                harmonic_results['freqs'], data, fraction=smoothing_fraction
            )
        elif isinstance(data, dict):
            # It's a harmonic sub-dictionary (e.g., 1, 2, 3...)
            smoothed_results[k] = data.copy()
            if 'mag_db' in data:
                smoothed_results[k]['mag_db'] = fractional_octave_smoothing(
                    data['freqs'], data['mag_db'], fraction=smoothing_fraction
                )
        else:
            # Copy over global keys like 'freqs', 'phase', etc., unchanged
            smoothed_results[k] = data
            
    return smoothed_results



def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Read the raw microphone data and Farina inverse filter
        sample_rate, mic_data = read_wav_file(args.mic_data)
        _, inv_filter = read_wav_file(args.farina_filter)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
    
    # Compute the full impulse response using Farina deconvolution
    print("Computing full impulse response via convolution...")
    full_ir = scipy.signal.fftconvolve(mic_data, inv_filter, mode='full')
    
    total_duration_ms = (len(full_ir) / sample_rate) * 1000.0
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")

    time_axis = np.arange(len(full_ir)) / sample_rate
    
    # Determine y-axis label for magnitude based on whether mic sensitivity is supplied
    if (
        args.mic_sensitivity is not None
        and args.mic_cal is not None
    ):
        mag_ylabel = f"SPL [dB re 20 uPa @ {args.v_stimulus:.2f}V" 
        if args.distance is not None:
            mag_ylabel += "/1m"
    else:
        mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"
    
    # Setup Figure and Subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23)
    
    # Plot Time Domain Impulse Response Data
    ax_time.plot(time_axis * 1000, full_ir, color='b', lw=1, label='IR')
    ax_time.set_title("Impulse Response (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    
    # Initial gate indices
    initial_gate_time_ms = time_axis[-1] * 1000
    initial_gate_idx = len(full_ir)
    initial_start_idx = 0  

    initial_start_time_ms = (initial_start_idx / sample_rate) * 1000.0

    # Add lines for both start and end gates
    start_line = ax_time.axvline(x=initial_start_time_ms, color='orange', linestyle='--', lw=1.5, label='Window Start')
    gate_line  = ax_time.axvline(x=initial_gate_time_ms,  color='g',     linestyle='--', lw=1.5, label='Window End')
    
    # State variables
    state = {
        'gate_idx': initial_gate_idx,            
        'gate_start_idx': initial_start_idx,    
        'selecting_gate_step': 0,               
        'temp_start_time_ms': 0.0,              
        'selecting_gate': False,
        'raw_harmonic_results': None,
        'smoothed_harmonic_results': None,
        'selected_harmonic': 'H1',
        'hd_max': 3,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': False,
        'tukey_alpha': 0.1,
        'resolution': None,
        'smoothing_octaves': 12
    }
    
    initial_win = scipy.signal.get_window(('tukey', state['tukey_alpha']), initial_gate_idx)
    ylim_initial = ax_time.get_ylim()
    initial_y_top = ylim_initial[1] if ylim_initial[1] != 0 else 1.0
    window_line, = ax_time.plot(time_axis[:initial_gate_idx] * 1000, initial_win * (0.95 * initial_y_top), color='r', lw=1.5, label='Tukey Window')
    
    ax_time.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')
    
    # Configure Frequency Domain Plot
    ax_freq.set_title("Frequency Domain Data")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel(mag_ylabel)
    ax_freq.set_xscale('log')
    ax_freq.set_xlim(20, sample_rate / 2.0)
    ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    
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
    ax_box_tukey = plt.axes([0.08, 0.10, 0.05, 0.03])
    textbox_tukey = TextBox(ax_box_tukey, 'TUKEY: ', initial=str(state['tukey_alpha']), textalignment='center')

    # HD Max
    ax_box_hd_max = plt.axes([0.26, 0.10, 0.02, 0.03])
    textbox_hd_max = TextBox(ax_box_hd_max, 'HD MAX: ', initial=str(state['hd_max']), textalignment='center')

    # Smoothing Octaves
    ax_box_sm_octaves = plt.axes([0.52, 0.10, 0.02, 0.03])
    textbox_sm_octaves = TextBox(ax_box_sm_octaves, 'OCTAVE SMOOTHING 1/: ', initial=str(state['smoothing_octaves']), textalignment='center')

    # Phase Type
    ax_box_phase_type = plt.axes([0.65, 0.10, 0.13, 0.03])
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
            win_len = end_idx - start_idx
            win = scipy.signal.get_window(('tukey', state['tukey_alpha']), win_len)
            
            window_line.set_xdata(time_axis[start_idx:end_idx] * 1000)
            window_line.set_ydata(win * (0.95 * y_top))
            
            start_time_ms = (start_idx / sample_rate) * 1000.0
            end_time_ms = (end_idx / sample_rate) * 1000.0
            start_line.set_xdata([start_time_ms, start_time_ms])
            gate_line.set_xdata([end_time_ms, end_time_ms])
            
        fig.canvas.draw_idle()

    ax_time.callbacks.connect('ylim_changed', update_tukey_visual)
    ax_time.callbacks.connect('xlim_changed', update_tukey_visual)

    def submit_tukey_alpha(text):
        try:
            val = float(text)
            if 0.01 <= val <= 0.5:
                state['tukey_alpha'] = val
                print(f"Tukey alpha updated to: {state['tukey_alpha']}")
                update_tukey_visual()
            else:
                print(f"Error: Tukey alpha ({val}) out of bounds. Must be between 0.01 and 0.5.")
                textbox_tukey.set_val(str(state['tukey_alpha']))
        except ValueError:
            textbox_tukey.set_val(str(state['tukey_alpha']))

    textbox_tukey.on_submit(submit_tukey_alpha)

    def submit_hd_max(text):
        try:
            val = int(text)
            if 2 <= val <= 6:
                state['hd_max'] = val
                print(f"HD MAX updated to: {state['hd_max']}")
                state['selected_harmonic'] = 'H1'
                textbox_dataset.set_val('H1')
                if state['smoothed_harmonic_results'] is not None:
                    refresh_frequency_plot()
            else:
                print("Error: HD MAX must be between 2 and 6.")
                textbox_hd_max.set_val(str(state['hd_max']))
        except ValueError:
            textbox_hd_max.set_val(str(state['hd_max']))

    textbox_hd_max.on_submit(submit_hd_max)

    def submit_smoothing(text):
        try:
            # Allow users to turn off smoothing by typing "None", "off", "0", or leaving it blank
            cleaned = text.strip().lower()
            if cleaned in ('none', 'off', '0', ''):
                val = None
            else:
                val = int(text)
                
            # Validate reasonable fractional octave range (e.g., 1 to 48, or None)
            if val is None or (1 <= val <= 48):
                state['smoothing_octaves'] = val
                print(f"Smoothing Octaves updated to: {state['smoothing_octaves']}")
                
                # Recompute the smoothed dataset from the immutable raw results
                if state.get('raw_harmonic_results') is not None:
                    state['smoothed_harmonic_results'] = generate_smoothed_results(
                        state['raw_harmonic_results'], state['smoothing_octaves']
                    )
                    refresh_frequency_plot()
            else:
                print("Error: Smoothing fraction must be between 1 and 48 (or None).")
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
            if clicked_time_ms is not None and clicked_time_ms >= 0:
                clicked_sample = int((clicked_time_ms / 1000.0) * sample_rate)
                
                if state['selecting_gate_step'] == 1:
                    state['temp_start_time_ms'] = clicked_time_ms
                    print(f"Start taper end placed at: {clicked_time_ms:.2f} ms. Now click the Window End (Reflection limit).")
                    state['selecting_gate_step'] = 2
                    
                elif state['selecting_gate_step'] == 2:
                    clicked_onset_sample = int((state['temp_start_time_ms'] / 1000.0) * sample_rate)
                    end_sample = clicked_sample
                    
                    if end_sample > clicked_onset_sample:
                        temp_window_len = end_sample - clicked_onset_sample
                        taper_len = int(temp_window_len * state['tukey_alpha'] / 2.0)
                        true_start_sample = max(0, clicked_onset_sample - taper_len)
                        
                        state['gate_start_idx'] = true_start_sample
                        state['gate_idx'] = end_sample
                        
                        actual_start_ms = (true_start_sample / sample_rate) * 1000.0
                        clicked_onset_ms = state['temp_start_time_ms']
                        actual_end_ms = (end_sample / sample_rate) * 1000.0
                        
                        print(f"Window Updated!")
                        print(f"  - Taper starts at: {actual_start_ms:.2f} ms")
                        print(f"  - Taper ends (Signal Onset): {clicked_onset_ms:.2f} ms")
                        print(f"  - Window ends: {actual_end_ms:.2f} ms")
                        
                        start_line.set_xdata([actual_start_ms, actual_start_ms])
                        gate_line.set_xdata([actual_end_ms, actual_end_ms])
                        update_tukey_visual()
                    else:
                        print("Error: Window end must be after the window start. Try again.")
                    
                    state['selecting_gate_step'] = 0
                    btn_gate.label.set_color('black')
                    ax_btn_gate.set_facecolor('aliceblue')
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click_time_plot)

    def handle_gate_button(event):
        if state['selecting_gate_step'] == 0:
            state['selecting_gate_step'] = 1
            btn_gate.label.set_color('red')
            ax_btn_gate.set_facecolor('#ffcccc')
            print("Step 1/2: Click the time plot where the signal leaves the baseline (Start taper end).")
        else:
            state['selecting_gate_step'] = 0
            btn_gate.label.set_color('black')
            ax_btn_gate.set_facecolor('aliceblue')
            print("Gate selection cancelled.")
        fig.canvas.draw_idle()
        
    def refresh_frequency_plot():
        if state['smoothed_harmonic_results'] is None:
            return

        # 1. Check if we are viewing the exact same plot type as last time
        current_sel = state['selected_harmonic']
        current_phase = state['showing_phase']
        
        same_view = (
            state.get('_last_sel') == current_sel and 
            state.get('_last_phase') == current_phase
        )

        # 2. Capture current axis limits before clearing
        try:
            old_xlim = ax_freq.get_xlim()
            old_ylim = ax_freq.get_ylim()
        except Exception:
            old_xlim = None
            old_ylim = None

        # 3. Clear and set up base properties
        ax_freq.clear()
        ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_freq.set_xscale('log')
        ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

        results = state['smoothed_harmonic_results']
        sel = current_sel

        if current_phase:
            if sel == 'ALL' or sel == 'THD':
                ax_freq.set_ylabel("Phase (degrees)")
                ax_freq.set_title("Phase View (Blank for ALL/THD)")
                state['freq_line_plot'] = []
            else:
                k_idx = int(sel[1]) if len(sel) > 1 else 1
                if k_idx <= state['hd_max']:
                    current_phase_data = results[k_idx]['min_phase'] if state['using_min_phase'] else results[k_idx]['phase']
                    title_label = f"Harmonic {k_idx} - {'Minimum Phase' if state['using_min_phase'] else 'Phase'}"
                    line, = ax_freq.plot(results[k_idx]['freqs'], current_phase_data, color='g', lw=1.2)
                    ax_freq.set_ylabel("Phase (degrees)")
                    ax_freq.set_title(title_label)
                    ax_freq.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')
                    state['freq_line_plot'] = [line]
        else:
            if sel == 'ALL':
                freqs = results[1]['freqs']
                line1, = ax_freq.plot(freqs, results[1]['mag_db'], color='r', lw=1.2, label='H1 (Fund)')
                lines = [line1]
                colors = ['orange', 'purple', 'brown', 'pink', 'gray']
                for idx, k in enumerate(range(2, state['hd_max'] + 1)):
                    if k in results:
                        l, = ax_freq.plot(results[k]['freqs'], results[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='--', label=f'H{k}')
                        lines.append(l)
                ax_freq.legend(loc='upper right')
                ax_freq.set_ylabel(mag_ylabel)
                ax_freq.set_title("Frequency Domain Data - All Harmonics Magnitude")
                state['freq_line_plot'] = lines
            elif sel == 'THD':
                freqs = results[1]['freqs']
                line, = ax_freq.plot(freqs, results['thd_percent'], color='blue', lw=1.2)
                ax_freq.set_ylabel("THD (%)")
                ax_freq.set_title("Total Harmonic Distortion (THD)")
                state['freq_line_plot'] = [line]
            else:
                k_idx = int(sel[1]) if len(sel) > 1 else 1
                if k_idx <= state['hd_max']:
                    line, = ax_freq.plot(results[k_idx]['freqs'], results[k_idx]['mag_db'], color='r', lw=1.2)
                    ax_freq.set_ylabel(mag_ylabel)
                    ax_freq.set_title(f"Frequency Domain Data - Magnitude (Harmonic {k_idx})")
                    state['freq_line_plot'] = [line]

            ax_freq.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')

        # 4. Handle X-axis limits (preserve user frequency zoom if customized)
        if old_xlim and old_xlim != (0.0, 1.0):
            ax_freq.set_xlim(old_xlim)
        else:
            ax_freq.set_xlim(20, sample_rate / 2.0)

        # 5. Handle Y-axis limits intelligently:
        # Only preserve Y limits if we are looking at the exact same plot view (e.g., smoothing change).
        # If switching plots (e.g. Magnitude -> THD), auto-scale Y to fit the new data!
        if same_view and old_ylim and old_ylim != (0.0, 1.0):
            ax_freq.set_ylim(old_ylim)
        else:
            ax_freq.relim()
            ax_freq.autoscale_view(scalex=False, scaley=True)

        # 6. Store current view signature for the next call
        state['_last_sel'] = current_sel
        state['_last_phase'] = current_phase

        # 7. Force the GUI canvas to re-render the updated plot
        fig.canvas.draw_idle()

    def handle_calc_button(event):
        if state['smoothed_harmonic_results'] is not None:
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            ax_freq.set_ylabel(mag_ylabel)
            ax_freq.set_title("Frequency Domain Data (Cleared)")
            ax_freq.set_xlabel('Frequency (Hz)')
            fig.canvas.draw_idle()
            state['smoothed_harmonic_results'] = None
            state['resolution'] = None
            print("Lower plot cleared. All calculated data cleared.")
        else:
            print(f"Computing Farina Method response up to HD {state['hd_max']}...")
            
            linear_peak_idx = np.argmax(np.abs(full_ir))
            gate_start_offset = state['gate_start_idx'] - linear_peak_idx
            gate_end_offset = state['gate_idx'] - linear_peak_idx
            
            results = run_farina_analysis(
                recorded_audio=mic_data,
                sample_rate=sample_rate,
                inv_filter=inv_filter,
                start_freq=args.start_f,
                gate_start_offset=gate_start_offset,
                gate_end_offset=gate_end_offset,
                tukey_alpha=state['tukey_alpha'],
                mic_cal_file=args.mic_cal,
                mic_sensitivity=args.mic_sensitivity,
                v_stimulus=args.v_stimulus,
                distance=args.distance,
                max_harmonic=state['hd_max']
            )
            state['raw_harmonic_results'] = results
            
            octave_smoothing = 12
            state['smoothed_harmonic_results'] = generate_smoothed_results(results, octave_smoothing)
            
            window_duration = (state['gate_idx'] - state['gate_start_idx']) / sample_rate
            state['resolution'] = 1.0 / window_duration
            
            refresh_frequency_plot()
            print(f"Analysis complete! Resolution: {state['resolution']:.3g} Hz")

    def handle_view_button(event):
        if state['smoothed_harmonic_results'] is not None:
            state['showing_phase'] = not state['showing_phase']
            refresh_frequency_plot()
            update_phase_indicator()
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
        if state['smoothed_harmonic_results'] is not None:
            refresh_frequency_plot()

    def handle_cycle_right(event):
        h_list = get_harmonic_list()
        try:
            curr_idx = h_list.index(state['selected_harmonic'])
        except ValueError:
            curr_idx = 0
        new_idx = (curr_idx + 1) % len(h_list)
        state['selected_harmonic'] = h_list[new_idx]
        textbox_dataset.set_val(state['selected_harmonic'])
        if state['smoothed_harmonic_results'] is not None:
            refresh_frequency_plot()

    def handle_save_exit_button(event):
        if state['smoothed_harmonic_results'] is None:
            print("Computing response before saving...")
            
            linear_peak_idx = np.argmax(np.abs(full_ir))
            gate_start_offset = state['gate_start_idx'] - linear_peak_idx
            gate_end_offset = state['gate_idx'] - linear_peak_idx
            
            results = run_farina_analysis(
                recorded_audio=mic_data,
                sample_rate=sample_rate,
                inv_filter=inv_filter,
                start_freq=args.start_f,
                gate_start_offset=gate_start_offset,
                gate_end_offset=gate_end_offset,
                tukey_alpha=state['tukey_alpha'],
                mic_cal_file=args.mic_cal,
                mic_sensitivity=args.mic_sensitivity,
                v_stimulus=args.v_stimulus,
                distance=args.distance,
                max_harmonic=state['hd_max']
            )
            state['raw_harmonic_results'] = results
            
            octave_smoothing = 12
            state['smoothed_harmonic_results'] = generate_smoothed_results(results, octave_smoothing)
             
            window_duration = (state['gate_idx'] - state['gate_start_idx']) / sample_rate
            state['resolution'] = 1.0 / window_duration

        res = state['smoothed_harmonic_results']
        base_name = args.output  
        
        for k in range(1, state['hd_max'] + 1):
            if k in res:
                h_data = res[k]
                phase_to_save = h_data['min_phase'] if state['using_min_phase'] else h_data['phase']
                filename_k = f"{base_name}_HD{k}.frd"
                write_frd_file(filename_k, h_data['freqs'], h_data['mag_db'], phase_to_save)

        print("Saving plots at 600 DPI in png file format...")
        
        # Plot A: Fundamental Magnitude (_FREQ.png)
        fig_freq, ax_f = plt.subplots(figsize=(8, 5))
        ax_f.plot(res[1]['freqs'], res[1]['mag_db'], color='r', lw=1.2)
        ax_f.set_title("Fundamental Frequency Response - Magnitude")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel(mag_ylabel)
        ax_f.set_xscale('log')
        ax_f.set_xlim(20, sample_rate / 2.0)
        ax_f.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig_freq.savefig(f"{base_name}_FREQ.png", dpi=600, bbox_inches='tight')
        plt.close(fig_freq)

        # Plot B: Fundamental Phase (_PHASE.png)
        fig_phase, ax_p = plt.subplots(figsize=(8, 5))
        phase_to_plot = res[1]['min_phase'] if state['using_min_phase'] else res[1]['phase']
        phase_title = "Fundamental Minimum Phase" if state['using_min_phase'] else "Fundamental Phase"
        ax_p.plot(res[1]['freqs'], phase_to_plot, color='g', lw=1.2)
        ax_p.set_title(phase_title)
        ax_p.set_xlabel("Frequency (Hz)")
        ax_p.set_ylabel("Phase (degrees)")
        ax_p.set_xscale('log')
        ax_p.set_xlim(20, sample_rate / 2.0)
        ax_p.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig_phase.savefig(f"{base_name}_PHASE.png", dpi=600, bbox_inches='tight')
        plt.close(fig_phase)

        # Plot C: All Harmonics Magnitude (_HD.png)
        fig_hd, ax_h = plt.subplots(figsize=(8, 5))
        freqs = res[1]['freqs']
        ax_h.plot(freqs, res[1]['mag_db'], color='r', lw=1.2, label='H1 (Fund)')
        colors = ['orange', 'purple', 'brown', 'pink', 'gray']
        for idx, k in enumerate(range(2, state['hd_max'] + 1)):
            if k in res:
                ax_h.plot(res[k]['freqs'], res[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='--', label=f'H{k}')
        ax_h.legend(loc='upper right')
        ax_h.set_title("Harmonic Distortion - All Harmonics")
        ax_h.set_xlabel("Frequency (Hz)")
        ax_h.set_ylabel(mag_ylabel)
        ax_h.set_xscale('log')
        ax_h.set_xlim(20, sample_rate / 2.0)
        ax_h.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig_hd.savefig(f"{base_name}_HD.png", dpi=600, bbox_inches='tight')
        plt.close(fig_hd)

        # Plot D: Total Harmonic Distortion (_THD.png)
        fig_thd, ax_t = plt.subplots(figsize=(8, 5))
        ax_t.plot(res[1]['freqs'], res['thd_percent'], color='blue', lw=1.2)
        ax_t.set_title("Total Harmonic Distortion (THD)")
        ax_t.set_xlabel("Frequency (Hz)")
        ax_t.set_ylabel("THD (%)")
        ax_t.set_xscale('log')
        ax_t.set_xlim(20, sample_rate / 2.0)
        ax_t.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig_thd.savefig(f"{base_name}_THD.png", dpi=600, bbox_inches='tight')
        plt.close(fig_thd)

        print("All data files and plots successfully saved.")
        print("Closing application.")
        plt.close('all')
        sys.exit(0)

    btn_gate.on_clicked(handle_gate_button)
    btn_calc.on_clicked(handle_calc_button)
    btn_view.on_clicked(handle_view_button)
    btn_min.on_clicked(handle_min_phase_button)
    btn_save_exit.on_clicked(handle_save_exit_button)
    btn_prev.on_clicked(handle_cycle_left)
    btn_next.on_clicked(handle_cycle_right)
    
    plt.show()


if __name__ == "__main__":
    main()

    