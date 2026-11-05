"""
H1_Time2Freq - Convert the Time Domain Response to the Frequency Domain Response 
Description: Calculates the time domain frequency response from a windowed 
    impulse response, plus correction factors. Can produce relative SPL levels 
    or in units of dB re 20 uPa @ 2.83V/1m. Both the regular phase and minimum 
    phase responses are also made available.

Author: Charlie Laub
Date: October 2026
"""

import sys
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.ticker import LogLocator
import argparse, textwrap
import struct
from datetime import datetime



def parse_inline_config(arg_str):
    """
    Parses a string like 'V1=6.15, V2=2.45, Vg=20, Smic=0.01, d=0.25' 
    into a dictionary of floats.
    """
    config = {}
    if not arg_str:
        return config
        
    for pair in arg_str.split(','):
        if '=' not in pair:
            raise argparse.ArgumentTypeError(f"Invalid parameter format: '{pair}'. Use key=value.")
        key, val = pair.split('=', 1)
        key = key.strip()
        try:
            config[key] = float(val.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value for '{key}' must be a number, got '{val}'")
            
    return config



def parse_arguments():
    """
    Parses command-line arguments for input WAV file, output FRD file,
    optional microphone calibration, sensitivity parameters, and amplifier gain.
    """
    parser = argparse.ArgumentParser(description="Process loudspeaker impulse response WAV file and export to FRD format (Interactive GUI).")
    parser.add_argument("input_wav", 
            type=str, 
            help="Path to the input impulse response WAV file."
    )
    parser.add_argument("-o", "--output", 
            type=str, 
            default="FR.frd",
            help="Path to save the output in FRD format (default: FR.frd)."
    )
    parser.add_argument("-s","--smoothing", 
            type=int, 
            default=24,
            help="Response Smoothing in Points per Octave, valid entries are 1..48 (default: 24)."
    )
    parser.add_argument("-c","--mic_cal_file", 
            type=str, 
            default=None,
            help="Path to microphone calibration file (*.mic or *.cal)."
    )
    # Compact inline configuration string
    parser.add_argument("-k","--spl_corrections", 
            type=parse_inline_config,
            default={},
            help=textwrap.dedent("""\
        Comma-delimited quoted string of the five parameters used to set the SPL level.
            
        Example: -k "V1=6.15, V2=2.45, Ag=20.0, Smic=10.0, d=0.25"
            
        V1   : Channel 1 (loopback) input sensitivity, Volts to reach 0dBFS (+/-1.0) on the audio interface
        V2   : Channel 2 (microphone) input sensitivity, Volts to reach 0dBFS (+/-1.0) on the audio interface
        Ag   : Amplifier voltage gain, Volts at speaker terminals per Volt applied to the input
        Smic : Microphone sensitivity in mV/Pa
        d    : The distance in meters between the loudspeaker and microphone during the measurement.
            """).strip()
    )
    return parser.parse_args()



def read_wav_file_plus_info(filename):
    """
    Reads a 32-bit float mono WAV file and extracts the audio data (as a NumPy array),
    the sample rate, and custom xDAT metadata into a dictionary.
    Optimized with NumPy frombuffer for high speed and low memory usage.
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
        audio_data = np.array([], dtype=np.float32)
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
                audio_format, num_channels, sample_rate = struct.unpack(
                    "<HHI", fmt_data[:8]
                )
                if audio_format != 3:  # 3 indicates IEEE float format
                    raise ValueError(
                        "WAV file is not in 32-bit IEEE float format (format tag 3)."
                    )

            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
                # Zero-copy memory mapping via NumPy frombuffer
                audio_data = np.frombuffer(data_bytes, dtype=np.float32).copy()

            elif chunk_id == b"xDAT":
                xdat_data = f.read(chunk_size)
                offset = 0
                while offset < chunk_size:
                    tag = xdat_data[offset : offset + 4].decode("utf-8", errors="ignore")
                    offset += 4

                    if tag in ("date", "reps"):
                        val = struct.unpack("<I", xdat_data[offset : offset + 4])[0]
                    else:
                        val = struct.unpack("<f", xdat_data[offset : offset + 4])[0]
                    offset += 4

                    custom_metadata[tag] = val
            else:
                # Skip unknown chunks
                f.seek(chunk_size, 1)

            # Handle RIFF word-alignment padding
            if chunk_size % 2 != 0:
                f.seek(1, 1)

    return sample_rate, audio_data, custom_metadata



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




def compute_frequency_response(
    audio_data,
    sample_rate,
    gate_end_idx,
    tukey_alpha,
    start_freq,
    end_freq,
    Fgrid_density
):
    """Applies a Tukey window, computes standard FFT and minimum phase FFT,
    and interpolates the response to Fgrid_density points per octave.
    Returns frequencies, magnitude (dB), standard phase (deg), minimum phase (deg), 
    and delta_frequency (Hz).
    """
    gate_end_idx = max(10, min(gate_end_idx, len(audio_data)))

    # Get Tukey window using scipy.signal.get_window
    win = scipy.signal.get_window(('tukey', tukey_alpha), gate_end_idx)

    # Restrict windowed_data strictly to the gated segment length
    windowed_data = audio_data[:gate_end_idx].astype(np.float32) * win

    # Zero-pad for smoother frequency resolution in the FFT
    n_fft = max(2048, 2 * len(windowed_data))

    # Standard FFT response
    fft_result = np.fft.rfft(windowed_data, n=n_fft)
    
    # 1. Get raw magnitude (No division by sample_rate or window sum needed!)
    mag = np.abs(fft_result)
    
    # 2. Calculate dB
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    
    # UNWRAP PHASE BEFORE INTERPOLATION
    raw_phase_rad = np.angle(fft_result)
    phase_deg = np.degrees(np.unwrap(raw_phase_rad))

    # Compute Minimum Phase response on the active windowed data
    try:
        min_phase_signal = scipy.signal.minimum_phase(
            windowed_data, method='homomorphic', half=False
        )
        min_fft_result = np.fft.rfft(min_phase_signal, n=n_fft)
        
        # UNWRAP MIN PHASE BEFORE INTERPOLATION
        min_phase_raw_rad = np.angle(min_fft_result)
        min_phase_deg = np.degrees(np.unwrap(min_phase_raw_rad))
    except Exception:
        min_phase_deg = phase_deg.copy()

    # --- Frequency Resolution Calculation via NCC & IFFT ---
    try:
        # 1. Generate minimum phase time response using amplitude and minimum phase
        min_phase_rad = np.deg2rad(min_phase_deg)
        min_complex_spectrum = mag * np.exp(1j * min_phase_rad)
        impulse_minphase = np.fft.irfft(min_complex_spectrum, n=n_fft)

        # Truncate impulse_minphase back to the window length for accurate cross-correlation
        impulse_minphase_trimmed = impulse_minphase[: len(windowed_data)]

        # 2. Compute Normalized Cross-Correlation (NCC) between the gated data and min-phase impulse
        f = windowed_data - np.mean(windowed_data)
        g = impulse_minphase_trimmed - np.mean(impulse_minphase_trimmed)
        xcorr = scipy.signal.correlate(f, g, mode='full')
        norm = np.sqrt(np.sum(f**2) * np.sum(g**2))
        if norm > 0:
            xcorr /= norm

        # 3. Find peak position, convert to time, and calculate delta frequency
        ncc_max_pos = np.argmax(xcorr)
        # In 'full' mode, the zero-lag/start alignment offset is len(g) - 1
        peak_index_offset = abs(ncc_max_pos - (len(g) - 1))
        
        ncc_max_time = float(peak_index_offset) / sample_rate
        gate_end_time = float(gate_end_idx) / sample_rate

        time_diff = gate_end_time - ncc_max_time
        delta_frequency = (
            float(1.0 / time_diff) if time_diff > 0 else float(sample_rate / n_fft)
        )
    except Exception:
        delta_frequency = float(sample_rate / n_fft)

    # --- Format output according to frequency grid preference ---
    if Fgrid_density > 0:
        # Pre-calculate logarithmic frequency grid
        octaves = np.log2(end_freq / start_freq)
        num_points = int(np.ceil(octaves * Fgrid_density))
        log_freqs = start_freq * (2.0 ** (np.arange(num_points) / Fgrid_density))
        log_freqs = log_freqs[log_freqs <= end_freq]

        valid_mask = (freqs >= start_freq) & (freqs <= end_freq)
        if np.sum(valid_mask) > 1:
            f_interp = freqs[valid_mask]
            m_interp = mag_db[valid_mask]
            p_interp = phase_deg[valid_mask]
            mp_interp = min_phase_deg[valid_mask]
            
            res_freqs = log_freqs
            res_mag = np.interp(res_freqs, f_interp, m_interp, left=m_interp[0], right=m_interp[-1])
            res_phase = np.interp(res_freqs, f_interp, p_interp, left=p_interp[0], right=p_interp[-1])
            res_min_phase = np.interp(res_freqs, f_interp, mp_interp, left=mp_interp[0], right=mp_interp[-1])
        else:
            res_freqs = log_freqs
            res_mag = np.zeros_like(log_freqs)
            res_phase = np.zeros_like(log_freqs)
            res_min_phase = np.zeros_like(log_freqs)
        
        return res_freqs, res_mag, res_phase, res_min_phase, delta_frequency
    else:
        # Standard un-interpolated return
        valid_idx = freqs <= (sample_rate / 2.0)
        return (
            freqs[valid_idx],
            mag_db[valid_idx],
            phase_deg[valid_idx],
            min_phase_deg[valid_idx],
            delta_frequency,
        )



class MicCalibrationHandler:
    """
    Loads a microphone calibration file once, computes trend lines for extrapolation,
    and builds an Akima spline interpolator for efficient repeated evaluations.
    """
    def __init__(self, filename):
        self.filename = filename
        self.is_valid = False
        self.f_min = None
        self.f_max = None
        self.interpolator = None
        self.poly_coeff_low = None
        self.poly_coeff_high = None
        self._load_and_setup()

    def _load_and_setup(self):
        cal_freqs = []
        cal_db = []
        try:
            with open(self.filename, 'r') as f:
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
                return
                
            cal_freqs = np.array(cal_freqs)
            cal_db = np.array(cal_db)
            
            # Sort calibration data by frequency
            sort_indices = np.argsort(cal_freqs)
            cal_freqs = cal_freqs[sort_indices]
            cal_db = cal_db[sort_indices]
            
            self.f_min = cal_freqs[0]
            self.f_max = cal_freqs[-1]
            
            # Base Akima interpolator for the interior region (extrapolation disabled here)
            self.interpolator = Akima1DInterpolator(cal_freqs, cal_db, extrapolate=False)
            
            # --- 1. Low-Frequency Trend Fit (< f_min) ---
            low_data_mask = cal_freqs <= 100.0
            if np.sum(low_data_mask) >= 2:
                fit_freqs = cal_freqs[low_data_mask]
                fit_db = cal_db[low_data_mask]
            else:
                n_pts = min(3, len(cal_freqs))
                fit_freqs = cal_freqs[:n_pts]
                fit_db = cal_db[:n_pts]
            self.poly_coeff_low = np.polyfit(np.log10(fit_freqs), fit_db, 1)
            
            # --- 2. High-Frequency Trend Fit (> f_max) with 3-point moving average ---
            hf_data_mask = cal_freqs >= 10000.0
            if np.sum(hf_data_mask) >= 3:
                fit_freqs = cal_freqs[hf_data_mask]
                fit_db = cal_db[hf_data_mask]
                
                window_size = 3
                weights = np.ones(window_size) / window_size
                smoothed_db = np.convolve(fit_db, weights, mode='valid')
                smoothed_freqs = np.convolve(fit_freqs, weights, mode='valid')
            elif np.sum(hf_data_mask) >= 1:
                smoothed_freqs = cal_freqs[hf_data_mask]
                smoothed_db = cal_db[hf_data_mask]
            else:
                n_pts = min(3, len(cal_freqs))
                smoothed_freqs = cal_freqs[-n_pts:]
                smoothed_db = cal_db[-n_pts:]
                
            self.poly_coeff_high = np.polyfit(np.log10(smoothed_freqs), smoothed_db, 1)
            self.is_valid = True
            
        except Exception as e:
            print(f"Error reading microphone calibration file: {e}")
            self.is_valid = False

    def get_corrections(self, freqs):
        """
        Evaluates the pre-calculated Akima spline and extrapolation models for a given frequency array.
        """
        if not self.is_valid:
            return np.zeros_like(freqs)
            
        corrections = np.zeros_like(freqs, dtype=float)
        
        # 1. Low-Frequency Extrapolation (< f_min)
        low_mask = freqs < self.f_min
        if np.any(low_mask) and self.poly_coeff_low is not None:
            corrections[low_mask] = np.polyval(self.poly_coeff_low, np.log10(freqs[low_mask]))
        
        # 2. High-Frequency Extrapolation (> f_max)
        high_mask = freqs > self.f_max
        if np.any(high_mask) and self.poly_coeff_high is not None:
            corrections[high_mask] = np.polyval(self.poly_coeff_high, np.log10(freqs[high_mask]))
            
        # 3. Core Interior Range (f_min <= freqs <= f_max)
        mid_mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        if np.any(mid_mask):
            corrections[mid_mask] = self.interpolator(freqs[mid_mask])
            
        return corrections



def main():
    # Parse command-line arguments
    args = parse_arguments()
    wav_filename = args.input_wav
    output_filename = args.output
    
    try:
        sample_rate, audio_data, impulse_metadata = read_wav_file_plus_info(wav_filename)
    except FileNotFoundError:
        print(f"Error: WAV file '{wav_filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # Extract parameters with fallback defaults if a key wasn't specified in the string
    v_in1_max = args.spl_corrections.get('V1', 0.0)
    v_in2_max = args.spl_corrections.get('V2', 0.0)
    amp_gain  = args.spl_corrections.get('Ag', 0.0)
    mic_sens  = args.spl_corrections.get('Smic', 0.0)
    distance  = args.spl_corrections.get('d', 0.0)
    
    if args.mic_cal_file is not None:
        print(f"Using mic cal file: {args.mic_cal_file}")

    measurement_start_freq = impulse_metadata.get("f0sw", 0.0)
    HF_window_startF = impulse_metadata.get("fhfw", 0.0)
    measurement_timestamp = datetime.fromtimestamp(impulse_metadata.get("date", 0))
    RMS_internal_level = impulse_metadata.get("levl",0.0)
    # Print out validated parameters
    print(f"\n------------------------ Measurement Parameters: -------------------------------")
    print(f"             Start Frequency (f0sw): {impulse_metadata.get("f0sw", 0.0)} Hz")
    print(f"               End Frequency (f1sw): {impulse_metadata.get("f1sw", 0.0)} Hz")
    print(f"                  Chirp Rate (rchr): {impulse_metadata.get("rchr", 0.0)} seconds/decade Hz")
    print(f"                 Repetitions (reps): {impulse_metadata.get("reps", 1)}")
    print(f"              Internal Level (levl): {impulse_metadata.get("levl", 0.0):.3e}")
    print(f"             HF Window Start (fhfw): {impulse_metadata.get("fhfw", 0.0)} Hz")
    print(f"                   Timestamp (date): {measurement_timestamp.strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"--------------------------------------------------------------------------------\n")

    
    # State variables
    state = {
        'gate_idx': None,
        'selecting_gate': False,
        'freqs': None,
        'mag_db': None,
        'phase_deg': None,
        'min_phase_deg': None,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': False,
        'tukey_alpha': 0.25,
        'resolution': None,
        'SPL_factor': None,
        'start_freq': measurement_start_freq,
        'end_freq': HF_window_startF
    }
    
    total_duration_ms = (len(audio_data) / sample_rate) * 1000.0
    print(f"Processing WAV: {args.input_wav}")
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")

    time_axis = np.arange(len(audio_data)) / sample_rate

    
    # Determine y-axis label for magnitude based on whether spl correction factor was supplied
    if all(v > 0 for v in (v_in1_max, v_in2_max, amp_gain, mic_sens, distance, RMS_internal_level)) and args.mic_cal_file is not None:
        mag_ylabel = f"SPL [dB re 20 uPa @ 2.83V/1m" 
        # Calculate the SPL correction factor, Vdrive = RMS_internal_level * v_in1_max
        SPL_factor = distance / ( 0.001 * mic_sens ) 
        SPL_factor *= ( v_in2_max / v_in1_max )
        SPL_factor *= 2.83
        SPL_factor /= ( RMS_internal_level * v_in1_max * amp_gain )
        SPL_factor *= 50000.0
        state['SPL_factor'] = SPL_factor
        print(f"The SPL conversion factor is {SPL_factor:.3e}\n")
    else:
        mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"

    
    # Initialize the microphone calibration handler
    mic_cal_handler = None
    if args.mic_cal_file is not None:
        mic_cal_handler = MicCalibrationHandler(args.mic_cal_file)
        print(f"Using the microphone calibration data from {args.mic_cal_file}.\n")

    print(f"The data will be calculated using {args.smoothing} points per octave.\n")
    
    # Setup Figure and Subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23)
    
    # 1. Plot Time Domain Data
    ax_time.plot(time_axis * 1000, audio_data, color='b', lw=1, label='IR')
    ax_time.set_title(f"Impulse Response: {wav_filename} (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    ax_time.set_xlim(-0.1, 100.0)
    
    # Initial gate line position set to the end of the data (Green)
    initial_gate_time_ms = 50.0
    initial_gate_idx = int( len(audio_data) * initial_gate_time_ms / total_duration_ms )
    gate_line = ax_time.axvline(x=initial_gate_time_ms, color='g', linestyle='--', lw=1.5, label='Window End')
    state['gate_idx'] = initial_gate_idx
    
    # Plot Tukey window function as a red line initially scaled to 95% of current y max
    initial_win = scipy.signal.get_window(('tukey', state['tukey_alpha']), initial_gate_idx)
    ylim_initial = ax_time.get_ylim()
    initial_y_top = ylim_initial[1] if ylim_initial[1] != 0 else 1.0
    window_line, = ax_time.plot(time_axis[:initial_gate_idx] * 1000, initial_win * (0.95 * initial_y_top), color='r', lw=1.5, label='Tukey Window')
    
    ax_time.legend(loc='lower right',
        framealpha=1.0,  # Makes the background fully opaque (1.0) instead of semi-transparent
        facecolor='white',  # Sets the background color of the legend box explicitly
        edgecolor='black',  # Sets the border color of the legend box
    )
    
    # 2. Configure Frequency Domain Plot with automatic scaling and gridlines
    ax_freq.set_title("Frequency Domain Data")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel(mag_ylabel)
    ax_freq.set_xscale('log')
    ax_freq.set_xlim(20, sample_rate / 2.0)
    ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Configure X-axis minor and major gridlines for log scale
    ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    
    # Define Button Axes [left, bottom, width, height] - Evenly distributed, height set to 0.05
    btn_y = 0.03
    btn_h = 0.04
    
    ax_btn_gate       = plt.axes([0.03,  btn_y, 0.14, btn_h])
    ax_btn_calc       = plt.axes([0.19,  btn_y, 0.16, btn_h])
    ax_btn_view       = plt.axes([0.37,  btn_y, 0.18, btn_h])
    ax_btn_min        = plt.axes([0.57,  btn_y, 0.22, btn_h])
    ax_btn_save_exit  = plt.axes([0.81,  btn_y, 0.15, btn_h])
    
    btn_gate = Button(ax_btn_gate, 'Place Gate', color='aliceblue', hovercolor='palegreen')
    btn_calc = Button(ax_btn_calc, 'Compute/Clear', color='aliceblue', hovercolor='palegreen')
    btn_view = Button(ax_btn_view, 'Toggle Mag/Phase', color='aliceblue', hovercolor='palegreen')
    btn_min  = Button(ax_btn_min, 'Toggle Phase-MinPhase', color='aliceblue', hovercolor='palegreen')
    btn_save_exit = Button(ax_btn_save_exit, 'Save & Exit', color='aliceblue', hovercolor='palegreen')
    
    # Define Tukey Alpha Text Box Axes above Place Gate & Compute buttons
    ax_box_tukey = plt.axes([0.21, 0.10, 0.06, 0.03])
    textbox_tukey = TextBox(ax_box_tukey, 'TUKEY WINDOW WIDTH:  ', initial=str(state['tukey_alpha']),textalignment='center')

    #Create text box to display the current phase type
    # Adjust coordinates [left, bottom, width, height] to place it just above toggle Phase-Minphase button
    ax_box_phase_type = plt.axes([0.66, 0.10, 0.13, 0.03])
    # The label argument automatically places "PHASE TYPE:" to the left of the box
    initial_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
    phase_textbox = TextBox(
        ax_box_phase_type, 'PHASE TYPE:  ', initial=initial_text, textalignment='center'
    )
    # Disable user editing since this is purely a status indicator
    phase_textbox.set_active(False)


    def update_tukey_visual(event=None):
        """Redraws the Tukey window line to stay within view and reach 95% of y-axis max."""
        ylim = ax_time.get_ylim()
        y_top = ylim[1]
        win = scipy.signal.get_window(('tukey', state['tukey_alpha']), state['gate_idx'])
        window_line.set_xdata(time_axis[:state['gate_idx']] * 1000)
        window_line.set_ydata(win * (0.95 * y_top))
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
            print("Error: Invalid numeric value entered for Tukey alpha.")
            textbox_tukey.set_val(str(state['tukey_alpha']))

    textbox_tukey.on_submit(submit_tukey_alpha)

    # Event Handlers & Button Callbacks
    def on_click_time_plot(event):
        if state['selecting_gate'] and event.inaxes == ax_time:
            clicked_time_ms = event.xdata
            if clicked_time_ms is not None and clicked_time_ms >= 0:
                state['gate_idx'] = int((clicked_time_ms / 1000.0) * sample_rate)
                state['gate_idx'] = max(10, min(state['gate_idx'], len(audio_data)))
                
                actual_time_ms = (state['gate_idx'] / sample_rate) * 1000.0
                gate_line.set_xdata([actual_time_ms, actual_time_ms])
                
                update_tukey_visual()
                print(f"Window end placed at: {actual_time_ms:.2f} ms (sample {state['gate_idx']})")
                
            state['selecting_gate'] = False
            btn_gate.label.set_color('black')
            ax_btn_gate.set_facecolor('aliceblue')
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click_time_plot)
    
    def handle_gate_button(event):
        state['selecting_gate'] = not state['selecting_gate']
        if state['selecting_gate']:
            btn_gate.label.set_color('red')
            ax_btn_gate.set_facecolor('#ffcccc')
            print("Click on the time domain plot to place the window end line.")
        else:
            btn_gate.label.set_color('black')
            ax_btn_gate.set_facecolor('aliceblue')
        fig.canvas.draw_idle()
        
    def handle_calc_button(event):
        if state['freq_line_plot'] is not None:
            # Clear lower plot
            for artist in state['freq_line_plot']:
                artist.remove()
            state['freq_line_plot'] = None
            ax_freq.relim()
            ax_freq.autoscale_view()
            ax_freq.set_ylabel(mag_ylabel)
            ax_freq.set_title("Frequency Domain Data (Cleared)")
            ax_freq.set_xlabel(f'Frequency (Hz)')
            fig.canvas.draw_idle()
            # Clear calculated data
            state['freqs'] = None
            state['mag_db'] = None
            state['phase_deg'] = None
            state['min_phase_deg'] = None
            state['resolution'] = None
            print("Lower plot cleared. All calculated data cleared.")
        else:
            # Compute and plot frequency domain data
            freqs, mag_db, phase_deg, min_phase_deg, delta_f = compute_frequency_response(
                audio_data, 
                sample_rate, 
                state['gate_idx'], 
                state['tukey_alpha'],
                state['start_freq'], 
                state['end_freq'],
                args.smoothing 
            )


            # Create a boolean mask where True means the frequency is VALID
            # Lower limit: delta_f (resolution)
            # Upper limit: state['end_freq']
            valid_mask = (freqs >= delta_f) & (freqs <= state['end_freq'])

            # Apply np.nan in place to all indices where the mask is False
            mag_db[~valid_mask] = np.nan
            phase_deg[~valid_mask] = np.nan
            min_phase_deg[~valid_mask] = np.nan

            state['freqs'] = freqs
            state['phase_deg'] = phase_deg
            state['min_phase_deg'] = min_phase_deg
            state['resolution'] = delta_f

            # Apply SPL corrections: ==============================================
            # Convert the SPL correction factor into a decibel offset
            if state['SPL_factor'] is not None and state['SPL_factor'] > 0:
                spl_offset_db = 20 * np.log10(state['SPL_factor'])
            else:
                spl_offset_db = 0.0

            # Apply the system SPL correction factor
            mag_db += spl_offset_db
            
            # 2. Apply the microphone calibration corrections using the pre-loaded handler
            if mic_cal_handler is not None and mic_cal_handler.is_valid:
                cal_corrections = mic_cal_handler.get_corrections(freqs)
                mag_db -= np.asarray(cal_corrections, dtype=float)
                
            # Update the dictionary with the fully calibrated magnitude data
            state['mag_db'] = mag_db
            
            # Done applying SPL corections ========================================

            
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            power = np.floor(np.log10(state['resolution']))
            highest_power_of_10 = 10 ** power
            ax_freq.set_xlim(highest_power_of_10, sample_rate / 2.0)
            
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            
            if state['showing_phase']:
                current_phase = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
                title_label = "Frequency Domain Data - Minimum Phase" if state['using_min_phase'] else "Frequency Domain Data - Phase"
                line, = ax_freq.plot(freqs, current_phase, color='g', lw=1.2)
                ax_freq.set_ylabel("Phase (degrees)")
                ax_freq.set_title(title_label)
            else:
                line, = ax_freq.plot(freqs, mag_db, color='r', lw=1.2)
                ax_freq.set_ylabel(mag_ylabel)
                ax_freq.set_title("Frequency Domain Data - Magnitude")
            
            ax_freq.set_xlabel(f'Frequency (Hz) at {state['resolution']:.3g} Hz resolution')
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            
            # Construct formatted completion message based on command-line arguments
            print(f"The computed frequency domain data has a resolution of {delta_f:.3g} Hz")

            # Print information about the frequency magnitude values
            SPL_info_text = "SPL units for computed frequency response = "
            if state['SPL_factor'] is not None and mic_cal_handler is not None:
              SPL_info_text += "dB re 20 uPa @ 2.83V/1m" 
            else:
              SPL_info_text += "Relative SPL Level in dB"
            print(f"{SPL_info_text}")


    def handle_view_button(event):
        if state['freqs'] is not None:
            state['showing_phase'] = not state['showing_phase']
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            highest_power_of_10 = 10 ** power
            ax_freq.set_xlim(highest_power_of_10, sample_rate / 2.0)
            
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            
            if state['showing_phase']:
                current_phase = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
                title_label = "Frequency Domain Data - Minimum Phase" if state['using_min_phase'] else "Frequency Domain Data - Phase"
                line, = ax_freq.plot(state['freqs'], current_phase, color='g', lw=1.2)
                ax_freq.set_ylabel("Phase (degrees)")
                ax_freq.set_title(title_label)
            else:
                line, = ax_freq.plot(state['freqs'], state['mag_db'], color='r', lw=1.2)
                ax_freq.set_ylabel(mag_ylabel)
                ax_freq.set_title("Frequency Domain Data - Magnitude")
                
            ax_freq.set_xlabel(f'Frequency (Hz) at {state['resolution']:.3g} Hz resolution')
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            print(f"Toggled view to {'Phase (' + ('MinPhase' if state['using_min_phase'] else 'Standard') + ')' if state['showing_phase'] else 'Magnitude'}.")
            # Update the text box that shows which phase type is active
            update_phase_indicator()
        else:
            print("No frequency data computed yet. Click 'Compute/Clear' first.")


    def update_phase_indicator():
      """Call this function inside your toggle button callback
      whenever `using_min_phase` changes state.  """
      new_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
      phase_textbox.set_val(new_text)


    def handle_min_phase_button(event):
        state['using_min_phase'] = not state['using_min_phase']
        # Update the text box showing the phase type
        update_phase_indicator()
        if state['using_min_phase']:
            print("MinPhase enabled.")
        else:
            print("MinPhase disabled.")
        
        # Refresh lower plot if showing phase currently
        if state['freqs'] is not None and state['showing_phase']:
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            
            current_phase = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
            title_label = "Frequency Domain Data - Minimum Phase" if state['using_min_phase'] else "Frequency Domain Data - Phase"
            line, = ax_freq.plot(state['freqs'], current_phase, color='g', lw=1.2)
            ax_freq.set_ylabel("Phase (degrees)")
            ax_freq.set_title(title_label)
            ax_freq.set_xlabel(f'Frequency (Hz) at {state['resolution']:.3g} Hz resolution')
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()

    def handle_save_exit_button(event):
        if state['freqs'] is None:
            print("The frequency response data has not yet been computed!")
            return
        phase_to_save = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
        if state['using_min_phase']:
            print(f"Saving FRD data files (minimum phase response)...")
        else:
            print(f"Saving FRD data files...")
        write_frd_file(output_filename, state['freqs'], state['mag_db'], phase_to_save)
        
        print("Closing application.")
        plt.close('all')
        sys.exit(0)

    # Connect buttons to handlers
    btn_gate.on_clicked(handle_gate_button)
    btn_calc.on_clicked(handle_calc_button)
    btn_view.on_clicked(handle_view_button)
    btn_min.on_clicked(handle_min_phase_button)
    btn_save_exit.on_clicked(handle_save_exit_button)
    
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