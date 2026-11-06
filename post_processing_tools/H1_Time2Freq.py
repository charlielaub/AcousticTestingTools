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
import matplotlib.ticker as ticker
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
    optional microphone calibration, sensitivity parameters, amplifier gain,
    and automated mode flags.
    """
    parser = argparse.ArgumentParser(description="Process loudspeaker impulse response WAV file and export to FRD format.",
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_wav", 
            type=str, 
            help="Path to the input impulse response WAV file."
    )
    parser.add_argument("-o", "--output", 
            type=str, 
            default="FR.frd",
            help="File and pathname to save the output in FRD format (default: FR.frd)."
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
    
    # --- AUTOMATED MODE ARGUMENTS ---
    parser.add_argument("-g","--end_gate_time", 
            type=float, 
            default=None,
            help="Automated mode trigger: End gate time in milliseconds. If provided, the script runs headless without launching the GUI."
    )
    parser.add_argument("-a","--tukey_alpha", 
            type=float, 
            default=0.1,
            help="Tukey window alpha value (default: 0.1)."
    )
    parser.add_argument("-m","--min_phase", 
            action="store_true",
            help="If set, uses minimum phase instead of regular phase. Applies to automated mode output, and sets the default view in the GUI."
    )
    parser.add_argument("--AES17_SPL", 
            action="store_true",
            help="If set, use the AES17 SPL level convention."
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
    """
    with open(filename, "rb") as f:
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
                if audio_format != 3:
                    raise ValueError("WAV file is not in 32-bit IEEE float format (format tag 3).")

            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
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
                f.seek(chunk_size, 1)

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
    """
    try:
        freqs = np.asarray(freqs)
        mag_db = np.asarray(mag_db)
        phase_deg = np.asarray(phase_deg)
        
        valid_mask = np.isfinite(freqs) & np.isfinite(mag_db) & np.isfinite(phase_deg)
        
        with open(filename, 'w') as f:
            f.write("* Frequency Response Data (FRD)\n")
            f.write("* Columns: Frequency (Hz), Magnitude (dB), Phase (deg)\n")
            
            for freq, mag, phase in zip(freqs[valid_mask], mag_db[valid_mask], phase_deg[valid_mask]):
                f.write(f"{freq:.1f}\t{mag:.2f}\t{phase:.1f}\n")
                
        print(f"{filename}")
    except Exception as e:
        print(f" WARNING: The following error was encountered when attempting to save file {filename}: {e}")



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
    and interpolates the response.
    """
    gate_end_idx = max(10, min(gate_end_idx, len(audio_data)))
    win = scipy.signal.get_window(('tukey', tukey_alpha), gate_end_idx)
    windowed_data = audio_data[:gate_end_idx].astype(np.float32) * win
    n_fft = max(2048, 2 * len(windowed_data))

    # Standard FFT response
    fft_result = np.fft.rfft(windowed_data, n=n_fft)
    
    # Calculate the frequency bins (This is the line that was missing!)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    
    # 1. Get raw magnitude (No division by sample_rate or window sum needed!)
    mag = np.abs(fft_result)
    
    # 2. Calculate dB
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    
    raw_phase_rad = np.angle(fft_result)
    phase_deg = np.degrees(np.unwrap(raw_phase_rad))

    try:
        min_phase_signal = scipy.signal.minimum_phase(
            windowed_data, method='homomorphic', half=False
        )
        min_fft_result = np.fft.rfft(min_phase_signal, n=n_fft)
        min_phase_raw_rad = np.angle(min_fft_result)
        min_phase_deg = np.degrees(np.unwrap(min_phase_raw_rad))
    except Exception:
        min_phase_deg = phase_deg.copy()

    try:
        min_phase_rad = np.deg2rad(min_phase_deg)
        min_complex_spectrum = mag * np.exp(1j * min_phase_rad)
        impulse_minphase = np.fft.irfft(min_complex_spectrum, n=n_fft)
        impulse_minphase_trimmed = impulse_minphase[: len(windowed_data)]

        f = windowed_data - np.mean(windowed_data)
        g = impulse_minphase_trimmed - np.mean(impulse_minphase_trimmed)
        xcorr = scipy.signal.correlate(f, g, mode='full')
        norm = np.sqrt(np.sum(f**2) * np.sum(g**2))
        if norm > 0:
            xcorr /= norm

        ncc_max_pos = np.argmax(xcorr)
        peak_index_offset = abs(ncc_max_pos - (len(g) - 1))
        ncc_max_time = float(peak_index_offset) / sample_rate
        gate_end_time = float(gate_end_idx) / sample_rate

        time_diff = gate_end_time - ncc_max_time
        delta_frequency = (
            float(1.0 / time_diff) if time_diff > 0 else float(sample_rate / n_fft)
        )
    except Exception:
        delta_frequency = float(sample_rate / n_fft)

    if Fgrid_density > 0:
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
        
        # Re-wrap universally for the Fgrid_density > 0 path
        res_phase = (res_phase + 180.0) % 360.0 - 180.0
        res_min_phase = (res_min_phase + 180.0) % 360.0 - 180.0
        
        return res_freqs, res_mag, res_phase, res_min_phase, delta_frequency
    else:
        valid_idx = freqs <= (sample_rate / 2.0)
        return (
            freqs[valid_idx],
            mag_db[valid_idx],
            phase_deg[valid_idx],
            min_phase_deg[valid_idx],
            delta_frequency,
        )


def process_and_apply_corrections(audio_data, sample_rate, gate_idx, tukey_alpha, start_freq, end_freq, smoothing, SPL_factor, use_AES17, mic_cal_handler):
    """
    Helper function to run the DSP calculation and apply all masks and SPL/Mic offsets.
    Shared by both the GUI and Headless automated pathways.
    """
    freqs, mag_db, phase_deg, min_phase_deg, delta_f = compute_frequency_response(
        audio_data, sample_rate, gate_idx, tukey_alpha, start_freq, end_freq, smoothing
    )

    valid_mask = (freqs >= delta_f) & (freqs <= end_freq)
    mag_db[~valid_mask] = np.nan
    phase_deg[~valid_mask] = np.nan
    min_phase_deg[~valid_mask] = np.nan

    # Apply SPL corrections
    if SPL_factor is not None and SPL_factor > 0:
        spl_offset_db = 20 * np.log10(SPL_factor)
    else:
        spl_offset_db = 0.0

    #  If the user wishes to obtain the AES17 SPL level, add 3.01dB
    if use_AES17:
        spl_offset_db += 3.01

    
    mag_db += spl_offset_db
    
    # Apply microphone calibration corrections
    if mic_cal_handler is not None and mic_cal_handler.is_valid:
        cal_corrections = mic_cal_handler.get_corrections(freqs)
        mag_db -= np.asarray(cal_corrections, dtype=float)
        
    return freqs, mag_db, phase_deg, min_phase_deg, delta_f



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
            
            sort_indices = np.argsort(cal_freqs)
            cal_freqs = cal_freqs[sort_indices]
            cal_db = cal_db[sort_indices]
            
            self.f_min = cal_freqs[0]
            self.f_max = cal_freqs[-1]
            
            self.interpolator = Akima1DInterpolator(cal_freqs, cal_db, extrapolate=False)
            
            low_data_mask = cal_freqs <= 100.0
            if np.sum(low_data_mask) >= 2:
                fit_freqs = cal_freqs[low_data_mask]
                fit_db = cal_db[low_data_mask]
            else:
                n_pts = min(3, len(cal_freqs))
                fit_freqs = cal_freqs[:n_pts]
                fit_db = cal_db[:n_pts]
            self.poly_coeff_low = np.polyfit(np.log10(fit_freqs), fit_db, 1)
            
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
        if not self.is_valid:
            return np.zeros_like(freqs)
            
        corrections = np.zeros_like(freqs, dtype=float)
        
        low_mask = freqs < self.f_min
        if np.any(low_mask) and self.poly_coeff_low is not None:
            corrections[low_mask] = np.polyval(self.poly_coeff_low, np.log10(freqs[low_mask]))
        
        high_mask = freqs > self.f_max
        if np.any(high_mask) and self.poly_coeff_high is not None:
            corrections[high_mask] = np.polyval(self.poly_coeff_high, np.log10(freqs[high_mask]))
            
        mid_mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        if np.any(mid_mask):
            corrections[mid_mask] = self.interpolator(freqs[mid_mask])
            
        return corrections


def main():
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

    print(f"\n------------------------ Measurement Parameters: -------------------------------")
    print(f"             Start Frequency (f0sw): {impulse_metadata.get('f0sw', 0.0)} Hz")
    print(f"               End Frequency (f1sw): {impulse_metadata.get('f1sw', 0.0)} Hz")
    print(f"                 Chirp Rate (rchr): {impulse_metadata.get('rchr', 0.0)} seconds/decade Hz")
    print(f"                 Repetitions (reps): {impulse_metadata.get('reps', 1)}")
    print(f"              Internal Level (levl): {impulse_metadata.get('levl', 0.0):.3e}")
    print(f"             HF Window Start (fhfw): {impulse_metadata.get('fhfw', 0.0)} Hz")
    print(f"                   Timestamp (date): {measurement_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"--------------------------------------------------------------------------------\n")

    total_duration_ms = (len(audio_data) / sample_rate) * 1000.0
    print(f"Processing WAV: {args.input_wav}")
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")

    time_axis = np.arange(len(audio_data)) / sample_rate
    
    SPL_factor = None
    if all(v > 0 for v in (v_in1_max, v_in2_max, amp_gain, mic_sens, distance, RMS_internal_level)) and args.mic_cal_file is not None:
        mag_ylabel = f"SPL [dB re 20 uPa @ 2.83V/1m" 
        SPL_factor = distance / ( 0.001 * mic_sens ) 
        SPL_factor *= ( v_in2_max / v_in1_max )
        SPL_factor *= 2.83
        SPL_factor /= amp_gain
        SPL_factor *= 50000.0
        print(f"The SPL conversion factor is {SPL_factor:.3e}\n")
    else:
        mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"
    
    if args.AES17_SPL: 
        print(f"The AES17 SPL convention will be used. \n")
    
    mic_cal_handler = None
    if args.mic_cal_file is not None:
        mic_cal_handler = MicCalibrationHandler(args.mic_cal_file)
        print(f"Using the microphone calibration data from {args.mic_cal_file}.\n")

    if args.smoothing > 48 or args.smoothing < 1:
        args.smoothing = 24
    print(f"The data will be calculated using {args.smoothing} points per octave.\n")
    
    # Check and enforce that the output filename includes the frd extension
    my_string = args.output
    # strip any trailing whitespaces
    my_string = my_string.rstrip()
    # Check if it ends with the FRD extension, if not, append it
    if not my_string.endswith(".frd"):
        my_string += ".frd"
        args.output = my_string

    # State variables setup using argparse defaults to seed the GUI
    state = {
        'gate_idx': None,
        'selecting_gate': False,
        'freqs': None,
        'mag_db': None,
        'phase_deg': None,
        'min_phase_deg': None,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': args.min_phase,
        'tukey_alpha': args.tukey_alpha,
        'resolution': None,
        'SPL_factor': SPL_factor,
        'start_freq': measurement_start_freq,
        'end_freq': HF_window_startF
    }
    
    # =========================================================================
    # HEADLESS / AUTOMATED MODE 
    # =========================================================================
    if args.end_gate_time is not None:
        print(f"\n--- Running Automated Mode ---")
        print(f"End Gate Time: {args.end_gate_time} ms")
        print(f"Tukey Alpha: {args.tukey_alpha}")
        print(f"Min Phase Output: {'Yes' if args.min_phase else 'No'}\n")
        
        gate_idx = int((args.end_gate_time / 1000.0) * sample_rate)
        gate_idx = max(10, min(gate_idx, len(audio_data)))
        
        freqs, mag_db, phase_deg, min_phase_deg, delta_f = process_and_apply_corrections(
            audio_data, sample_rate, gate_idx, args.tukey_alpha,
            measurement_start_freq, HF_window_startF, args.smoothing,
            SPL_factor, args.AES17_SPL, mic_cal_handler
        )
        
        phase_to_save = min_phase_deg if args.min_phase else phase_deg
        
        print(f"The computed frequency domain data has a resolution of {delta_f:.3g} Hz")
        if state['using_min_phase']:
            print(f"Saving data to FRD file (minimum phase): ", end="")
        else:
            print(f"Saving data to FRD file: ", end="")
        write_frd_file(output_filename, freqs, mag_db, phase_to_save)
        print("Automated processing complete. Exiting.")
        return  # Exits smoothly without opening the Matplotlib GUI
    # =========================================================================
    
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23)
    
    ax_time.plot(time_axis * 1000, audio_data, color='b', lw=1, label='IR')
    ax_time.set_title(f"Impulse Response: {wav_filename} (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    ax_time.set_xlim(-0.1, 100.0)
    
    initial_gate_time_ms = 50.0
    initial_gate_idx = int( len(audio_data) * initial_gate_time_ms / total_duration_ms )
    gate_line = ax_time.axvline(x=initial_gate_time_ms, color='g', linestyle='--', lw=1.5, label='Window End')
    state['gate_idx'] = initial_gate_idx
    
    initial_win = scipy.signal.get_window(('tukey', state['tukey_alpha']), initial_gate_idx)
    ylim_initial = ax_time.get_ylim()
    initial_y_top = ylim_initial[1] if ylim_initial[1] != 0 else 1.0
    window_line, = ax_time.plot(time_axis[:initial_gate_idx] * 1000, initial_win * (0.95 * initial_y_top), color='r', lw=1.5, label='Tukey Window')
    
    ax_time.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='black')
    
    ax_freq.set_title("Frequency Domain Data")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel(mag_ylabel)
    ax_freq.set_xscale('log')
    ax_freq.set_xlim(10, sample_rate / 2.0)
    ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    
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
    
    ax_box_tukey = plt.axes([0.21, 0.10, 0.06, 0.03])
    textbox_tukey = TextBox(ax_box_tukey, 'TUKEY WINDOW WIDTH:  ', initial=str(state['tukey_alpha']),textalignment='center')

    ax_box_phase_type = plt.axes([0.66, 0.10, 0.13, 0.03])
    initial_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
    phase_textbox = TextBox(ax_box_phase_type, 'PHASE TYPE:  ', initial=initial_text, textalignment='center')
    phase_textbox.set_active(False)

    def update_tukey_visual(event=None):
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
            for artist in state['freq_line_plot']:
                artist.remove()
            state['freq_line_plot'] = None
            ax_freq.relim()
            ax_freq.autoscale_view()
            ax_freq.set_ylabel(mag_ylabel)
            ax_freq.set_title("Frequency Domain Data (Cleared)")
            ax_freq.set_xlabel(f'Frequency (Hz)')
            fig.canvas.draw_idle()
            
            state['freqs'] = None
            state['mag_db'] = None
            state['phase_deg'] = None
            state['min_phase_deg'] = None
            state['resolution'] = None
            print("Lower plot cleared. All calculated data cleared.")
        else:
            # Reusing the shared mathematical helper function
            freqs, mag_db, phase_deg, min_phase_deg, delta_f = process_and_apply_corrections(
                audio_data, sample_rate, state['gate_idx'], state['tukey_alpha'],
                state['start_freq'], state['end_freq'], args.smoothing,
                state['SPL_factor'], args. AES17_SPL, mic_cal_handler
            )

            state['freqs'] = freqs
            state['mag_db'] = mag_db
            state['phase_deg'] = phase_deg
            state['min_phase_deg'] = min_phase_deg
            state['resolution'] = delta_f
            
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
                ax_freq.set_ylim(-180,180)
                ax_freq.yaxis.set_major_locator(ticker.MultipleLocator(60))
                ax_freq.yaxis.set_minor_locator(ticker.MultipleLocator(30))
            else:
                line, = ax_freq.plot(freqs, mag_db, color='r', lw=1.2)
                ax_freq.set_ylabel(mag_ylabel)
                ax_freq.set_title("Frequency Domain Data - Magnitude")
                mag_data_for_scaling = mag_db
                valid_mag = mag_data_for_scaling[~np.isnan(mag_data_for_scaling)]
                data_max = np.max(valid_mag) if len(valid_mag) > 0 else 0.0
                y_max = np.ceil(data_max / 5.0) * 5.0
                y_min = y_max - 50.0
                ax_freq.set_ylim(y_min, y_max)
                ax_freq.yaxis.set_major_locator(ticker.MultipleLocator(10))
                ax_freq.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            
            ax_freq.set_xlabel(f"Frequency (Hz) at {state['resolution']:.3g} Hz resolution")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            
            print(f"The computed frequency domain data has a resolution of {delta_f:.3g} Hz")

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
            highest_power_of_10 = 10 ** np.floor(np.log10(state['resolution']))
            ax_freq.set_xlim(highest_power_of_10, sample_rate / 2.0)
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            ax_freq.set_ylim(-180,180)
            ax_freq.yaxis.set_major_locator(ticker.MultipleLocator(60))
            ax_freq.yaxis.set_minor_locator(ticker.MultipleLocator(30))
            
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
                mag_data_for_scaling = state['mag_db']
                valid_mag = mag_data_for_scaling[~np.isnan(mag_data_for_scaling)]
                data_max = np.max(valid_mag) if len(valid_mag) > 0 else 0.0
                y_max = np.ceil(data_max / 5.0) * 5.0
                y_min = y_max - 50.0
                ax_freq.set_ylim(y_min, y_max)
                ax_freq.yaxis.set_major_locator(ticker.MultipleLocator(10))
                ax_freq.yaxis.set_minor_locator(ticker.MultipleLocator(5))
                
            ax_freq.set_xlabel(f"Frequency (Hz) at {state['resolution']:.3g} Hz resolution")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            print(f"Toggled view to {'Phase (' + ('MinPhase' if state['using_min_phase'] else 'Standard') + ')' if state['showing_phase'] else 'Magnitude'}.")
            update_phase_indicator()
        else:
            print("No frequency data computed yet. Click 'Compute/Clear' first.")

    def update_phase_indicator():
      new_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
      phase_textbox.set_val(new_text)

    def handle_min_phase_button(event):
        state['using_min_phase'] = not state['using_min_phase']
        update_phase_indicator()
        if state['using_min_phase']:
            print("MinPhase enabled.")
        else:
            print("MinPhase disabled.")
        
        if state['freqs'] is not None and state['showing_phase']:
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            highest_power_of_10 = 10 ** np.floor(np.log10(state['resolution']))
            ax_freq.set_xlim(highest_power_of_10, sample_rate / 2.0)
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            ax_freq.set_ylim(-180,180)
            ax_freq.yaxis.set_major_locator(ticker.MultipleLocator(60))
            ax_freq.yaxis.set_minor_locator(ticker.MultipleLocator(30))
            
            current_phase = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
            title_label = "Frequency Domain Data - Minimum Phase" if state['using_min_phase'] else "Frequency Domain Data - Phase"
            line, = ax_freq.plot(state['freqs'], current_phase, color='g', lw=1.2)
            ax_freq.set_ylabel("Phase (degrees)")
            ax_freq.set_title(title_label)
            ax_freq.set_xlabel(f"Frequency (Hz) at {state['resolution']:.3g} Hz resolution")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()

    def handle_save_exit_button(event):
        if state['freqs'] is None:
            print("The frequency response data has not yet been computed!")
            return
        phase_to_save = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
        if state['using_min_phase']:
            print(f"Saving data to FRD file (minimum phase): ", end="")
        else:
            print(f"Saving data to FRD file: ", end="")
        write_frd_file(output_filename, state['freqs'], state['mag_db'], phase_to_save)
        
        print("Closing application.")
        plt.close('all')
        sys.exit(0)

    btn_gate.on_clicked(handle_gate_button)
    btn_calc.on_clicked(handle_calc_button)
    btn_view.on_clicked(handle_view_button)
    btn_min.on_clicked(handle_min_phase_button)
    btn_save_exit.on_clicked(handle_save_exit_button)
    
    toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None

    if toolbar is not None:
        if hasattr(toolbar, 'actions'):
            for action in toolbar.actions():
                text = action.text().lower()
                tooltip = action.toolTip().lower()
                if 'home' in text or 'reset' in text or 'home' in tooltip or 'reset' in tooltip:
                    toolbar.removeAction(action)
        elif hasattr(toolbar, '_buttons'):
            if 'Home' in toolbar._buttons:
                toolbar._buttons['Home'].pack_forget()

    if toolbar is not None:
        toolbar.zoom()

    plt.show()

if __name__ == "__main__":
    main()