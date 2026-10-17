"""
NoGuiTime2Freq - Convert Time Domain Response to the Frequency Domain Response 
Description: This program is similar to Time2Freq but runs completely via the
    parameters supplied on the command line for batch processing of files. It 
    calculates the time domain frequency response from a windowed impulse 
    response, plus correction factors. Can produce relative SPL levels or in 
    units of dB re 20 uPa @ 2.83V/1m. Both the regular phase and minimum phase 
    responses are also made available. 

Author: Charlie Laub
Date: October 2026
"""

import argparse
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

def parse_arguments():
    """
    Parses command-line arguments for the non-interactive impulse response processor,
    including optional microphone sensitivity, calibration file, and amplifier gain.
    """
    parser = argparse.ArgumentParser(description="Process loudspeaker impulse response WAV file and export to FRD format (non-interactive).")
    parser.add_argument("input_wav", 
            type=str, 
            help="Path to the input impulse response WAV file."
    )
    parser.add_argument("-o", "--output", 
            type=str, 
            default="FR.frd",
            help="Path to save the output in FRD format (default: FR.frd)."
    )
    parser.add_argument("--tukey_alpha", 
            type=float, 
            default=0.2,
            help="Tukey window alpha taper parameter (default: 0.2)."
    )
    parser.add_argument("--gate_time", 
            type=float, 
            default=None,
            help="Gate/window end time in milliseconds (default: end of audio data)."
    )
    parser.add_argument("--save_min_phase", 
            action="store_true",
            help="When present, saves the minimum phase response; otherwise saves standard phase."
    )
    parser.add_argument("--mic_cal", 
            type=str, 
            default=None,
            help="Path to microphone calibration file (*.mic or *.cal)."
    )
    parser.add_argument("--mic_sensitivity", 
            type=float, 
            default=None,
            help="Microphone sensitivity in milliVolts per Pascal (mV/Pa)."
    )
    parser.add_argument("--v_stimulus", 
            type=float, 
            default=2.83,
            help="RMS Voltage reported in SPL units (default: 2.83V)."
    )
    parser.add_argument("--distance", 
            type=float, 
            default=None,
            help="Distance between mic and source in meters."
    )
    return parser.parse_args()

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
    Standard FRD format: Frequency (Hz), Magnitude (dB), Phase (degrees)
    """
    try:
        with open(filename, 'w') as f:
            f.write("* Frequency Response Data (FRD)\n")
            f.write("* Columns: Frequency (Hz), Magnitude (dB), Phase (deg)\n")
            for freq, mag, phase in zip(freqs, mag_db, phase_deg):
                f.write(f"{freq:.4f}\t{mag:.4f}\t{phase:.4f}\n")
        print(f"Successfully saved FRD data to {filename}")
    except Exception as e:
        print(f"Error saving FRD file: {e}")

def compute_frequency_response(audio_data, sample_rate, gate_end_idx, 
            tukey_alpha, mic_cal_file, mic_sensitivity, v_stimulus, 
            distance
            ):
    """
    Applies a Tukey window, computes standard FFT and minimum phase FFT, 
    applies microphone calibration, sensitivity scaling, and amplifier gain correction,
    and returns frequencies, magnitude (dB), standard phase (deg), and minimum phase (deg).
    """
    gate_end_idx = max(10, min(gate_end_idx, len(audio_data)))
    
    # Get Tukey window using scipy.signal.get_window
    win = scipy.signal.get_window(('tukey', tukey_alpha), gate_end_idx)
    
    windowed_data = np.zeros_like(audio_data, dtype=np.float32)
    segment = audio_data[:gate_end_idx].astype(np.float32)
    windowed_data[:gate_end_idx] = segment * win
    
    # Zero-pad for smoother frequency resolution
    n_fft = max(2048, 2 * len(windowed_data))
    
    # Standard FFT response
    fft_result = np.fft.rfft(windowed_data, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    
    mag = np.abs(fft_result)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12)) # Prevent log(0)
    phase_deg = np.angle(fft_result, deg=True)
    
    # Apply Microphone Sensitivity Scaling
    if mic_sensitivity is not None and mic_sensitivity > 0:
        sensitivity_v_pa = mic_sensitivity / 1000.0  # mV/Pa to V/Pa
        p_ref = 20e-6  # 20 uPa reference pressure
        spl_offset = -20.0 * np.log10(sensitivity_v_pa * p_ref)
        mag_db += spl_offset
        
    # Apply Microphone Calibration File Compensation
    if mic_cal_file is not None:
        cal_corrections = read_mic_cal_file(mic_cal_file, freqs)
        mag_db -= cal_corrections
        
    # Scale for stimulus RMS voltage
    mag_db -= 20.0 * np.log10( v_stimulus )

    # Scale for measurement distance
    if distance is not None:
        mag_db += 20.0 * np.log10( distance )

    # Compute Minimum Phase response
    try:
        min_phase_signal = scipy.signal.minimum_phase(windowed_data, method='homomorphic', half=False)
        min_fft_result = np.fft.rfft(min_phase_signal, n=n_fft)
        min_phase_deg = np.angle(min_fft_result, deg=True)
    except Exception:
        min_phase_deg = phase_deg.copy()
    
    valid_idx = freqs <= (sample_rate / 2.0)
    return freqs[valid_idx], mag_db[valid_idx], phase_deg[valid_idx], min_phase_deg[valid_idx]

def main():
    args = parse_arguments()
    
    try:
        sample_rate, audio_data = read_wav_file(args.input_wav)
    except FileNotFoundError:
        print(f"Error: WAV file '{args.input_wav}' not found.")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
        
    total_duration_ms = (len(audio_data) / sample_rate) * 1000.0
    
    if args.gate_time is not None:
        gate_idx = int((args.gate_time / 1000.0) * sample_rate)
    else:
        gate_idx = len(audio_data)
        
    print(f"Processing WAV: {args.input_wav}")
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")
    print(f"Tukey Alpha: {args.tukey_alpha}")
    print(f"Gate End Time: {args.gate_time if args.gate_time is not None else total_duration_ms:.2f} ms")
    print(f"Exporting Minimum Phase: {args.save_min_phase}")
    
    # Compute frequency response with all parameters
    freqs, mag_db, phase_deg, min_phase_deg = compute_frequency_response(
        audio_data, sample_rate, gate_idx, args.tukey_alpha, args.mic_cal, args.mic_sensitivity, args.distance
    )
    
    # Construct and print terminal message confirming calculation parameters
    print(f"Frequency domain data computed with the following corrections:")
    if args.mic_sensitivity is not None:
        print(f"   \u2022 a microphone sensitivity of {args.mic_sensitivity} mV/Pa")
    if args.mic_cal is not None:
        print(f"   \u2022 microphone calibration data from file {args.mic_cal}")
    if args.v_stimulus is not None:
        print(f"   \u2022 a stimulus RMS voltage of of {args.v_stimulus} Volts")
    if args.distance is not None:
        print(f"   \u2022 a measurement distance of {args.distance} meters")

    # Print information about the frequency magnitude values
    SPL_info_text = "SPL units for computed frequency response = "
    if (
      args.mic_sensitivity is not None
      and args.mic_cal is not None
    ):
      SPL_info_text += "dB re 20 uPa @ {args.v_stimulus:.2f}V" 
      if args.distance is not None:
        SPL_info_text += "/1m"
    else:
      SPL_info_text += "Relative SPL Level in dB"
    print(f"{SPL_info_text}")
    
    phase_to_save = min_phase_deg if args.save_min_phase else phase_deg
    write_frd_file(args.output, freqs, mag_db, phase_to_save)

if __name__ == "__main__":
    main()