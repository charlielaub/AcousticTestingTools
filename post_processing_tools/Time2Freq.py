"""
Time2Freq - Convert the Time Domain Response to the Frequency Domain Response 
Description: Calculates the time domain frequency response from a windowed 
    impulse response, plus correction factors. Can produce relative SPL levels 
    or in units of dB re 20 uPa @ 2.83V/1m. Both the regular phase and minimum 
    phase responses are also made available.

Author: Charlie Laub
Date: October 2026
"""

import argparse
import sys
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.ticker import LogLocator

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
    applies correction factors and returns frequencies, magnitude (dB), 
    standard phase (deg), and minimum phase (deg).
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
    # Parse command-line arguments
    args = parse_arguments()
    wav_filename = args.input_wav
    output_filename = args.output
    
    try:
        sample_rate, audio_data = read_wav_file(wav_filename)
    except FileNotFoundError:
        print(f"Error: WAV file '{wav_filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
    
    total_duration_ms = (len(audio_data) / sample_rate) * 1000.0
    print(f"Processing WAV: {args.input_wav}")
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")

    time_axis = np.arange(len(audio_data)) / sample_rate
    
    # Determine y-axis label for magnitude based on whether mic sensitivity is supplied
    if (
      args.mic_sensitivity is not None
      and args.mic_cal is not None
    ):
      mag_ylabel = "[dB re 20 uPa @ {args.v_stimulus:.2f}V" 
      if args.distance is not None:
        mag_ylabel += "/1m"
    else:
      mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"
    
    # Setup Figure and Subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23)
    
    # 1. Plot Time Domain Data
    ax_time.plot(time_axis * 1000, audio_data, color='b', lw=1, label='IR')
    ax_time.set_title(f"Impulse Response: {wav_filename} (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    
    # Initial gate line position set to the end of the data (Green)
    initial_gate_time_ms = time_axis[-1] * 1000
    initial_gate_idx = len(audio_data)
    gate_line = ax_time.axvline(x=initial_gate_time_ms, color='g', linestyle='--', lw=1.5, label='Window End')
    
    # State variables
    state = {
        'gate_idx': initial_gate_idx,
        'selecting_gate': False,
        'freqs': None,
        'mag_db': None,
        'phase_deg': None,
        'min_phase_deg': None,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': False,
        'tukey_alpha': 0.05
    }
    
    # Plot Tukey window function as a red line initially scaled to 95% of current y max
    initial_win = scipy.signal.get_window(('tukey', state['tukey_alpha']), initial_gate_idx)
    ylim_initial = ax_time.get_ylim()
    initial_y_top = ylim_initial[1] if ylim_initial[1] != 0 else 1.0
    window_line, = ax_time.plot(time_axis[:initial_gate_idx] * 1000, initial_win * (0.95 * initial_y_top), color='r', lw=1.5, label='Tukey Window')
    
    ax_time.legend(loc='lower right')
    
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
    btn_y = 0.04
    btn_h = 0.05
    
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
    ax_box_tukey = plt.axes([0.19, 0.125, 0.12, 0.035])
    textbox_tukey = TextBox(ax_box_tukey, '', initial=str(state['tukey_alpha']))
    
    # Larger font label with colon
    fig.text(0.03, 0.13, "Tukey window taper:", fontsize=11, fontweight='medium')

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
            fig.canvas.draw_idle()
            print("Lower plot cleared.")
        else:
            # Compute and plot frequency domain data
            freqs, mag_db, phase_deg, min_phase_deg = compute_frequency_response(
                audio_data, sample_rate, state['gate_idx'], state['tukey_alpha'], 
                args.mic_cal, args.mic_sensitivity, args.v_stimulus,
                args.distance
            )
            state['freqs'] = freqs
            state['mag_db'] = mag_db
            state['phase_deg'] = phase_deg
            state['min_phase_deg'] = min_phase_deg
            
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            
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
            
            ax_freq.set_xlabel("Frequency (Hz)")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            
            # Construct formatted completion message based on command-line arguments
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


    def handle_view_button(event):
        if state['freqs'] is not None:
            state['showing_phase'] = not state['showing_phase']
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            
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
                
            ax_freq.set_xlabel("Frequency (Hz)")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()
            print(f"Toggled view to {'Phase (' + ('MinPhase' if state['using_min_phase'] else 'Standard') + ')' if state['showing_phase'] else 'Magnitude'}.")
        else:
            print("No frequency data computed yet. Click 'Compute/Clear' first.")

    def handle_min_phase_button(event):
        state['using_min_phase'] = not state['using_min_phase']
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
            ax_freq.set_xlabel("Frequency (Hz)")
            state['freq_line_plot'] = [line]
            fig.canvas.draw_idle()

    def handle_save_exit_button(event):
        if state['freqs'] is not None:
            phase_to_save = state['min_phase_deg'] if state['using_min_phase'] else state['phase_deg']
            write_frd_file(output_filename, state['freqs'], state['mag_db'], phase_to_save)
        else:
            print("No frequency data computed. Computing response before saving...")
            freqs, mag_db, phase_deg, min_phase_deg = compute_frequency_response(
                audio_data, sample_rate, state['gate_idx'], state['tukey_alpha'], 
                args.mic_cal, args.mic_sensitivity
            )
            phase_to_save = min_phase_deg if state['using_min_phase'] else phase_deg
            write_frd_file(output_filename, freqs, mag_db, phase_to_save)
        
        print("Closing application.")
        plt.close('all')
        sys.exit(0)

    # Connect buttons to handlers
    btn_gate.on_clicked(handle_gate_button)
    btn_calc.on_clicked(handle_calc_button)
    btn_view.on_clicked(handle_view_button)
    btn_min.on_clicked(handle_min_phase_button)
    btn_save_exit.on_clicked(handle_save_exit_button)
    
    plt.show()

if __name__ == "__main__":
    main()