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


def parse_arguments():
    """
    Parses command-line arguments for input WAV file, base filename for outputs,
    optional microphone calibration, sensitivity parameters, and amplifier gain.
    """
    parser = argparse.ArgumentParser(description="Process loudspeaker impulse response WAV file and export data/plots (Interactive GUI).")
    parser.add_argument("-i","--impulse_data", 
            type=str,
            required=True, 
            help="Path to the H1 estimated impulse data WAV file."
    )
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




def check_harmonic_window_overlaps(full_ir, inv_filter, sample_rate, start_freq, gate_end_idx, max_harmonic=5):
    """
    Checks if the given gate_end_idx causes overlapping windows between 
    adjacent harmonics in a Farina method impulse response.
    
    Returns:
    --------
    peak_indices : dict
        Peak sample index for each harmonic.
    warnings_list : list of str
        Formatted warning messages if overlaps occur.
    overlaps : numpy.ndarray
        Array of overlap sample counts indexed by harmonic order k (size max_harmonic + 1).
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
        
    # 2. Initialize overlap tracking array (1-indexed for convenience: index k corresponds to harmonic k)
    overlaps = np.zeros(max_harmonic + 1, dtype=int)
    warnings_list = []
    
    # 3. Check for overlaps between adjacent harmonics
    for k in range(2, max_harmonic + 1):
        current_peak = peak_indices[k]
        window_end = current_peak + gate_end_idx
        prev_peak = peak_indices[k - 1] # The lower-order harmonic to its right
        
        if window_end > prev_peak:
            overlap_samples = window_end - prev_peak
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
            "Recommendation: Consider reducing gate_end_idx or increasing the sweep duration."
        )
            
    return peak_indices, warnings_list, overlaps




def run_farina_analysis(
          recorded_audio,
          sample_rate,
          inv_filter,
          start_freq,
          gate_end_idx,
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
      # 1. Compute sweep parameters from the inverse filter length
      duration = len(inv_filter) / sample_rate
      nyquist = 0.5 * sample_rate
      end_freq = 0.99 * nyquist
      ln_ratio = np.log(end_freq / start_freq)

      # 2. Perform Fast Convolution (Deconvolution)
      full_ir = scipy.signal.fftconvolve(recorded_audio, inv_filter, mode='full')
      linear_peak_idx = np.argmax(np.abs(full_ir))

      # Run an overlap check on the harmonics (which pre-calculates all peak indices!)
      peak_indices, warnings_list, overlaps = check_harmonic_window_overlaps(
          full_ir, inv_filter, sample_rate, start_freq, gate_end_idx, max_harmonic
      )
      
      # Print warnings to the console if any were found
      if warnings_list:
          for w in warnings_list:
              print(w)

      harmonic_results = {}
      
      # 3. Extract and process each harmonic
      for k in range(1, max_harmonic + 1):
          # Skip processing if this harmonic's window overlaps with the next one
          if k > 1 and overlaps[k] > 0:
              print(f"Skipping Harmonic {k} due to window overlap of ({overlaps[k]} samples).")
              continue  # Skips to the next iteration so bad data isn't processed

          # use the offsets and peak indices already calculated in check_harmonic_window_overlaps
          peak_idx = peak_indices[k]
          start_idx = max(0, peak_idx)
          end_idx = min(len(full_ir), peak_idx + gate_end_idx)

          segment = full_ir[start_idx:end_idx]

          # Apply calibration only to the fundamental (k=1)
          current_mic_cal = mic_cal_file if k == 1 else None
          current_mic_sens = mic_sensitivity if k == 1 else None
          current_dist = distance if k == 1 else None

          freqs, mag_db, phase, min_phase, delta_f = compute_frequency_response(
              audio_data=segment,
              sample_rate=sample_rate,
              gate_end_idx=len(segment),
              tukey_alpha=tukey_alpha,
              mic_cal_file=current_mic_cal,
              mic_sensitivity=current_mic_sens,
              v_stimulus=v_stimulus,
              distance=current_dist
          )

          harmonic_results[k] = {
              "freqs": freqs,
              "mag_db": mag_db,
              "phase": phase,
              "min_phase": min_phase,
              "delta_f": delta_f
          }

      # 4. Compute Total Harmonic Distortion (THD %)
      # Ensure harmonic 1 and at least one other harmonic are present
      if 1 in harmonic_results:
          fund_mag = harmonic_results[1]["mag_db"]
          distortion_sum_sq = 0.0

          for k in range(2, max_harmonic + 1):
              if k in harmonic_results:
                  harm_mag = harmonic_results[k]["mag_db"]
                  rel_mag_linear = 10.0 ** ((harm_mag - fund_mag) / 20.0)
                  distortion_sum_sq += rel_mag_linear ** 2

          harmonic_results["thd_percent"] = 100.0 * np.sqrt(distortion_sum_sq)
      else:
          harmonic_results["thd_percent"] = None

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




def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Read the pre-calculated H1 impulse response 
        sample_rate, H1_impulse = read_wav_file(args.impulse_data)
        
        # Read the raw microphone data and Farina inverse filter
        _, mic_data = read_wav_file(args.mic_data)
        _, inv_filter = read_wav_file(args.farina_filter)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        return
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return
    
    total_duration_ms = (len(H1_impulse) / sample_rate) * 1000.0
    print(f"Processing impulse response WAV: {args.impulse_data}")
    print(f"Sample Rate: {sample_rate} Hz | Total Duration: {total_duration_ms:.2f} ms")

    time_axis = np.arange(len(H1_impulse)) / sample_rate
    
    # Determine y-axis label for magnitude based on whether mic sensitivity is supplied
    if (
        args.mic_sensitivity is not None
        and args.mic_cal is not None
    ):
        mag_ylabel = f"[dB re 20 uPa @ {args.v_stimulus:.2f}V" 
        if args.distance is not None:
            mag_ylabel += "/1m"
    else:
        mag_ylabel = "Relative SPL [dB"
    mag_ylabel += "]"
    
    # Setup Figure and Subplots
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8.5))
    plt.subplots_adjust(bottom=0.23)
    
    # Plot Time Domain Impulse Response Data
    #   The H1-estimated IR is used only for gate placement
    ax_time.plot(time_axis * 1000, H1_impulse, color='b', lw=1, label='IR')
    ax_time.set_title(f"Impulse Response: {args.impulse_data} (Time Domain)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True)
    
    initial_gate_time_ms = time_axis[-1] * 1000
    initial_gate_idx = len(H1_impulse)
    gate_line = ax_time.axvline(x=initial_gate_time_ms, color='g', linestyle='--', lw=1.5, label='Window End')
    
    # State variables
    state = {
        'gate_idx': initial_gate_idx,
        'selecting_gate': False,
        'harmonic_results': None,
        'selected_harmonic': 'H1',
        'hd_max': 3,
        'freq_line_plot': None,
        'showing_phase': False,
        'using_min_phase': False,
        'tukey_alpha': 0.25,
        'resolution': None
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
    
    # Row widgets (Tukey width, HD Max, Phase Type)
    ax_box_tukey = plt.axes([0.11, 0.10, 0.05, 0.03])
    textbox_tukey = TextBox(ax_box_tukey, 'TUKEY: ', initial=str(state['tukey_alpha']), textalignment='center')

    ax_box_hd_max = plt.axes([0.28, 0.10, 0.04, 0.03])
    textbox_hd_max = TextBox(ax_box_hd_max, 'HD MAX: ', initial=str(state['hd_max']), textalignment='center')

    ax_box_phase_type = plt.axes([0.50, 0.10, 0.13, 0.03])
    initial_text = 'minimum phase' if state['using_min_phase'] else 'regular phase'
    phase_textbox = TextBox(ax_box_phase_type, 'PHASE TYPE: ', initial=initial_text, textalignment='center')
    phase_textbox.set_active(False)

    # Replacement controls for RadioButtons: Left Button, Dataset Textbox, Right Button
    ax_btn_prev = plt.axes([0.72, 0.10, 0.035, 0.03])
    ax_box_dataset = plt.axes([0.760, 0.10, 0.05, 0.03])
    ax_btn_next = plt.axes([0.815, 0.10, 0.035, 0.03])

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
            textbox_tukey.set_val(str(state['tukey_alpha']))

    textbox_tukey.on_submit(submit_tukey_alpha)

    def submit_hd_max(text):
        try:
            val = int(text)
            if 2 <= val <= 6:
                state['hd_max'] = val
                print(f"HD MAX updated to: {state['hd_max']}")
                # Reset to H1 and refresh plot
                state['selected_harmonic'] = 'H1'
                textbox_dataset.set_val('H1')
                if state['harmonic_results'] is not None:
                    refresh_frequency_plot()
            else:
                print("Error: HD MAX must be between 2 and 6.")
                textbox_hd_max.set_val(str(state['hd_max']))
        except ValueError:
            textbox_hd_max.set_val(str(state['hd_max']))

    textbox_hd_max.on_submit(submit_hd_max)

    def on_click_time_plot(event):
        if state['selecting_gate'] and event.inaxes == ax_time:
            clicked_time_ms = event.xdata
            if clicked_time_ms is not None and clicked_time_ms >= 0:
                state['gate_idx'] = int((clicked_time_ms / 1000.0) * sample_rate)
                state['gate_idx'] = max(10, min(state['gate_idx'], len(H1_impulse)))
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
        
    def refresh_frequency_plot():
            if state['harmonic_results'] is None:
                return

            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

            results = state['harmonic_results']
            sel = state['selected_harmonic']

            if state['showing_phase']:
                if sel == 'ALL' or sel == 'THD':
                    ax_freq.set_ylabel("Phase (degrees)")
                    ax_freq.set_title("Phase View (Blank for ALL/THD)")
                    state['freq_line_plot'] = []
                else:
                    k_idx = int(sel[1])
                    if k_idx <= state['hd_max']:
                        current_phase = results[k_idx]['min_phase_deg'] if state['using_min_phase'] else results[k_idx]['phase_deg']
                        title_label = f"Harmonic {k_idx} - {'Minimum Phase' if state['using_min_phase'] else 'Phase'}"
                        line, = ax_freq.plot(results[k_idx]['freqs'], current_phase, color='g', lw=1.2)
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
                            l, = ax_freq.plot(freqs, results[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='--', label=f'H{k}')
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
                    k_idx = int(sel[1])
                    if k_idx <= state['hd_max']:
                        line, = ax_freq.plot(results[k_idx]['freqs'], results[k_idx]['mag_db'], color='r', lw=1.2)
                        ax_freq.set_ylabel(mag_ylabel)
                        ax_freq.set_title(f"Frequency Domain Data - Magnitude (Harmonic {k_idx})")
                        state['freq_line_plot'] = [line]

                ax_freq.set_xlabel(f'Frequency (Hz) at {state["resolution"]:.3g} Hz resolution')

            fig.canvas.draw_idle()

    def handle_calc_button(event):
        if state['harmonic_results'] is not None:
            ax_freq.clear()
            ax_freq.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax_freq.set_xscale('log')
            ax_freq.set_xlim(20, sample_rate / 2.0)
            ax_freq.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
            ax_freq.set_ylabel(mag_ylabel)
            ax_freq.set_title("Frequency Domain Data (Cleared)")
            ax_freq.set_xlabel('Frequency (Hz)')
            fig.canvas.draw_idle()
            state['harmonic_results'] = None
            state['resolution'] = None
            print("Lower plot cleared. All calculated data cleared.")
        else:
            print(f"Computing Farina Method response up to HD {state['hd_max']}...")
            results = run_farina_analysis(
                recorded_audio=mic_data,
                sample_rate=sample_rate,
                inv_filter=inv_filter,
                start_freq=20.0,
                gate_end_idx=state['gate_idx'],
                tukey_alpha=state['tukey_alpha'],
                mic_cal_file=args.mic_cal,
                mic_sensitivity=args.mic_sensitivity,
                v_stimulus=args.v_stimulus,
                distance=args.distance,
                max_harmonic=state['hd_max']
            )
            state['harmonic_results'] = results
            state['resolution'] = results[1]['delta_f']
            refresh_frequency_plot()
            print(f"Analysis complete! Resolution: {state['resolution']:.3g} Hz")

    def handle_view_button(event):
        if state['harmonic_results'] is not None:
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
        if state['harmonic_results'] is not None and state['showing_phase']:
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
        if state['harmonic_results'] is not None:
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
        if state['harmonic_results'] is not None:
            refresh_frequency_plot()

    def handle_save_exit_button(event):
            # Ensure data is computed
            if state['harmonic_results'] is None:
                print("Computing response before saving...")
                results = run_farina_analysis(
                    recorded_audio=mic_data,
                    sample_rate=sample_rate,
                    inv_filter=inv_filter,
                    start_freq=20.0,
                    gate_end_idx=state['gate_idx'],
                    tukey_alpha=state['tukey_alpha'],
                    mic_cal_file=args.mic_cal,
                    mic_sensitivity=args.mic_sensitivity,
                    v_stimulus=args.v_stimulus,
                    distance=args.distance,
                    max_harmonic=state['hd_max']
                )
                state['harmonic_results'] = results
                state['resolution'] = results[1]['delta_f']

            res = state['harmonic_results']
            base_name = args.output  # args.output holds the base filename

            # Save FRD files for each harmonic up to HD MAX
            for k in range(1, state['hd_max'] + 1):
                if k in res:
                    h_data = res[k]
                    phase_to_save = h_data['min_phase_deg'] if state['using_min_phase'] else h_data['phase_deg']
                    filename_k = f"{base_name}_HD{k}.frd"
                    write_frd_file(filename_k, h_data['freqs'], h_data['mag_db'], phase_to_save)

            # Save Plots at 600 DPI (PNG format)
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
            phase_to_plot = res[1]['min_phase_deg'] if state['using_min_phase'] else res[1]['phase_deg']
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
                    ax_h.plot(freqs, res[k]['mag_db'], color=colors[idx % len(colors)], lw=1.0, linestyle='--', label=f'H{k}')
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

    