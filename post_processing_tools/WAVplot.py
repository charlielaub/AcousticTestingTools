"""
WAVplot - Plotting utility for up to four WAV files
Description: This program plots up to four WAV files, provided in a list on the  
     command line. Each plot can be saved as a PNG or PDF file at 600dpi.

Author: Charlie Laub
Date: October 2026
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.io import wavfile
import tkinter as tk
from tkinter import filedialog

def read_mono_wav(filepath):
    """Reads a mono WAV file (supports integer and float PCM) and returns time, data, and rate."""
    rate, audio_data = wavfile.read(filepath)
    
    # Ensure it's mono (if shape has 2 dimensions, take the first channel)
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
        
    n_frames = len(audio_data)
    time = np.linspace(0, n_frames / rate, num=n_frames)
    
    return time, audio_data, rate

def main():
    parser = argparse.ArgumentParser(description="Read and plot up to four mono WAV files with synced axes and interactive save dialogs.")
    parser.add_argument('wav_files', metavar='WAV_FILE', type=str, nargs='+',
                        help='Path to mono WAV file(s) (provide 1 to 4 files)')
    args = parser.parse_args()
    
    if len(args.wav_files) > 4:
        print("Error: You can provide a maximum of 4 WAV files.")
        return

    n_files = len(args.wav_files)
    
    # Create dynamic subplots with shared X and Y axes (change sharey=False if amplitudes vary wildly)
    fig, axes = plt.subplots(n_files, 1, figsize=(10, 2.5 * n_files), sharex=True, sharey=True)
    if n_files == 1:
        axes = [axes]  # Ensure axes is iterable when there's only 1 subplot
        
    # Read and plot each mono WAV file
    for i, filepath in enumerate(args.wav_files):
        try:
            time, data, rate = read_mono_wav(filepath)
            axes[i].plot(time, data, color='tab:blue', linewidth=0.7)
            axes[i].set_title(f"File: {filepath} (Sample Rate: {rate} Hz)")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, linestyle='--', alpha=0.5)
            
            # Only set the X-label on the bottom-most plot for cleanliness when shared
            if i == n_files - 1:
                axes[i].set_xlabel("Time (s)")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return

    # Adjust layout to make room for the button panel at the bottom
    fig.subplots_adjust(bottom=0.25)
    
    # Button configuration
    button_width = 0.15
    button_height = 0.07
    spacing = 0.03
    start_left = 0.1
    
    # List to hold persistent button references and prevent garbage collection
    active_buttons = []
    
    # Create EXIT Button on the far left
    ax_exit = fig.add_axes([start_left, 0.08, button_width, button_height])
    btn_exit = Button(ax_exit, 'EXIT', color='aliceblue', hovercolor='palegreen')
    
    def close_window(event):
        plt.close(fig)
    btn_exit.on_clicked(close_window)
    active_buttons.append(btn_exit)
    
    # Create SAVE Buttons dynamically for each active plot with file dialog prompt
    for i in range(n_files):
        left_pos = start_left + (i + 1) * (button_width + spacing)
        ax_save = fig.add_axes([left_pos, 0.08, button_width, button_height])
        btn_save = Button(ax_save, f'SAVE #{i+1}', color='aliceblue', hovercolor='palegreen')
        
        def make_save_action(index, filepath):
            def save_action(event):
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True) # Bring dialog to front
                
                file_path = filedialog.asksaveasfilename(
                    title=f"Save Plot #{index + 1} As",
                    defaultextension=".png",
                    initialfile=f"waveform_plot_{index + 1}",
                    filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")]
                )
                root.destroy()
                
                if not file_path:
                    print(f"Save cancelled for plot #{index + 1}.")
                    return

                try:
                    time, data, rate = read_mono_wav(filepath)
                    fig_temp, ax_temp = plt.subplots(figsize=(8, 3), dpi=600)
                    ax_temp.plot(time, data, color='tab:blue', linewidth=0.7)
                    ax_temp.set_title(f"File: {filepath} (Sample Rate: {rate} Hz)")
                    ax_temp.set_xlabel("Time (s)")
                    ax_temp.set_ylabel("Amplitude")
                    ax_temp.grid(True, linestyle='--', alpha=0.5)
                    
                    fig_temp.savefig(file_path, dpi=600, bbox_inches='tight')
                    plt.close(fig_temp)
                    print(f"Success: Saved plot #{index + 1} to '{file_path}' at 600 DPI")
                except Exception as e:
                    print(f"Error saving plot #{index + 1}: {e}")
            return save_action
            
        btn_save.on_clicked(make_save_action(i, args.wav_files[i]))
        
        # Append each save button to the persistent list
        active_buttons.append(btn_save)

    plt.show()

if __name__ == '__main__':
    main()