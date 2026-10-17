"""
GPDP - Groundplane Measurement Data Processing
Description: This program performs various corrections to data generated using  
    groundplane-based acoustic measurement techniques.

Author: Charlie Laub
Date: October 2026
"""

import numpy as np                    #the numpy module must be installed
import pandas as pd                   #the pandas module must be installed
import matplotlib.pyplot as plt       #the matplotlib module must be installed
import matplotlib.ticker as ticker 
from matplotlib.widgets import TextBox, Button
import tkinter as tk
from tkinter import filedialog
import argparse, textwrap


def load_frd(filepath):
    """Reads an FRD file and returns frequency and complex pressure."""
    # Read file ignoring lines starting with * (comments)
    df = pd.read_csv(
        filepath,
        engine='python',
        sep=r"\s+",
        comment="*",
        header=None,
        names=["Freq", "Mag_dB", "Phase_deg"],
    )

    # Convert dB to linear magnitude
    mag_lin = 10 ** (df["Mag_dB"] / 20)
    # Convert phase from degrees to radians
    phase_rad = np.radians(df["Phase_deg"])

    # Create complex number: text = mag * e^(j * phase)
    complex_val = mag_lin * np.exp(1j * phase_rad)

    return df["Freq"].values, complex_val


def save_frd(filepath, freq, complex_val):
    """Converts complex values back to Mag(dB)/Phase(deg) and saves as FRD."""
    mag_lin = np.abs(complex_val)
    # Avoid log10(0) by replacing zeros with a tiny number
    mag_lin = np.where(mag_lin == 0, 1e-12, mag_lin)

    mag_dB = 20 * np.log10(mag_lin)
    phase_deg = np.degrees(np.angle(complex_val))

    # Write to file with standard FRD headers
    with open(filepath, "w") as f:
        f.write("* Frequency(Hz) Magnitude(dB) Phase(deg)\n")
        for freq, mag, phase in zip(freq, mag_dB, phase_deg):
            f.write(f"{freq:<12.4f} {mag:<12.4f} {phase:<12.4f}\n")


def align_and_interpolate(freq_target, freq_source, p_source):
    """
    Checks if frequencies align. If not, interpolates source data (Mag/Phase)
    onto the target frequency grid using numpy.interp.
    """
    if freq_target.size == freq_source.size:
        if np.allclose(freq_target, freq_source):
            return p_source
    print("Interpolating FILE2 data to the FILE1 frequency grid...")

    # Deconstruct complex values back to dB and Phase for stable interpolation
    mag_dB = 20 * np.log10(np.abs(p_source))
    phase_deg = np.degrees(np.angle(p_source))
    # Unwrap phase component, phase in degrees
    phase_deg = -np.unwrap(-phase_deg, period=360)
    # Perform linear interpolation directly using np.interp
    mag_dB_interp = np.interp(freq_target, freq_source, mag_dB)
    phase_deg_interp = np.interp(freq_target, freq_source, phase_deg)
    # Wrap the interpolated phase back to the [-180, 180) interval
    phase_deg_interp = ((phase_deg_interp + 180) % 360) - 180
    # Reconstruct complex numbers on the target frequency grid
    mag_lin_interp = 10 ** (mag_dB_interp / 20)
    phase_rad_interp = np.radians(phase_deg_interp)
    print("   ...interpolation complete.")
    #return the complex pressure variable
    return mag_lin_interp * np.exp(1j * phase_rad_interp)


def plot_acoustic_data(freq_file1, result_complex):
    """
    Plots Sound Pressure Level (dB) vs Frequency (Hz) on a log-x scale.
    Dynamically recalculates 6 dB acoustic grid increments during a live zoom/pan.
    """
    pressure_magnitude = np.abs(result_complex)
    sound_pressure_db = 20 * np.log10(np.maximum(pressure_magnitude, 1e-12))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(bottom=0.25)
    
    ax.plot(freq_file1, sound_pressure_db, label="Sound Pressure")
    ax.set_xscale('log')
    
    initial_title = "Frequency Response"
    ax.set_title(initial_title, fontsize=16)
    
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sound Pressure Level [dB]")

    # Enable both major and minor grid styling
    ax.grid(True, which='major', ls='-', color='gray', alpha=0.6, linewidth=0.8)
    ax.grid(True, which='minor', ls=':', color='gray', alpha=0.4, linewidth=0.5)
    ax.grid(True, which='both', axis='x', ls='--', color='gray', alpha=0.4)

    # --- NEW: Callback function to update grid divisions during a zoom/pan ---
    def update_y_grid(changed_ax):
        # Temporarily disconnect to prevent an infinite recursion loop
        changed_ax.callbacks.disconnect(cid)
        
        # Get the new view limits after the zoom operation
        y_min, y_max = changed_ax.get_ylim()
        y_span = y_max - y_min

        # Dynamically scale choices based on the new visible viewport width
        if y_span > 72:
            major_step, minor_step = 12.0, 6.0
        elif y_span > 36:
            major_step, minor_step = 6.0, 3.0
        elif y_span > 12:
            major_step, minor_step = 3.0, 1.0
        else:
            major_step, minor_step = 1.0, 0.25

        # Re-apply the new dynamic subdivisions
        changed_ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))
        changed_ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        
        # Re-attach the event listener callback
        reconnect()

    def reconnect():
        global cid
        cid = ax.callbacks.connect('ylim_changed', update_y_grid)

    # Trigger the initial grid evaluation manually
    reconnect()
    update_y_grid(ax)

    # Setup standard UI input controls
    ax_box = fig.add_axes([0.15, 0.05, 0.35, 0.065])
    ax_update_btn = fig.add_axes([0.53, 0.05, 0.12, 0.065])
    ax_save_btn = fig.add_axes([0.68, 0.05, 0.12, 0.065])
    ax_close_btn = fig.add_axes([0.83, 0.05, 0.09, 0.065])
    
    text_box = TextBox(ax_box, 'Title Text: ', initial=initial_title)
    update_button = Button(ax_update_btn, 'Update Title', color='aliceblue', hovercolor='palegreen')
    save_button = Button(ax_save_btn, 'Save Image', color='aliceblue', hovercolor='palegreen')
    close_button = Button(ax_close_btn, 'Close', color='aliceblue', hovercolor='palegreen')
    
    def submit(event):
        new_title = text_box.text
        ax.set_title(new_title, fontsize=16)
        fig.canvas.draw_idle()
        
    update_button.on_clicked(submit)
    
    def save_plot(event):
        ax_box.set_visible(False)
        ax_update_btn.set_visible(False)
        ax_save_btn.set_visible(False)
        ax_close_btn.set_visible(False)
        fig.canvas.draw()
        
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")]
        )
        
        if file_path:
            fig.savefig(file_path, bbox_inches='tight', pad_inches=0.5, dpi=600)
            print(f"Plot saved to: {file_path}")
            
        ax_box.set_visible(True)
        ax_update_btn.set_visible(True)
        ax_save_btn.set_visible(True)
        ax_close_btn.set_visible(True)
        fig.canvas.draw_idle()

    save_button.on_clicked(save_plot)
    
    def close_window(event):
        plt.close(fig)
    
    close_button.on_clicked(close_window)    
    
    plot_acoustic_data.text_box = text_box
    plot_acoustic_data.update_button = update_button
    plot_acoustic_data.save_button = save_button
    plot_acoustic_data.close_button = close_button

    plt.show()




def main():
    parser = argparse.ArgumentParser(
        description="This program performs various corrections to data generated using groundplane-based acoustic measurement techniques.",
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # File arguments
    parser.add_argument(
        "--infile1", "--f1", type=str, required=True, help="Path and Filename of the first input file."
    )
    parser.add_argument(
        "--infile2", "--f2", type=str, required=True, help="Path and Filename of the second input file."
    )
    parser.add_argument(
        "--outfile", "--of", type=str, required=True, help="Path and Filename of the output file."
    )
    # Calculation type argument with restricted choices
    parser.add_argument(
        "--calc_type", "-c",
        type=int,
        required=True,
        choices=[1, 2, 3, 4], 
        help=textwrap.dedent('''\
        Indicate the type of calculation to perform by number:
        1 : Calculate the Groundplane Transfer Function as infile1 / infile2 - 1 
        2 : Calculate the Free-Field Response from the Lift and Tuck Data in infile 1 using the Groundplane Transfer Function in infile2
        3 : Calculate the Reflection Response from the MMMM Measurement 2 Data in infile1 using the Groundplane Transfer Function in infile2
        4 : Calculate the Free-Field Response from the MMMM Measurement 1 Data in infile1 by subtracting the Reflection Response in infile2
        NOTE: All input and output files should be in the FRD file format.
         ''')
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    try:
        # Load the files and return frequency 
        # Returns the pressure as a complex value 
        freq_file1, file1_pressure = load_frd(args.infile1)
        freq_file2, file2_pressure = load_frd(args.infile2)

        # Align file2_pressure to the freq_file1 frequency grid (if necessary)
        # Returns a complex variable
        file2_pressure = align_and_interpolate(freq_file1, freq_file2, file2_pressure)

        # Route execution based on the chosen calculation type
        if args.calc_type == 1:
            print(f"Calculating the Groundplane Transfer Function from files {file1} and {file2}")
            print(f"Performing the calculation: RESULT = FILE1 / FILE2 - 1")
            result_complex = file1_pressure / file2_pressure - 1.0
            message = f"The Groundplane Transfer function will now be saved to the file {args.file2}."
        elif args.calc_type == 2:
            print(f"Calculating the Free-Field Response from the Lift and Tuck Data in {args.infile1} using the Groundplane Transfer Function in {args.infile2}")
            print(f"Performing calculation: RESULT = FILE1 / (1 + FILE2)")
            result_complex = file1_pressure / (1.0 + file2_pressure)
            message = f"The Free-Field Data will now be saved to the file {args.outfile}."
        elif args.calc_type == 3:
            print(f"Calculating Reflection Response from the MMMM Measurement 2 Data in {args.infile1} using the Groundplane Transfer Function in {args.infile2}")
            print(f"Performing the calculation: RESULT = (FILE1 * FILE2) / (1 + FILE2)")
            result_complex = (file1_pressure * file2_pressure) / (1.0 + file2_pressure)
            message = f"The Reflection Data will now be saved to the file {args.outfile}."
        elif args.calc_type == 4:
            print(f"Calculating the Free-Field Response from the MMMM Measurement 1 Data in {args.infile1} by subtracting the Reflection Data in {args.infile2}")
            print(f"Performing the calculation: RESULT = FILE1 - FILE2")
            result_complex = file1_pressure - file2_pressure
            message = f"The Free-Field Data will now be saved to the file {args.outfile}." 

        #plot the result_complex data unless --no-plot was specified
        if not args.no_plot:
            print(f"Plotting the RESULT dataset...")
            plot_acoustic_data(freq_file1, result_complex)

        # Save result to file
        print(message)
        save_frd(args.outfile, freq_file1, result_complex)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
