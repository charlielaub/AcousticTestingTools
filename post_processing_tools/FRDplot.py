import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.ticker as ticker
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import colorchooser



def parse_frd_file(filepath):
    """
    Parses an FRD file, skipping comment lines and headers,
    and returns lists for frequency, SPL, and phase.
    """
    freqs = []
    spls = []
    phases = []
    
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines and common comment delimiters (#, !, %, //)
            if not stripped or stripped.startswith(('#', '!', '%', '//')):
                continue
            
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    f_val = float(parts[0])
                    s_val = float(parts[1])
                    p_val = float(parts[2])
                    freqs.append(f_val)
                    spls.append(s_val)
                    phases.append(p_val)
                except ValueError:
                    # Skip lines that cannot be converted to floats
                    continue
                    
    return freqs, spls, phases



import matplotlib.colors as mcolors
from tkinter import colorchooser

def open_legend_editor(lines, labels, ax, canvas):
    """
    Opens a persistent popup window allowing the user to select and update 
    multiple dataset legend labels, plot text, axis labels, and dataset line colors.
    """
    try:
        root = tk.Toplevel()
    except Exception:
        root = tk.Tk()
        
    root.title("Edit Plot Text, Labels & Colors")
    root.geometry("440x500")
    try:
        root.grab_set()  # Make window modal if possible
    except Exception:
        pass
    
    # --- 1. Dataset & Legend Editing Section ---
    tk.Label(root, text="Select Dataset:", font=("Arial", 10, "bold")).pack(anchor='w', padx=20, pady=(15, 2))
    
    combo_var = tk.StringVar(root)
    combo_var.set(labels[0] if labels else "")
    combo = ttk.Combobox(root, textvariable=combo_var, values=list(labels), state="readonly", width=45)
    combo.pack(padx=20, pady=2)
    
    tk.Label(root, text="New Legend Text:", font=("Arial", 9)).pack(anchor='w', padx=20, pady=(5, 2))
    entry_var = tk.StringVar(root)
    if labels:
        entry_var.set(labels[0])
    
    entry = tk.Entry(root, textvariable=entry_var, width=47, font=("Arial", 10))
    entry.pack(padx=20, pady=2)
    entry.focus_set()
    entry.icursor(tk.END)
    
    # Track current line colors during editing
    current_colors = [line.get_color() for line in lines]
    
    def pick_color():
        idx = combo.current()
        if idx < 0:
            sel = combo_var.get()
            if sel in labels:
                idx = labels.index(sel)
        
        if 0 <= idx < len(lines):
            try:
                # Safely convert Matplotlib color (float tuple, name, or hex) to integer RGB (0-255)
                rgb_float = mcolors.to_rgb(current_colors[idx])
                initial_rgb = tuple(int(c * 255) for c in rgb_float)
            except Exception:
                initial_rgb = (0, 0, 0)
                
            # Open native color picker dialog
            color_code = colorchooser.askcolor(title="Choose Dataset Color", initialcolor=initial_rgb)
            if color_code[1]:  # color_code returns ((r, g, b), hex_string)
                current_colors[idx] = color_code[1]

    # Button to open color picker dialog
    tk.Button(root, text="Change Dataset Color...", width=25, command=pick_color, bg="#f0f0f0").pack(padx=20, pady=8)
    
    # Update entry box when selection in combobox changes
    def on_select(event):
        sel = combo_var.get()
        if sel in labels:
            idx = labels.index(sel)
            entry_var.set(labels[idx])
            entry.icursor(tk.END)

    combo.bind("<<ComboboxSelected>>", on_select)
    
    # --- 2. Plot Title & Axis Labels Section ---
    tk.Label(root, text="Plot Layout Labels:", font=("Arial", 10, "bold")).pack(anchor='w', padx=20, pady=(10, 2))
    
    # Plot Title Entry
    tk.Label(root, text="Plot Title:", font=("Arial", 9)).pack(anchor='w', padx=20, pady=(2, 2))
    title_var = tk.StringVar(root, value=ax.get_title())
    title_entry = tk.Entry(root, textvariable=title_var, width=47, font=("Arial", 10))
    title_entry.pack(padx=20, pady=2)
    
    # X-Axis Label Entry
    tk.Label(root, text="X-Axis Label:", font=("Arial", 9)).pack(anchor='w', padx=20, pady=(2, 2))
    xlabel_var = tk.StringVar(root, value=ax.get_xlabel())
    xlabel_entry = tk.Entry(root, textvariable=xlabel_var, width=47, font=("Arial", 10))
    xlabel_entry.pack(padx=20, pady=2)
    
    # Y-Axis Label Entry
    tk.Label(root, text="Y-Axis Label:", font=("Arial", 9)).pack(anchor='w', padx=20, pady=(2, 2))
    ylabel_var = tk.StringVar(root, value=ax.get_ylabel())
    ylabel_entry = tk.Entry(root, textvariable=ylabel_var, width=47, font=("Arial", 10))
    ylabel_entry.pack(padx=20, pady=2)
    
    def close_window():
        try:
            root.grab_release()
        except Exception:
            pass
        root.destroy()

    def on_update():
        sel_label = combo_var.get()
        new_text = entry_var.get().strip()
        
        idx = combo.current()
        if idx < 0 and sel_label in labels:
            idx = labels.index(sel_label)
            
        if 0 <= idx < len(labels):
            if new_text:
                labels[idx] = new_text
                lines[idx].set_label(new_text)
            
            # Apply chosen color to the plot line
            lines[idx].set_color(current_colors[idx])
            
            combo['values'] = list(labels)
            combo_var.set(labels[idx])
            entry_var.set(labels[idx])
            entry.icursor(tk.END)
            
        # Update Plot Title and Axis Labels using configuration constants
        ax.set_title(
            title_var.get(), 
            fontsize=TITLE_FONT_SIZE, 
            fontweight=TITLE_FONT_WEIGHT, 
            fontname=TITLE_FONT_FAMILY,
            pad=TITLE_TO_PLOT_GAP
        )
        ax.set_xlabel(
            xlabel_var.get(), 
            fontsize=LABEL_FONT_SIZE, 
            fontweight=LABEL_FONT_WEIGHT, 
            fontname=LABEL_FONT_FAMILY
        )
        ax.set_ylabel(
            ylabel_var.get(), 
            fontsize=LABEL_FONT_SIZE, 
            fontweight=LABEL_FONT_WEIGHT, 
            fontname=LABEL_FONT_FAMILY
        )        
        # Redraw plot legend and canvas immediately
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        canvas.draw()

    def on_close():
        close_window()
        
    # Buttons Frame (UPDATE and CLOSE)
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=15)
    
    tk.Button(btn_frame, text="UPDATE", width=12, command=on_update, bg="#e0e0e0").pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="CLOSE", width=12, command=on_close, bg="#e0e0e0").pack(side=tk.RIGHT, padx=10)
    
    if isinstance(root, tk.Tk):
        root.mainloop()



def save_plot_file(fig, button_axes):
    """
    Prompts the user for a filename, hides interactive buttons, 
    saves the plot at 600 DPI with a ~1.5 cm margin, and restores buttons.
    """
    root = tk.Tk()
    root.withdraw() # Hide the main tk window
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
        title="Save Plot As"
    )
    root.destroy()
    
    if file_path:
        try:
            # Temporarily hide buttons so they don't appear in the saved image
            for ax_btn in button_axes:
                ax_btn.set_visible(False)
            
            # Save at 600 DPI with ~1.5 cm padding (0.6 inches = ~1.52 cm)
            fig.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0.6)
            
            # Restore button visibility
            for ax_btn in button_axes:
                ax_btn.set_visible(True)
                
            fig.canvas.draw()
            print(f"Successfully saved plot to: {file_path}")
        except Exception as e:
            # Ensure buttons are restored even if saving throws an error
            for ax_btn in button_axes:
                ax_btn.set_visible(True)
            fig.canvas.draw()
            print(f"Error saving plot: {e}")

def setup_dynamic_y_ticks(fig, ax):
    """
    Configures and dynamically updates y-axis major and minor ticks based on current span:
    Valid major increments: 1, 3, 6, 12, 24
    Corresponding minor increments: 0.25, 1, 2, 3, 6
    Target major tick count: between 4 and up
    """
    def update_ticks(axes):
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        
        # Prevent redundant calculations if limits haven't changed
        if getattr(ax, '_last_ylim', None) == (ymin, ymax):
            return
        ax._last_ylim = (ymin, ymax)
        
        if span < 5.0:
            selected_major, selected_minor = 1.0, 0.25
        else:
            candidates = [
                (24.0, 6.0),
                (12.0, 3.0),
                (6.0, 2.0),
                (3.0, 1.0),
                (1.0, 0.25)
            ]
            selected_major, selected_minor = candidates[0]
            for major, minor in candidates:
                num_ticks = int(span / major)
                if num_ticks >= 4:
                    selected_major = major
                    selected_minor = minor
                    break
                    
        ax.yaxis.set_major_locator(ticker.MultipleLocator(selected_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(selected_minor))
        fig.canvas.draw_idle()

    # Trigger tick updates whenever axis limits change (pan, zoom, etc.)
    ax.callbacks.connect('ylim_changed', update_ticks)
    ax.callbacks.connect('xlim_changed', update_ticks)
    
    # Run immediately upon creation so ticks are correctly set when first drawn
    update_ticks(ax)
                    


def main():
    # --- CONFIGURATION of TITLE and AXIS LABEL text ---
    # FONT FAMILY NAMES: DejaVu Sans, Arial, Liberation Sans, Verdana, Tahoma, 
    #   Times New Roman, DejaVu Serif, Liberation Serif, Courier New, 
    #   DejaVu Sans Mono, Liberation Mono
    TITLE_FONT_SIZE = 15
    TITLE_FONT_WEIGHT = 'bold'
    TITLE_FONT_FAMILY = 'Verdana'  # Or 'DejaVu Sans', 'Helvetica', etc.
    TITLE_TO_PLOT_GAP = 15  #controls the width of the gap between title and plot
    LABEL_FONT_SIZE = 12
    LABEL_FONT_WEIGHT = 'normal'
    LABEL_FONT_FAMILY = 'DejaVu Sans'  # Or 'DejaVu Sans', 'Helvetica', etc.

    if len(sys.argv) < 2:
        print("Usage: python plot_frd_interactive.py <file1.frd> [file2.frd ...]")
        sys.exit(1)
        
    filepaths = sys.argv[1:]
    
    # Set up a wider figure (12x6) with left-aligned plot layout and generous right-side space for legends
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.22, top=0.90)
    
    lines = []
    legend_labels = []
    all_spls = []
    
    # Plot each dataset and collect all SPL values
    for path in filepaths:
        if not os.path.exists(path):
            print(f"Warning: File '{path}' not found. Skipping.")
            continue
            
        freqs, spls, phases = parse_frd_file(path)
        if not freqs:
            print(f"Warning: No valid numeric data found in '{path}'. Skipping.")
            continue
            
        default_label = os.path.basename(path)
        legend_labels.append(default_label)
        all_spls.extend(spls)
        
        # Plot Frequency (X) vs SPL (Y)
        line, = ax.plot(freqs, spls, label=default_label, linewidth=1.5)
        lines.append(line)
        
    if not lines:
        print("Error: No valid data plots could be generated from the provided files.")
        sys.exit(1)
        
    # Formatting the plot - X-axis Log Scale configuration with whole number tick labels
    ax.set_xscale('log')
    # set text properties
    ax.set_title(
        'Sound Pressure Level vs Frequency', 
        fontsize=TITLE_FONT_SIZE, 
        fontweight=TITLE_FONT_WEIGHT, 
        fontname=TITLE_FONT_FAMILY,
        pad=TITLE_TO_PLOT_GAP
    )    
    ax.set_xlabel(
        'Frequency (Hz)', 
        fontsize=LABEL_FONT_SIZE, 
        fontweight=LABEL_FONT_WEIGHT, 
        fontname=LABEL_FONT_FAMILY
    )
    ax.set_ylabel(
        'Sound Pressure Level (SPL) [dB]', 
        fontsize=LABEL_FONT_SIZE, 
        fontweight=LABEL_FONT_WEIGHT, 
        fontname=LABEL_FONT_FAMILY
    )
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Initial setup for Y-axis limits based on data span
    y_min, y_max = min(all_spls), max(all_spls)
    span = y_max - y_min
    selected_major = 6.0
    for major, _ in [(24.0, 6.0), (12.0, 3.0), (6.0, 2.0), (3.0, 1.0), (1.0, 0.25)]:
        if (span / major) + 1 >= 4:
            selected_major = major
            break
    ymin_tick = np.floor(y_min / selected_major) * selected_major
    ymax_tick = np.ceil(y_max / selected_major) * selected_major
    ax.set_ylim(ymin_tick, ymax_tick)
    
    # Enable dynamic tick updates on axis limit changes
    setup_dynamic_y_ticks(fig, ax)
    
    # Enable major and minor gridlines for both X and Y axes
    ax.grid(True, which="major", ls="-", alpha=0.7)
    ax.grid(True, which="minor", ls="--", alpha=0.4)
    
    # Place initial legend outside to the right with extra window width
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # Add three evenly distributed buttons below the plot matching the plot's left/right span [0.10 to 0.70]
    # position specifiers: [left, bottom, width, height]
    ax_btn1 = plt.axes([0.08, 0.06, 0.18, 0.05])
    ax_btn2 = plt.axes([0.35, 0.06, 0.14, 0.05])
    ax_btn3 = plt.axes([0.58, 0.06, 0.12, 0.05])   
    
    btn_legend = Button(ax_btn1, 'Change Legend Text', color='#f0f0f0', hovercolor='#e0e0e0')
    btn_save = Button(ax_btn2, 'SAVE PLOT', color='#f0f0f0', hovercolor='#e0e0e0')
    btn_exit = Button(ax_btn3, 'CLOSE', color='#f0f0f0', hovercolor='#e0e0e0')
    
    button_axes = [ax_btn1, ax_btn2, ax_btn3]
    
    # Connect button actions
    btn_legend.on_clicked(lambda event: open_legend_editor(lines, legend_labels, ax, fig.canvas))
    btn_save.on_clicked(lambda event: save_plot_file(fig, button_axes))
    btn_exit.on_clicked(lambda event: (plt.close('all'), sys.exit(0)))
    
    # Trigger default zoom tool. Access the toolbar via the canvas manager
    toolbar = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
    if toolbar is not None:
        toolbar.zoom()

    plt.show()

if __name__ == '__main__':
    main()