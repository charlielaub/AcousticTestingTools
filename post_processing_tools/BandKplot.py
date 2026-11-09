import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

# COMMAND LINE ARGUMENT PARSING FOR DATA FILE
parser = argparse.ArgumentParser(description="Plot loudspeaker frequency response from a text file.")
parser.add_argument("filename", type=str, help="Path to the data file containing frequency, SPL, and phase.")

# Use parse_known_args() so it doesn't crash if run inside interactive environments like Jupyter/IPython
args, unknown = parser.parse_known_args()

# 1. GLOBAL KEYBOARD SHORTCUT PURGE
plt.rcParams['keymap.back'] = []
plt.rcParams['keymap.forward'] = []
plt.rcParams['keymap.home'] = []
plt.rcParams['keymap.xscale'] = []
plt.rcParams['keymap.yscale'] = []
plt.rcParams['keymap.pan'] = []
plt.rcParams['keymap.zoom'] = []

# 2. CHOOSE VINTAGE B&K HISTORICAL COLOR PALETTE
USE_RED_PALETTE = False

if USE_RED_PALETTE:
    BG_COLOR   = '#FDFBF7'  # Vintage parchment cream
    MINOR_GRID = '#F5D1C3'  # 1 dB Salmon pink subdivisions
    MAJOR_GRID = '#D86B5A'  # 5/10 dB Bold rusty coral
    PEN_TRACE  = '#0A2540'  # Deep navy mechanical ink pen
else:
    BG_COLOR   = '#FAFAF4'  # Technical off-white
    MINOR_GRID = '#CCE3D3'  # 1 dB Pale mint subdivisions
    MAJOR_GRID = '#528B6A'  # 5/10 dB Forest green framework
    PEN_TRACE  = '#1A1A1A'  # Dense charcoal-black ink pen

# 3. FILE IMPORT FUNCTION WITH HEADER SKIPPING & 3-COLUMN PARSING
def load_frequency_data(filepath):
    frequencies = []
    spl_values = []
    phase_values = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reading_data = False
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # Skip blank lines
            
            # Normalize comma or whitespace delimiters into standard tokens
            tokens = [token for token in stripped_line.replace(',', ' ').split() if token]
            
            if not reading_data:
                # Check if this line consists of exactly 3 numeric columns
                if len(tokens) == 3:
                    try:
                        f_val = float(tokens[0])
                        s_val = float(tokens[1])
                        p_val = float(tokens[2])
                        # Valid data row found; switch state and record values
                        reading_data = True
                        frequencies.append(f_val)
                        spl_values.append(s_val)
                        phase_values.append(p_val)
                        continue
                    except ValueError:
                        # Columns couldn't be parsed as numbers; treat as header and skip
                        pass
                else:
                    # Not 3 columns; treat as header and skip
                    pass
            else:
                # We are past the header; read data rows
                if len(tokens) >= 3:
                    try:
                        frequencies.append(float(tokens[0]))
                        spl_values.append(float(tokens[1]))
                        phase_values.append(float(tokens[2]))
                    except ValueError:
                        pass # Skip malformed lines within data section
                        
    return np.array(frequencies), np.array(spl_values), np.array(phase_values)

# Load data using the path provided via command line
try:
    frequencies, mock_spl, mock_phase = load_frequency_data(args.filename)
    if len(frequencies) == 0:
        raise ValueError("No valid 3-column data found in the file.")
except Exception as e:
    print(f"Error reading data file: {e}")
    sys.exit(1)

# 4. INITIALIZE FIGURE WITH THE ORIGINAL 5.5" x 2.5" B&W ASPECT RATIO
fig, ax = plt.subplots(figsize=(5.5, 2.5), dpi=150, facecolor=BG_COLOR)

# Fetch the name of this script file, then use that for the window title
program_name = Path(__file__).stem
fig.canvas.manager.set_window_title(program_name)

ax.set_facecolor(BG_COLOR)

# Plot response trace curve (SPL versus Frequency)
ax.plot(frequencies, mock_spl, color=PEN_TRACE, linewidth=1.2, alpha=0.9, zorder=3)

# 5. CONSTRAIN GRID AND DIVISIONS (50 dB Window, 5 dB Majors, 1 dB Minors)
X_MIN, X_MAX = 20, 20000
Y_SPAN = 50

ax.set_xscale('log')
ax.set_xlim(X_MIN, X_MAX)

# Center the initial 50 dB view cleanly around the data baseline average
data_center = float(np.mean(mock_spl))
ax.set_ylim(data_center - (Y_SPAN / 2), data_center + (Y_SPAN / 2))

# Assign the exact physical chart paper increments
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# Format the grid layers using the historical B&K color schema
ax.grid(True, which='minor', color=MINOR_GRID, linestyle='-', linewidth=0.4, zorder=1)
ax.grid(True, which='major', color=MAJOR_GRID, linestyle='-', linewidth=0.8, zorder=2)

# Custom frequency ticks mirroring classic calibration cards
ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

# Color match spines and text markers directly to the grid boundaries
for spine in ax.spines.values():
    spine.set_color(MAJOR_GRID)
    spine.set_linewidth(0.8)

ax.tick_params(axis='both', which='both', colors=MAJOR_GRID, labelsize=7)
ax.set_xlabel('Frequency (Hz)', fontsize=8, labelpad=2, color=MAJOR_GRID)
ax.set_ylabel('Sound Pressure Level (dB)', fontsize=8, labelpad=2, color=MAJOR_GRID)
plt.title('FREQUENCY RESPONSE', fontsize=8, fontweight='bold', pad=5, color=MAJOR_GRID)

# 6. CUSTOM TOOLBAR TO LOCK X-AXIS, RESTRICT PAN TO Y-AXIS, AND DISABLE ZOOM/SCROLL
class VerticalOnlyToolbar(NavigationToolbar2Tk):
    def zoom(self, *args, **kwargs):
        # Effectively disables the zoom-to-rectangle tool
        pass

    def drag_pan(self, event):
        # Perform standard pan calculations
        super().drag_pan(event)
        
        # Force lock the X limits back to original bounds
        ax.set_xlim(X_MIN, X_MAX)
        
        # Enforce strict 50 dB span on the Y-axis
        y_min, y_max = ax.get_ylim()
        y_center = (y_min + y_max) / 2
        ax.set_ylim(y_center - (Y_SPAN / 2), y_center + (Y_SPAN / 2))
            
        fig.canvas.draw_idle()

# Replace default toolbar implementation with the custom vertical-only toolbar
fig.canvas.manager.toolbar.destroy()
toolbar = VerticalOnlyToolbar(fig.canvas, fig.canvas.manager.window)

# Wipe scroll-wheel hook signals entirely to eliminate zooming/scrolling out of bounds
def disable_scroll(event): 
    return
fig.canvas.mpl_connect('scroll_event', disable_scroll)

# 7. RENDER DESKTOP APPMATRIX WINDOW
plt.tight_layout()
plt.show()