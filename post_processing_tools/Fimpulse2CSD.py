import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize

class WaterfallAnalyzer:
    def __init__(self, filepath, config):
        self.filepath = filepath
        self.config = config
        
        # Load and prepare data
        self.fs, self.raw_ir = wav.read(filepath)
        # Convert to mono float if stereo/int
        if self.raw_ir.ndim > 1:
            self.raw_ir = self.raw_ir[:, 0]
        self.raw_ir = self.raw_ir.astype(np.float64)
        self.raw_ir /= np.max(np.abs(self.raw_ir)) # Normalize
        
        self.align_and_window()
        self.precompute_csd()
        self.precompute_pcsd()
        
        self.view_3d = True
        self.setup_gui()

    def align_and_window(self):
        """Locates the peak, snaps to a left zero-crossing, and applies the half-Hann window."""
        peak_idx = np.argmax(np.abs(self.raw_ir))
        
        # Convert ms to samples
        left_samp = int((self.config['left_window_ms'] / 1000.0) * self.fs)
        right_samp = int((self.config['right_window_ms'] / 1000.0) * self.fs)
        
        # Find nearest zero-crossing for the left edge to prevent ringing
        search_start = max(0, peak_idx - left_samp)
        pre_peak_data = self.raw_ir[search_start:peak_idx]
        zero_crossings = np.where(pre_peak_data[:-1] * pre_peak_data[1:] <= 0)[0]
        
        if len(zero_crossings) > 0:
            left_samp = peak_idx - (search_start + zero_crossings[-1]) # Grab closest to peak
            
        self.start_idx = max(0, peak_idx - left_samp)
        self.end_idx = min(len(self.raw_ir), peak_idx + right_samp)
        self.ir_sliced = self.raw_ir[self.start_idx : self.end_idx]
        
        # Recalculate actual samples gathered
        left_samp = peak_idx - self.start_idx
        right_samp = self.end_idx - peak_idx
        
        # Create asymmetric window: Rectangular left, Half-Hann right
        rect_win = np.ones(left_samp)
        hann_full = np.hanning(2 * right_samp)
        hann_half = hann_full[right_samp:]
        self.window = np.concatenate((rect_win, hann_half))
        
        # Apply window to the base IR
        self.ir_windowed = self.ir_sliced * self.window

    def precompute_csd(self):
        """Computes the classic Waterfall (Time vs Frequency) using sliding FFT."""
        slice_count = self.config['csd_slices']
        time_span_s = self.config['time_span_ms'] / 1000.0
        time_shifts = np.linspace(0, time_span_s, slice_count)
        
        nfft = 2**int(np.ceil(np.log2(len(self.ir_windowed))))
        self.csd_freqs = np.fft.rfftfreq(nfft, 1/self.fs)
        self.csd_time = time_shifts * 1000.0 # Store in ms
        self.csd_db = np.zeros((slice_count, len(self.csd_freqs)))
        
        for i, shift_s in enumerate(time_shifts):
            shift_samp = int(shift_s * self.fs)
            if shift_samp >= len(self.ir_sliced):
                break
                
            # Slide window forward. Pad with zeros as signal runs out
            chunk = self.ir_sliced[shift_samp : shift_samp + len(self.window)]
            if len(chunk) < len(self.window):
                chunk = np.pad(chunk, (0, len(self.window) - len(chunk)))
            
            chunk_windowed = chunk * self.window
            fft_mag = np.abs(np.fft.rfft(chunk_windowed, n=nfft))
            
            # Convert to dB with safety floor
            fft_mag = np.maximum(fft_mag, 1e-12)
            self.csd_db[i, :] = 20 * np.log10(fft_mag)
            
        # Normalize top of CSD to 0 dB
        self.csd_db -= np.max(self.csd_db)

    def _cwt_morlet(self, data, freqs, w):
        """
        Computes the Continuous Wavelet Transform using a complex Morlet wavelet.
        Replaces scipy.signal.cwt and morlet2 which were removed in SciPy >= 1.15.
        """
        cwt_matrix = np.zeros((len(freqs), len(data)), dtype=complex)
        for i, freq in enumerate(freqs):
            width = w * self.fs / (2 * np.pi * freq)
            M = int(min(10 * width, len(data)))
            
            # Generate complex Morlet wavelet
            n = (np.arange(0, M) - (M - 1.0) / 2.0) / width
            wavelet = np.sqrt(1/width) * np.pi**(-0.25) * np.exp(1j * w * n) * np.exp(-0.5 * n**2)
            
            # CWT is a correlation, which is convolution with the reversed conjugate
            wavelet_rev = np.conj(wavelet)[::-1]
            cwt_matrix[i, :] = signal.fftconvolve(data, wavelet_rev, mode='same')
            
        return cwt_matrix

    def precompute_pcsd(self):
        """Computes Burst Decay (Periods vs Frequency) using Continuous Wavelet Transform."""
        max_time_s = self.config['period_span'] / self.config['freq_min']
        cwt_samples = min(len(self.ir_windowed), int(max_time_s * self.fs) + 1000)
        cwt_data = self.ir_windowed[:cwt_samples]
        
        self.pcsd_freqs = np.logspace(np.log10(self.config['freq_min']), 
                                      np.log10(self.config['freq_max']), 
                                      num=200)
        
        w = self.config['wavelet_cycles']
        cwt_matrix = self._cwt_morlet(cwt_data, self.pcsd_freqs, w)
        
        cwt_mag = np.abs(cwt_matrix)
        cwt_db_raw = 20 * np.log10(np.maximum(cwt_mag, 1e-12))
        
        cwt_db_raw -= np.max(cwt_db_raw)
        
        num_periods = self.config['pcsd_slices']
        self.pcsd_periods = np.linspace(0, self.config['period_span'], num_periods)
        self.pcsd_db = np.zeros((num_periods, len(self.pcsd_freqs)))
        
        time_axis = np.arange(cwt_samples) / self.fs
        for i, freq in enumerate(self.pcsd_freqs):
            period_axis = time_axis * freq
            self.pcsd_db[:, i] = np.interp(self.pcsd_periods, period_axis, cwt_db_raw[i, :])

    def setup_gui(self):
        """Builds the Matplotlib UI."""
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.canvas.manager.set_window_title('Farina CSD & PCSD Analyzer')
        
        plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95, wspace=0.2)
        
        self.draw_plots()
        
        axcolor = 'lightgoldenrodyellow'
        ax_dyn = plt.axes([0.1, 0.15, 0.35, 0.03], facecolor=axcolor)
        ax_time = plt.axes([0.1, 0.1, 0.35, 0.03], facecolor=axcolor)
        ax_per = plt.axes([0.1, 0.05, 0.35, 0.03], facecolor=axcolor)
        
        self.sl_dyn = Slider(ax_dyn, 'Dynamic Range (dB)', 10, 60, valinit=self.config['dynamic_range'], valstep=1)
        self.sl_time = Slider(ax_time, 'CSD Time (ms)', 2, 100, valinit=self.config['time_span_ms'], valstep=1)
        self.sl_per = Slider(ax_per, 'PCSD Periods', 5, 50, valinit=self.config['period_span'], valstep=1)
        
        ax_btn = plt.axes([0.6, 0.08, 0.15, 0.06])
        self.btn_toggle = Button(ax_btn, 'Toggle 2D/3D View')
        
        self.sl_dyn.on_changed(self.update_limits)
        self.sl_time.on_changed(self.update_limits)
        self.sl_per.on_changed(self.update_limits)
        self.btn_toggle.on_clicked(self.toggle_view)
        
        plt.show()

    def draw_plots(self):
        """Renders the plots based on current UI state, toggling visibility to prevent crashes."""
        # Create both sets of axes once at startup
        if not hasattr(self, 'ax1_3d'):
            self.ax1_3d = self.fig.add_subplot(121, projection='3d')
            self.ax2_3d = self.fig.add_subplot(122, projection='3d')
            self.ax1_2d = self.fig.add_subplot(121)
            self.ax2_2d = self.fig.add_subplot(122)
            
            # Set default 3D camera angles
            self.ax1_3d.view_init(elev=25, azim=-70)
            self.ax2_3d.view_init(elev=25, azim=-70)

        # Clear previous data (leaves the axes safely attached to the window)
        self.ax1_3d.cla()
        self.ax2_3d.cla()
        self.ax1_2d.cla()
        self.ax2_2d.cla()

        # Data masking based on frequency bounds
        f_min, f_max = self.config['freq_min'], self.config['freq_max']
        dyn_range = self.sl_dyn.val if hasattr(self, 'sl_dyn') else self.config['dynamic_range']
        
        # -- CSD Plot --
        csd_f_mask = (self.csd_freqs >= f_min) & (self.csd_freqs <= f_max)
        X_csd, Y_csd = np.meshgrid(self.csd_freqs[csd_f_mask], self.csd_time)
        Z_csd = np.clip(self.csd_db[:, csd_f_mask], -dyn_range, 0)
        
        # -- PCSD Plot --
        pcsd_f_mask = (self.pcsd_freqs >= f_min) & (self.pcsd_freqs <= f_max)
        X_pcsd, Y_pcsd = np.meshgrid(self.pcsd_freqs[pcsd_f_mask], self.pcsd_periods)
        Z_pcsd = np.clip(self.pcsd_db[:, pcsd_f_mask], -dyn_range, 0)

        # Apply slider limits for depth
        t_limit = self.sl_time.val if hasattr(self, 'sl_time') else self.config['time_span_ms']
        p_limit = self.sl_per.val if hasattr(self, 'sl_per') else self.config['period_span']
        
        t_idx = np.searchsorted(self.csd_time, t_limit)
        p_idx = np.searchsorted(self.pcsd_periods, p_limit)
        
        # Toggle which layers are visible based on the button state
        self.ax1_3d.set_visible(self.view_3d)
        self.ax2_3d.set_visible(self.view_3d)
        self.ax1_2d.set_visible(not self.view_3d)
        self.ax2_2d.set_visible(not self.view_3d)

        # ==========================================
        # Render 3D View (Classic ARTA Style)
        # ==========================================
        if self.view_3d:
            # Create a "front wall" by duplicating the last time slice and dropping Z to the floor
            X_c = np.vstack((X_csd[:t_idx, :], X_csd[t_idx-1, :]))
            # (Adding a microscopic 1e-4 offset to Y prevents Matplotlib math warnings for vertical walls)
            Y_c = np.vstack((Y_csd[:t_idx, :], Y_csd[t_idx-1, :] + 1e-4))
            Z_c = np.vstack((Z_csd[:t_idx, :], np.full_like(Z_csd[t_idx-1, :], -dyn_range)))

            X_p = np.vstack((X_pcsd[:p_idx, :], X_pcsd[p_idx-1, :]))
            Y_p = np.vstack((Y_pcsd[:p_idx, :], Y_pcsd[p_idx-1, :] + 1e-4))
            Z_p = np.vstack((Z_pcsd[:p_idx, :], np.full_like(Z_pcsd[p_idx-1, :], -dyn_range)))

            self.ax1_3d.plot_surface(X_c, Y_c, Z_c, 
                                  color='white', edgecolor='black', linewidth=0.3, antialiased=True)
            self.ax2_3d.plot_surface(X_p, Y_p, Z_p, 
                                  color='white', edgecolor='black', linewidth=0.3, antialiased=True)
            
            for ax in [self.ax1_3d, self.ax2_3d]:
                ax.set_xscale('log')
                ax.set_zlim(-dyn_range, 0)
                ax.set_zlabel('Amplitude (dB)')
                
            # Formatting (Inverting Y-limits so t=0 is at the back)
            self.ax1_3d.set_ylim(t_limit, 0)
            self.ax2_3d.set_ylim(p_limit, 0)

            self.ax1_3d.set_title('Cumulative Spectral Decay (CSD)')
            self.ax1_3d.set_xlabel('Frequency (Hz)')
            self.ax1_3d.set_ylabel('Time (ms)')
            self.ax1_3d.set_xlim(f_min, f_max)
            
            self.ax2_3d.set_title('Periodic Cumulative Spectral Decay (PCSD)')
            self.ax2_3d.set_xlabel('Frequency (Hz)')
            self.ax2_3d.set_ylabel('Periods (Cycles)')
            self.ax2_3d.set_xlim(f_min, f_max)

        # ==========================================
        # Render 2D View (Hot/Cold Heatmap Style)
        # ==========================================
        else:
            self.ax1_2d.pcolormesh(X_csd[:t_idx, :], Y_csd[:t_idx, :], Z_csd[:t_idx, :], 
                                cmap='Spectral_r', vmin=-dyn_range, vmax=0, shading='gouraud')
            self.ax2_2d.pcolormesh(X_pcsd[:p_idx, :], Y_pcsd[:p_idx, :], Z_pcsd[:p_idx, :], 
                                cmap='Spectral_r', vmin=-dyn_range, vmax=0, shading='gouraud')
            
            for ax in [self.ax1_2d, self.ax2_2d]:
                ax.set_xscale('log')

            # Formatting (Inverting Y-limits so t=0 is at the top/back)
            self.ax1_2d.set_ylim(t_limit, 0)
            self.ax2_2d.set_ylim(p_limit, 0)
            
            self.ax1_2d.set_title('Cumulative Spectral Decay (CSD)')
            self.ax1_2d.set_xlabel('Frequency (Hz)')
            self.ax1_2d.set_ylabel('Time (ms)')
            self.ax1_2d.set_xlim(f_min, f_max)
            
            self.ax2_2d.set_title('Periodic Cumulative Spectral Decay (PCSD)')
            self.ax2_2d.set_xlabel('Frequency (Hz)')
            self.ax2_2d.set_ylabel('Periods (Cycles)')
            self.ax2_2d.set_xlim(f_min, f_max)

        self.fig.canvas.draw_idle()

    def update_limits(self, val):
        """Callback for sliders."""
        self.draw_plots()
        
    def toggle_view(self, event):
        """Callback for 2D/3D toggle button."""
        self.view_3d = not self.view_3d
        self.draw_plots()

# ==========================================
# Configuration and Execution
# ==========================================
if __name__ == '__main__':
    
    # EDIT THIS TO YOUR FARINA WAV FILE PATH
    WAV_FILE = "Fimpulse.wav" 
    
    config = {
        'left_window_ms': 2.0,   
        'right_window_ms': 20.0, 
        'freq_min': 200,         
        'freq_max': 20000,       
        'dynamic_range': 35,     
        'time_span_ms': 20,      
        'csd_slices': 60,        
        'period_span': 30,       
        'pcsd_slices': 60,       
        'wavelet_cycles': 6      
    }

    try:
        app = WaterfallAnalyzer(WAV_FILE, config)
    except FileNotFoundError:
        print(f"Error: Could not find '{WAV_FILE}'. Please update the WAV_FILE variable.")