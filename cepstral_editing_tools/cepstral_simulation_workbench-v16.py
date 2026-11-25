# -*- coding: utf-8 -*-
import json
import os
import struct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def read_wav_file_plus_info(filename):
    """
    Reads IEEE float (32/64-bit) or PCM integer (8/16/24/32-bit) WAV files,
    peeking at the fmt chunk header to select the exact binary decoder,
    and extracts the audio data (as float64 NumPy array), sample rate,
    and optional custom xDAT metadata into a dictionary.
    """
    with open(filename, "rb") as f:
        # 1. Verify RIFF header
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a valid RIFF WAV file.")

        _file_size = struct.unpack("<I", f.read(4))[0]
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a valid WAVE file.")

        sample_rate = None
        num_channels = 1
        audio_format = None
        bits_per_sample = None
        raw_data_bytes = None
        custom_metadata = {}

        # 2. Traverse file chunks iteratively (peek fmt before decoding data)
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
                audio_format, num_channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                # Handle WAVE_FORMAT_EXTENSIBLE (0xFFFE = 65534)
                if audio_format == 0xFFFE and len(fmt_data) >= 26:
                    sub_format = struct.unpack("<H", fmt_data[24:26])[0]
                    audio_format = sub_format

            elif chunk_id == b"data":
                raw_data_bytes = f.read(chunk_size)

            elif chunk_id == b"xDAT":
                xdat_data = f.read(chunk_size)
                offset = 0
                while offset + 8 <= chunk_size:
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

    if raw_data_bytes is None or audio_format is None:
        raise ValueError("WAV file is missing required 'fmt ' or 'data' chunks.")

    # 3. Decode sample buffer according to peeked fmt header
    if audio_format == 3:  # IEEE Float
        if bits_per_sample == 32:
            audio_data = np.frombuffer(raw_data_bytes, dtype="<f4").astype(np.float64)
        elif bits_per_sample == 64:
            audio_data = np.frombuffer(raw_data_bytes, dtype="<f8").copy()
        else:
            raise ValueError(f"Unsupported IEEE float bit depth: {bits_per_sample}-bit.")
    elif audio_format == 1:  # Linear PCM Integer
        if bits_per_sample == 16:
            audio_data = np.frombuffer(raw_data_bytes, dtype="<i2").astype(np.float64) / 32768.0
        elif bits_per_sample == 24:
            raw_u8 = np.frombuffer(raw_data_bytes, dtype=np.uint8)
            n_samples_total = len(raw_u8) // 3
            raw_u8 = raw_u8[: n_samples_total * 3].reshape(-1, 3)
            # Sign-extend 24-bit little-endian to 32-bit signed integer
            ints = (
                raw_u8[:, 0].astype(np.int32)
                | (raw_u8[:, 1].astype(np.int32) << 8)
                | (raw_u8[:, 2].astype(np.int32) << 16)
            )
            ints = np.where(ints >= 0x800000, ints - 0x1000000, ints)
            audio_data = ints.astype(np.float64) / 8388608.0
        elif bits_per_sample == 32:
            audio_data = np.frombuffer(raw_data_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
        elif bits_per_sample == 8:
            audio_data = (np.frombuffer(raw_data_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported PCM integer bit depth: {bits_per_sample}-bit.")
    else:
        raise ValueError(f"Unsupported WAV format tag: {audio_format} (expected 1=PCM or 3=IEEE Float).")

    if num_channels and num_channels > 1:
        audio_data = audio_data[0::num_channels]

    fmt_desc = f"{'IEEE Float' if audio_format == 3 else 'PCM Int'} {bits_per_sample}-bit, {num_channels}ch"
    custom_metadata["_wav_fmt_desc"] = fmt_desc
    return sample_rate, audio_data, custom_metadata



class MicCalibrationHandler:
    """
    Loads a microphone calibration file once, computes log-frequency trend lines
    for out-of-range extrapolation, and builds an Akima spline interpolator for
    repeated evaluations across the FFT frequency grid.
    """
    def __init__(self, filename):
        self.filename = filename
        self.is_valid = False
        self.f_min = None
        self.f_max = None
        self.num_points = 0
        self.interpolator = None
        self.poly_coeff_low = None
        self.poly_coeff_high = None
        self._load_and_setup()

    def _load_and_setup(self):
        cal_freqs = []
        cal_db = []
        try:
            with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("*", "#", ";")):
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        try:
                            f_val = float(parts[0])
                            db_val = float(parts[1])
                            if f_val > 0.0:
                                cal_freqs.append(f_val)
                                cal_db.append(db_val)
                        except ValueError:
                            continue

            if len(cal_freqs) < 2:
                return

            cal_freqs = np.array(cal_freqs, dtype=np.float64)
            cal_db = np.array(cal_db, dtype=np.float64)

            sort_indices = np.argsort(cal_freqs)
            cal_freqs = cal_freqs[sort_indices]
            cal_db = cal_db[sort_indices]

            # Remove duplicate frequency entries if present (Akima requires strictly increasing x)
            unique_freqs, unique_idx = np.unique(cal_freqs, return_index=True)
            cal_freqs = unique_freqs
            cal_db = cal_db[unique_idx]
            if len(cal_freqs) < 2:
                return

            self.f_min = float(cal_freqs[0])
            self.f_max = float(cal_freqs[-1])
            self.num_points = len(cal_freqs)

            self.interpolator = Akima1DInterpolator(cal_freqs, cal_db)

            # 1. Low-Frequency Trend Fit (< f_min)
            low_data_mask = cal_freqs <= 100.0
            if np.sum(low_data_mask) >= 2:
                fit_freqs = cal_freqs[low_data_mask]
                fit_db = cal_db[low_data_mask]
            else:
                n_pts = min(3, len(cal_freqs))
                fit_freqs = cal_freqs[:n_pts]
                fit_db = cal_db[:n_pts]
            self.poly_coeff_low = np.polyfit(np.log10(fit_freqs), fit_db, 1)

            # 2. High-Frequency Trend Fit (> f_max) with 3-point moving average (needs >= 4 pts for >= 2 smoothed pts)
            hf_data_mask = cal_freqs >= 10000.0
            if np.sum(hf_data_mask) >= 4:
                fit_freqs = cal_freqs[hf_data_mask]
                fit_db = cal_db[hf_data_mask]
                window_size = 3
                weights = np.ones(window_size) / window_size
                smoothed_db = np.convolve(fit_db, weights, mode="valid")
                smoothed_freqs = np.convolve(fit_freqs, weights, mode="valid")
            elif np.sum(hf_data_mask) >= 2:
                smoothed_freqs = cal_freqs[hf_data_mask]
                smoothed_db = cal_db[hf_data_mask]
            else:
                n_pts = min(3, len(cal_freqs))
                smoothed_freqs = cal_freqs[-n_pts:]
                smoothed_db = cal_db[-n_pts:]

            self.poly_coeff_high = np.polyfit(np.log10(smoothed_freqs), smoothed_db, 1)
            self.is_valid = True
        except Exception:
            self.is_valid = False

    def get_corrections(self, freqs):
        """
        Evaluates the Akima spline and log-frequency trend extrapolation
        for a given frequency array (safely guarding f <= 0).
        """
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        if not self.is_valid:
            return np.zeros_like(freqs_arr, dtype=np.float64)

        corrections = np.zeros_like(freqs_arr, dtype=np.float64)
        f_safe = np.maximum(freqs_arr, 1e-3)

        # 1. Low-Frequency Extrapolation (< f_min)
        low_mask = freqs_arr < self.f_min
        if np.any(low_mask) and self.poly_coeff_low is not None:
            corrections[low_mask] = np.polyval(self.poly_coeff_low, np.log10(f_safe[low_mask]))

        # 2. High-Frequency Extrapolation (> f_max)
        high_mask = freqs_arr > self.f_max
        if np.any(high_mask) and self.poly_coeff_high is not None:
            corrections[high_mask] = np.polyval(self.poly_coeff_high, np.log10(f_safe[high_mask]))

        # 3. Core Interior Range (f_min <= freqs <= f_max)
        mid_mask = (freqs_arr >= self.f_min) & (freqs_arr <= self.f_max)
        if np.any(mid_mask):
            corrections[mid_mask] = self.interpolator(freqs_arr[mid_mask])

        return corrections


class CepstralSimulationWorkbench:
    """
    Interactive Cepstral Simulation & Measurement Workbench (v16 - Log-Frequency Binning & Adjustable Freq Limits).
    Tabs:
      1. Data Source/Import (Mode selection & 3-File Farina WAV Deconvolution)
      2. Loudspeaker Model (Sealed/Vented LF, Variable HF, Baffle-Step Shelf, 6 PEQ Bands)
      3. Room Model (3D Gymnasium Geometry, Reflections & Noise Floor)
      4. Cepstrum Editor (Real/Complex Cepstrum, Mag/Phase View, Multi-Point PCHIP & Tail)
      5. SPL Corrections (Hardware Sensitivity, Amp Gain, Distance & Mic Cal Curve)
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Cepstral Simulation & Measurement Workbench (v16)")
        self.root.geometry("1740x1000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Configure Global Large Fonts for High Legibility ---
        self.font_base = ("Calibri", 14)
        self.font_bold = ("Calibri", 14, "bold")
        self.font_italic = ("Calibri", 13, "italic")
        self.font_mono = ("Consolas", 12)
        self.font_scale = ("Calibri", 12)

        self.root.option_add("*Font", self.font_base)

        self.style = ttk.Style(self.root)
        self.style.configure(".", font=self.font_base)
        self.style.configure("TLabel", font=self.font_base)
        self.style.configure("TCheckbutton", font=self.font_base)
        self.style.configure("TRadiobutton", font=self.font_base)
        self.style.configure("TButton", font=self.font_bold)
        self.style.configure("TNotebook.Tab", font=("Calibri", 13, "bold"), padding=[6, 6])
        self.style.configure("TLabelframe.Label", font=self.font_bold)

        # --- Core DSP Grid Parameters ---
        self.fs = 48000
        self.n_fft = 65536
        self._update_dsp_grids()

        # --- WAV Data Storage (Supports Both 3-File Farina & Single Impulse WAV) ---
        self.last_dir = os.getcwd()
        self.wav_input_type = "three_file"   # "three_file" or "single_ir"
        self.wav_meas_path = None
        self.wav_stim_path = None
        self.wav_inv_path = None
        self.wav_single_ir_path = None
        self.ir_meas_raw = None
        self.ir_stim_raw = None
        self.wav_metadata = {}
        self.wav_has_xdat = False
        self.wav_ready = False
        self.detected_onset_idx = 0

        # --- SPL & Microphone Calibration State (Data Processing Mode Post-Processing) ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            script_dir = os.getcwd()
        self.spl_defaults_file = os.path.join(script_dir, "spl_defaults.json")
        self.mic_cal_path = None
        self.mic_cal_handler = None
        self.spl_offset_db = 0.0
        self.spl_factor_val = None
        self.spl_absolute_active = False
        self.spl_mag_ylabel = "Relative SPL [dB]"
        self.spl_manual_widgets = []
        self.spl_hardware_widgets = []

        # --- Calculated Spectra & Cepstra ---
        self.P_direct = None
        self.P_tot = None
        self.P_edited = None

        self.ceps_clean = None
        self.ceps_raw = None
        self.ceps_edited = None

        self.phase_direct_rad = None
        self.phase_tot_rad = None
        self.phase_edited_rad = None

        # Non-destructive edit stack & interactive multi-point PCHIP picker state
        self.edit_stack = []
        self.picking_mode = False
        self.picked_points = []          # List of [x_ms, y_val] in current Y-scale units
        self.active_point_idx = None     # Index into self.picked_points for arrow-key nudging
        self.picking_scale = "linear"    # "linear" or "log"
        self._preview_artists = []       # Matplotlib artists for live preview without zoom reset

        # EWCC (Exponentially Weighted Complex Cepstrum) state
        self.ewcc_auto_beta_min = 0.0
        self.ewcc_beta_max_safe = 11.5
        self.ewcc_widgets = []

        # Lists of mode-specific widgets for enabling/disabling
        self.model_only_widgets = []
        self.wav_only_widgets = []

        # Build UI, load saved SPL defaults if present, and run initial calculation
        self._build_gui()
        self._load_spl_defaults(silent=True)
        self._sync_mode_widget_states()
        self.recalculate_all()

    def _update_dsp_grids(self):
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.fs)
        self.quefrency_ms = (np.arange(self.n_fft) / self.fs) * 1000.0
        self.half_n = self.n_fft // 2 + 1

    # =========================================================================
    # GUI CONSTRUCTION (Left: Dual Plots | Right: 4-Tab Control Notebook)
    # =========================================================================
    def _build_gui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- LEFT FRAME: Dual Plots (~50% width) ---
        plot_container = ttk.Frame(main_paned, width=860)
        main_paned.add(plot_container, weight=10)

        self.fig, (self.ax_freq, self.ax_ceps) = plt.subplots(2, 1, figsize=(8.2, 8.6))
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.07, hspace=0.30)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_container)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.mpl_click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_ceps_click
        )
        self.mpl_scroll_cid = self.canvas.mpl_connect(
            "scroll_event", self._on_ceps_scroll
        )

        # Arrow-key fine nudging for the active PCHIP control point (Shift = 5x finer)
        self.root.bind("<Up>", lambda e: self._on_nudge_key(e, 0, +1, fine=False))
        self.root.bind("<Down>", lambda e: self._on_nudge_key(e, 0, -1, fine=False))
        self.root.bind("<Left>", lambda e: self._on_nudge_key(e, -1, 0, fine=False))
        self.root.bind("<Right>", lambda e: self._on_nudge_key(e, +1, 0, fine=False))
        self.root.bind("<Shift-Up>", lambda e: self._on_nudge_key(e, 0, +1, fine=True))
        self.root.bind("<Shift-Down>", lambda e: self._on_nudge_key(e, 0, -1, fine=True))
        self.root.bind("<Shift-Left>", lambda e: self._on_nudge_key(e, -1, 0, fine=True))
        self.root.bind("<Shift-Right>", lambda e: self._on_nudge_key(e, +1, 0, fine=True))

        # --- RIGHT FRAME: Four-Tab Control Notebook (~50% width) ---
        right_container = ttk.Frame(main_paned, width=840)
        main_paned.add(right_container, weight=10)

        self.notebook = ttk.Notebook(right_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_spk = ttk.Frame(self.notebook, padding=10)
        self.tab_room = ttk.Frame(self.notebook, padding=10)
        self.tab_edit = ttk.Frame(self.notebook, padding=10)
        self.tab_spl = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_data, text="Data Source/Import")
        self.notebook.add(self.tab_spk, text="Loudspeaker Model")
        self.notebook.add(self.tab_room, text="Room Model")
        self.notebook.add(self.tab_edit, text="Cepstrum Editor")
        self.notebook.add(self.tab_spl, text="SPL Corrections")

        self._build_tab_data_source()
        self._build_tab_loudspeaker_model()
        self._build_tab_room_model()
        self._build_tab_cepstrum_editor()
        self._build_tab_spl_corrections()

        # Dedicated Exit Button below the Notebook tabs (visible from all 4 tabs)
        tk.Button(
            right_container,
            text="Exit Workbench",
            command=self.on_closing,
            bg="#333333",
            fg="white",
            font=self.font_bold,
            pady=5
        ).pack(fill=tk.X, pady=(8, 0))

    # -------------------------------------------------------------------------
    # TAB 1: DATA SOURCE / IMPORT
    # -------------------------------------------------------------------------
    def _build_tab_data_source(self):
        # 1. Master Operating Mode Selector & Status Field
        mode_frame = ttk.LabelFrame(self.tab_data, text="Program Operating Mode", padding=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        self.modeling_mode_var = tk.BooleanVar(value=True)
        cb_mode = ttk.Checkbutton(
            mode_frame,
            text="Modeling Mode (Checked = Synthetic Models | Unchecked = WAV Data Processing)",
            variable=self.modeling_mode_var,
            command=self._on_toggle_operating_mode
        )
        cb_mode.pack(anchor="w", pady=2)

        status_row = ttk.Frame(mode_frame)
        status_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(status_row, text="Active Mode:", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 8))
        self.mode_status_var = tk.StringVar(value="MODELING MODE (Synthetic Loudspeaker + Room Model)")
        self.entry_mode_status = tk.Entry(
            status_row,
            textvariable=self.mode_status_var,
            font=self.font_bold,
            state="readonly",
            readonlybackground="#1f4e79",
            fg="white",
            relief=tk.FLAT,
            justify="center"
        )
        self.entry_mode_status.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)

        # 2A. Single Impulse Response WAV Import (e.g., Floor-Subtracted IR)
        single_frame = ttk.LabelFrame(
            self.tab_data,
            text="Option A: Single Impulse Response WAV (e.g., Groundplane-Subtracted IR)",
            padding=8
        )
        single_frame.pack(fill=tk.X, pady=(0, 6))

        s_row = ttk.Frame(single_frame)
        s_row.pack(fill=tk.X, pady=2)
        btn_single_ir = tk.Button(
            s_row, text="Load Single Impulse WAV...", command=self.load_single_impulse_wav,
            bg="#1f4e79", fg="white", font=self.font_bold, width=25, anchor="w", padx=8, pady=3
        )
        btn_single_ir.pack(side=tk.LEFT)
        self.lbl_single_ir_wav = ttk.Label(s_row, text="Not selected", font=self.font_italic)
        self.lbl_single_ir_wav.pack(side=tk.LEFT, padx=10)

        # 2B. 3-File Farina WAV Measurement Import & Deconvolution
        self.wav_frame = ttk.LabelFrame(
            self.tab_data,
            text="Option B: 3-File Farina Measurement (Measurement + Stimulus + Inverse Filter)",
            padding=8
        )
        self.wav_frame.pack(fill=tk.X, pady=(0, 6))

        file_grid = ttk.Frame(self.wav_frame)
        file_grid.pack(fill=tk.X, pady=2)

        btn_meas = tk.Button(
            file_grid, text="1. Select Measurement WAV...", command=self.select_meas_wav,
            bg="#2e7d32", fg="white", font=self.font_bold, width=25, anchor="w", padx=8, pady=2
        )
        btn_meas.grid(row=0, column=0, pady=2, sticky="w")
        self.lbl_meas_wav = ttk.Label(file_grid, text="Not selected", font=self.font_italic)
        self.lbl_meas_wav.grid(row=0, column=1, padx=10, sticky="w")

        btn_stim = tk.Button(
            file_grid, text="2. Select Stimulus WAV...", command=self.select_stim_wav,
            bg="#2e7d32", fg="white", font=self.font_bold, width=25, anchor="w", padx=8, pady=2
        )
        btn_stim.grid(row=1, column=0, pady=2, sticky="w")
        self.lbl_stim_wav = ttk.Label(file_grid, text="Not selected", font=self.font_italic)
        self.lbl_stim_wav.grid(row=1, column=1, padx=10, sticky="w")

        btn_inv = tk.Button(
            file_grid, text="3. Select Inverse Filter WAV...", command=self.select_inv_wav,
            bg="#2e7d32", fg="white", font=self.font_bold, width=25, anchor="w", padx=8, pady=2
        )
        btn_inv.grid(row=2, column=0, pady=2, sticky="w")
        self.lbl_inv_wav = ttk.Label(file_grid, text="Not selected", font=self.font_italic)
        self.lbl_inv_wav.grid(row=2, column=1, padx=10, sticky="w")

        btn_deconv = tk.Button(
            self.wav_frame,
            text="Deconvolve & Load 3-File Farina Measurement",
            command=self.run_three_file_deconvolution,
            bg="#d9534f", fg="white", font=self.font_bold, pady=4
        )
        btn_deconv.pack(fill=tk.X, pady=(5, 2))

        # 3. Impulse Alignment, Windowing & Band-Edge Conditioning Controls
        cond_frame = ttk.LabelFrame(
            self.tab_data,
            text="Impulse Conditioning & Spectral Band Edges (Auto-set if xDAT present)",
            padding=8
        )
        cond_frame.pack(fill=tk.BOTH, expand=True)

        cond_grid = ttk.Frame(cond_frame)
        cond_grid.pack(fill=tk.X, pady=2)

        ttk.Label(cond_grid, text="Onset Threshold (dB):").grid(row=0, column=0, sticky="w")
        self.onset_thresh_db_var = tk.DoubleVar(value=-45.0)
        sp_thresh = tk.Spinbox(
            cond_grid, textvariable=self.onset_thresh_db_var, from_=-80.0, to=-20.0, increment=1.0,
            width=7, font=self.font_bold, command=self._on_wav_window_param_change
        )
        sp_thresh.grid(row=0, column=1, padx=6, pady=2, sticky="w")
        sp_thresh.bind("<Return>", lambda _: self._on_wav_window_param_change())
        sp_thresh.bind("<FocusOut>", lambda _: self._on_wav_window_param_change())

        ttk.Label(cond_grid, text="Fine-Tune (samples):").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.onset_shift_var = tk.IntVar(value=0)
        sp_shift = tk.Spinbox(
            cond_grid, textvariable=self.onset_shift_var, from_=-200, to=200, increment=1,
            width=7, font=self.font_bold, command=self._on_wav_window_param_change
        )
        sp_shift.grid(row=0, column=3, padx=6, pady=2, sticky="w")
        sp_shift.bind("<Return>", lambda _: self._on_wav_window_param_change())
        sp_shift.bind("<FocusOut>", lambda _: self._on_wav_window_param_change())

        ttk.Label(cond_grid, text="IR Window (ms):").grid(row=1, column=0, sticky="w")
        self.ir_win_ms_var = tk.DoubleVar(value=500.0)
        sp_win = tk.Spinbox(
            cond_grid, textvariable=self.ir_win_ms_var, from_=100.0, to=500.0, increment=25.0,
            width=7, font=self.font_bold, command=self._on_wav_window_param_change
        )
        sp_win.grid(row=1, column=1, padx=6, pady=2, sticky="w")
        sp_win.bind("<Return>", lambda _: self._on_wav_window_param_change())
        sp_win.bind("<FocusOut>", lambda _: self._on_wav_window_param_change())

        # Low & High Band Edges (editable when xDAT is missing)
        ttk.Label(cond_grid, text="Low Band Edge (Hz):").grid(row=2, column=0, sticky="w")
        self.band_low_hz_var = tk.DoubleVar(value=20.0)
        self.sp_band_low = tk.Spinbox(
            cond_grid, textvariable=self.band_low_hz_var, from_=5.0, to=500.0, increment=1.0,
            width=7, font=self.font_bold, command=self._on_wav_window_param_change
        )
        self.sp_band_low.grid(row=2, column=1, padx=6, pady=2, sticky="w")
        self.sp_band_low.bind("<Return>", lambda _: self._on_wav_window_param_change())
        self.sp_band_low.bind("<FocusOut>", lambda _: self._on_wav_window_param_change())

        ttk.Label(cond_grid, text="High Band Edge (Hz):").grid(row=2, column=2, sticky="w", padx=(10, 0))
        self.band_high_hz_var = tk.DoubleVar(value=20000.0)
        self.sp_band_high = tk.Spinbox(
            cond_grid, textvariable=self.band_high_hz_var, from_=1000.0, to=22000.0, increment=250.0,
            width=7, font=self.font_bold, command=self._on_wav_window_param_change
        )
        self.sp_band_high.grid(row=2, column=3, padx=6, pady=2, sticky="w")
        self.sp_band_high.bind("<Return>", lambda _: self._on_wav_window_param_change())
        self.sp_band_high.bind("<FocusOut>", lambda _: self._on_wav_window_param_change())

        # xDAT / WAV Metadata Display Box
        ttk.Label(cond_frame, text="WAV Format & xDAT Metadata Status:", font=self.font_bold).pack(anchor="w", pady=(4, 2))
        self.txt_xdat = tk.Text(cond_frame, height=5, font=self.font_mono, bg="#f8f9fa", state=tk.DISABLED)
        self.txt_xdat.pack(fill=tk.BOTH, expand=True)
        self._set_xdat_text(
            "No WAV files loaded yet.\n"
            "Load a Single Impulse WAV (Option A) or 3-File Farina Set (Option B) above."
        )

        self.wav_only_widgets.extend([sp_thresh, sp_shift, sp_win, self.sp_band_low, self.sp_band_high])

    # -------------------------------------------------------------------------
    # TAB 2: LOUDSPEAKER MODEL (Sealed/Vented LF, Variable HF, Baffle Shelf, 6 PEQ)
    # -------------------------------------------------------------------------
    def _build_tab_loudspeaker_model(self):
        # 1. Low-End Enclosure Alignment & Corner Frequency
        lf_frame = ttk.LabelFrame(self.tab_spk, text="Low-Frequency Enclosure Roll-Off (Q = 0.707)", padding=8)
        lf_frame.pack(fill=tk.X, pady=(0, 6))

        lf_row = ttk.Frame(lf_frame)
        lf_row.pack(fill=tk.X, pady=2)

        self.enclosure_type_var = tk.StringVar(value="VENTED")
        rb_sealed = ttk.Radiobutton(
            lf_row, text="SEALED (12 dB/oct, Q=0.707)",
            variable=self.enclosure_type_var, value="SEALED", command=self.recalculate_all
        )
        rb_sealed.pack(side=tk.LEFT, padx=(0, 14))

        rb_vented = ttk.Radiobutton(
            lf_row, text="VENTED (24 dB/oct, Q=0.707)",
            variable=self.enclosure_type_var, value="VENTED", command=self.recalculate_all
        )
        rb_vented.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(lf_row, text="Corner Freq (Hz):", font=self.font_bold).pack(side=tk.LEFT, padx=(8, 4))
        self.lf_fc_var = tk.DoubleVar(value=35.0)
        sp_lf_fc = tk.Spinbox(
            lf_row, textvariable=self.lf_fc_var, from_=10.0, to=250.0, increment=1.0,
            width=7, font=self.font_bold, command=self.recalculate_all
        )
        sp_lf_fc.pack(side=tk.LEFT)
        sp_lf_fc.bind("<Return>", lambda _: self.recalculate_all())
        sp_lf_fc.bind("<FocusOut>", lambda _: self.recalculate_all())

        self.model_only_widgets.extend([rb_sealed, rb_vented, sp_lf_fc])

        # 2. High-End Low-Pass Roll-Off (Variable Frequency & Q)
        hf_frame = ttk.LabelFrame(self.tab_spk, text="High-Frequency Low-Pass Roll-Off (2nd-Order)", padding=8)
        hf_frame.pack(fill=tk.X, pady=6)

        hf_row = ttk.Frame(hf_frame)
        hf_row.pack(fill=tk.X, pady=2)

        ttk.Label(hf_row, text="Corner Freq (Hz):", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 6))
        self.hf_fc_var = tk.DoubleVar(value=16000.0)
        sp_hf_fc = tk.Spinbox(
            hf_row, textvariable=self.hf_fc_var, from_=1000.0, to=22000.0, increment=250.0,
            width=8, font=self.font_bold, command=self.recalculate_all
        )
        sp_hf_fc.pack(side=tk.LEFT, padx=(0, 20))
        sp_hf_fc.bind("<Return>", lambda _: self.recalculate_all())
        sp_hf_fc.bind("<FocusOut>", lambda _: self.recalculate_all())

        ttk.Label(hf_row, text="HF Q Factor:", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 6))
        self.hf_q_var = tk.DoubleVar(value=0.707)
        sp_hf_q = tk.Spinbox(
            hf_row, textvariable=self.hf_q_var, from_=0.30, to=4.00, increment=0.05,
            width=7, font=self.font_bold, command=self.recalculate_all
        )
        sp_hf_q.pack(side=tk.LEFT)
        sp_hf_q.bind("<Return>", lambda _: self.recalculate_all())
        sp_hf_q.bind("<FocusOut>", lambda _: self.recalculate_all())

        self.model_only_widgets.extend([sp_hf_fc, sp_hf_q])

        # 3. 1st-Order Baffle-Step Shelf Filter (LF Step-Down Only)
        bs_frame = ttk.LabelFrame(self.tab_spk, text="1st-Order Baffle-Step Shelf Filter (Low-Frequency Reduction)", padding=8)
        bs_frame.pack(fill=tk.X, pady=6)

        bs_row = ttk.Frame(bs_frame)
        bs_row.pack(fill=tk.X, pady=2)

        self.baffle_enable_var = tk.BooleanVar(value=True)
        cb_bs = ttk.Checkbutton(
            bs_row, text="Enable Baffle Step",
            variable=self.baffle_enable_var, command=self.recalculate_all
        )
        cb_bs.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(bs_row, text="Cutoff Freq (Hz):", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 4))
        self.baffle_fc_var = tk.DoubleVar(value=350.0)
        sp_bs_fc = tk.Spinbox(
            bs_row, textvariable=self.baffle_fc_var, from_=50.0, to=2000.0, increment=10.0,
            width=7, font=self.font_bold, command=self.recalculate_all
        )
        sp_bs_fc.pack(side=tk.LEFT, padx=(0, 14))
        sp_bs_fc.bind("<Return>", lambda _: self.recalculate_all())
        sp_bs_fc.bind("<FocusOut>", lambda _: self.recalculate_all())

        ttk.Label(bs_row, text="Step-Down (dB):", font=self.font_bold).pack(side=tk.LEFT, padx=(0, 4))
        self.baffle_step_db_var = tk.DoubleVar(value=-4.0)
        sp_bs_db = tk.Spinbox(
            bs_row, textvariable=self.baffle_step_db_var, from_=-12.0, to=0.0, increment=0.5,
            width=6, font=self.font_bold, command=self.recalculate_all
        )
        sp_bs_db.pack(side=tk.LEFT)
        sp_bs_db.bind("<Return>", lambda _: self.recalculate_all())
        sp_bs_db.bind("<FocusOut>", lambda _: self.recalculate_all())

        self.model_only_widgets.extend([cb_bs, sp_bs_fc, sp_bs_db])

        # 4. 6 Parametric EQ (PEQ) Bands
        peq_frame = ttk.LabelFrame(self.tab_spk, text="Parametric EQ Bands (Resonances, Crossover Dips & Diffraction Features)", padding=8)
        peq_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        headers = ["Band", "On", "Center Freq (Hz)", "Gain (dB)", "Q Factor"]
        for col, text in enumerate(headers):
            ttk.Label(peq_frame, text=text, font=self.font_bold).grid(row=0, column=col, padx=8, pady=4)

        default_peq = [
            (True,   42.0,  +4.5,  8.0),   # Port / enclosure resonance peak
            (True,   85.0,  -9.0, 12.0),   # Narrow driver/network notch
            (True,  2500.0, -3.0,  6.0),   # Diffraction dip
            (False, 500.0,  +2.0,  4.0),
            (False, 1200.0, -2.5,  5.0),
            (False, 6500.0, +3.0,  7.0),
        ]

        self.peq_bands = []
        for idx, (en, f0, gain, q_val) in enumerate(default_peq, start=1):
            ttk.Label(peq_frame, text=f"PEQ {idx}", font=self.font_bold).grid(row=idx, column=0, padx=8, pady=4, sticky="w")

            en_var = tk.BooleanVar(value=en)
            cb_peq = ttk.Checkbutton(peq_frame, variable=en_var, command=self.recalculate_all)
            cb_peq.grid(row=idx, column=1, padx=8, pady=4)

            f_var = tk.DoubleVar(value=f0)
            sp_f = tk.Spinbox(
                peq_frame, textvariable=f_var, from_=15.0, to=20000.0, increment=5.0,
                width=9, font=self.font_bold, command=self.recalculate_all
            )
            sp_f.grid(row=idx, column=2, padx=8, pady=4)
            sp_f.bind("<Return>", lambda _: self.recalculate_all())
            sp_f.bind("<FocusOut>", lambda _: self.recalculate_all())

            g_var = tk.DoubleVar(value=gain)
            sp_g = tk.Spinbox(
                peq_frame, textvariable=g_var, from_=-24.0, to=24.0, increment=0.5,
                width=7, font=self.font_bold, command=self.recalculate_all
            )
            sp_g.grid(row=idx, column=3, padx=8, pady=4)
            sp_g.bind("<Return>", lambda _: self.recalculate_all())
            sp_g.bind("<FocusOut>", lambda _: self.recalculate_all())

            q_var = tk.DoubleVar(value=q_val)
            sp_q = tk.Spinbox(
                peq_frame, textvariable=q_var, from_=0.5, to=50.0, increment=0.5,
                width=7, font=self.font_bold, command=self.recalculate_all
            )
            sp_q.grid(row=idx, column=4, padx=8, pady=4)
            sp_q.bind("<Return>", lambda _: self.recalculate_all())
            sp_q.bind("<FocusOut>", lambda _: self.recalculate_all())

            self.peq_bands.append({
                "enabled": en_var, "freq": f_var, "gain": g_var, "q": q_var
            })
            self.model_only_widgets.extend([cb_peq, sp_f, sp_g, sp_q])

        # Apply & Reset Loudspeaker Model Buttons
        spk_btn_row = ttk.Frame(self.tab_spk)
        spk_btn_row.pack(fill=tk.X, pady=(6, 0))
        btn_apply_spk = tk.Button(
            spk_btn_row,
            text="Apply Loudspeaker Parameters",
            command=self.recalculate_all,
            bg="#1f4e79", fg="white", font=self.font_bold, pady=4
        )
        btn_apply_spk.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_reset_spk = tk.Button(
            spk_btn_row,
            text="Reset Defaults",
            command=self._reset_loudspeaker_model,
            bg="#595959", fg="white", font=self.font_base, pady=4
        )
        btn_reset_spk.pack(side=tk.LEFT)
        self.model_only_widgets.extend([btn_apply_spk, btn_reset_spk])

    # -------------------------------------------------------------------------
    # TAB 3: ROOM MODEL (Disabled when Modeling Mode is unchecked)
    # -------------------------------------------------------------------------
    def _build_tab_room_model(self):
        # 1. 3D Gymnasium & Transducer Geometry
        geom_frame = ttk.LabelFrame(
            self.tab_room,
            text="3D Gymnasium & Transducer Coordinates (meters)",
            padding=8
        )
        geom_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(geom_frame, text="Room (Lx, Ly, Lz):", font=self.font_bold).grid(row=0, column=0, sticky="w", pady=4)
        self.entry_room = tk.Entry(geom_frame, width=20, font=self.font_bold)
        self.entry_room.insert(0, "30.0, 20.0, 8.0")
        self.entry_room.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(geom_frame, text="Speaker (xs, ys, zs):", font=self.font_bold).grid(row=1, column=0, sticky="w", pady=4)
        self.entry_spk = tk.Entry(geom_frame, width=20, font=self.font_bold)
        self.entry_spk.insert(0, "5.0, 10.0, 1.2")
        self.entry_spk.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(geom_frame, text="Mic (xm, ym, zm):", font=self.font_bold).grid(row=2, column=0, sticky="w", pady=4)
        self.entry_mic = tk.Entry(geom_frame, width=20, font=self.font_bold)
        self.entry_mic.insert(0, "8.0, 10.0, 1.2")
        self.entry_mic.grid(row=2, column=1, sticky="w", padx=8, pady=4)

        btn_geom = tk.Button(
            geom_frame, text="Apply\nGeometry", command=self.recalculate_all,
            bg="#595959", fg="white", font=self.font_bold, padx=12
        )
        btn_geom.grid(row=0, column=2, rowspan=3, padx=10, pady=4, sticky="nsew")

        self.model_only_widgets.extend([self.entry_room, self.entry_spk, self.entry_mic, btn_geom])

        # 2. Master Reflection & Noise Controls
        self.master_frame = ttk.LabelFrame(
            self.tab_room,
            text="Global Room Environment Controls",
            padding=8
        )
        self.master_frame.pack(fill=tk.X, pady=8)

        self.enable_refl_var = tk.BooleanVar(value=True)
        cb_refl = ttk.Checkbutton(
            self.master_frame,
            text="MASTER ENABLE: Synthesize Room Boundary Reflections",
            variable=self.enable_refl_var,
            command=self.recalculate_all
        )
        cb_refl.pack(anchor="w", pady=2)

        noise_row = ttk.Frame(self.master_frame)
        noise_row.pack(fill=tk.X, pady=3)
        self.enable_noise_var = tk.BooleanVar(value=False)
        cb_noise = ttk.Checkbutton(
            noise_row, text="Add Synthetic Noise Floor (dBFS):",
            variable=self.enable_noise_var, command=self.recalculate_all
        )
        cb_noise.pack(side=tk.LEFT)
        self.noise_db_var = tk.DoubleVar(value=-85.0)
        sc_noise = tk.Scale(
            noise_row, variable=self.noise_db_var, from_=-120, to=-40, resolution=1,
            orient=tk.HORIZONTAL, command=lambda _: self.recalculate_all(),
            length=220, font=self.font_scale
        )
        sc_noise.pack(side=tk.RIGHT, padx=6)

        self.model_only_widgets.extend([cb_refl, cb_noise, sc_noise])

        # 3. Per-Boundary Reflectivity & HF Absorption Matrix
        self.surf_frame = ttk.LabelFrame(
            self.tab_room,
            text="Boundary Reflectivity (R0) & HF Absorption Corner (fc)",
            padding=8
        )
        self.surf_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        headers = ["Surface", "On", "R0 (0-1)", "HF Roll-off fc (Hz)"]
        for col, text in enumerate(headers):
            ttk.Label(self.surf_frame, text=text, font=self.font_bold).grid(row=0, column=col, padx=6, pady=3)

        default_surfaces = [
            ("Floor",      True,  0.90, 8000.0, "#d9534f"),
            ("Ceiling",    True,  0.75, 2500.0, "#f0ad4e"),
            ("Left Wall",  True,  0.85, 5000.0, "#5bc0de"),
            ("Right Wall", True,  0.85, 5000.0, "#9370DB"),
            ("Back Wall",  True,  0.80, 4000.0, "#2e7d32"),
            ("Front Wall", True,  0.80, 4000.0, "#ff7f0e"),
        ]

        self.surface_controls = []
        for idx, (name, en, r0, fc, color) in enumerate(default_surfaces, start=1):
            lbl_s = tk.Label(self.surf_frame, text=name, fg=color, font=self.font_bold)
            lbl_s.grid(row=idx, column=0, sticky="w", padx=4)
            en_var = tk.BooleanVar(value=en)
            cb_s = ttk.Checkbutton(self.surf_frame, variable=en_var, command=self.recalculate_all)
            cb_s.grid(row=idx, column=1, padx=4)

            r0_var = tk.DoubleVar(value=r0)
            sc_r0 = tk.Scale(
                self.surf_frame, variable=r0_var, from_=0.0, to=1.0, resolution=0.02,
                orient=tk.HORIZONTAL, command=lambda _: self.recalculate_all(),
                length=165, showvalue=True, font=self.font_scale
            )
            sc_r0.grid(row=idx, column=2, padx=6)

            fc_var = tk.DoubleVar(value=fc)
            sc_fc = tk.Scale(
                self.surf_frame, variable=fc_var, from_=200.0, to=20000.0, resolution=100.0,
                orient=tk.HORIZONTAL, command=lambda _: self.recalculate_all(),
                length=200, showvalue=True, font=self.font_scale
            )
            sc_fc.grid(row=idx, column=3, padx=6)

            self.surface_controls.append({
                "name": name, "enabled": en_var, "r0": r0_var, "fc": fc_var, "color": color
            })
            self.model_only_widgets.extend([cb_s, sc_r0, sc_fc])

    # -------------------------------------------------------------------------
    # TAB 4: CEPSTRUM EDITOR
    # -------------------------------------------------------------------------
    def _build_tab_cepstrum_editor(self):
        # 1. Cepstral Mode & Visualization Controls
        vis_frame = ttk.LabelFrame(self.tab_edit, text="Cepstral Formulation & Plot Options", padding=8)
        vis_frame.pack(fill=tk.X, pady=(0, 8))

        self.ceps_mode_var = tk.StringVar(value="Real")
        ttk.Radiobutton(
            vis_frame, text="Real Cepstrum + Hilbert Min-Phase (Immune to Phase-Wrap)",
            variable=self.ceps_mode_var, value="Real", command=self._on_ceps_mode_change
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            vis_frame, text="Complex Cepstrum with Unwrapped Phase & Optional EWCC",
            variable=self.ceps_mode_var, value="Complex", command=self._on_ceps_mode_change
        ).pack(anchor="w", pady=2)

        # Exponentially Weighted Complex Cepstrum (EWCC) Controls (Active in Complex mode only)
        ewcc_row = ttk.Frame(vis_frame)
        ewcc_row.pack(fill=tk.X, pady=(4, 2), padx=(18, 0))

        self.ewcc_enable_var = tk.BooleanVar(value=False)
        self.cb_ewcc = ttk.Checkbutton(
            ewcc_row,
            text="Exp Weight (EWCC) beta (1/s):",
            variable=self.ewcc_enable_var,
            command=self._on_ewcc_toggle
        )
        self.cb_ewcc.pack(side=tk.LEFT)

        self.ewcc_beta_var = tk.StringVar(value="0.0")
        self.btn_beta_dn = tk.Button(
            ewcc_row, text="▼", width=2, font=self.font_bold,
            command=lambda: self._step_ewcc_beta(-0.5)
        )
        self.btn_beta_dn.pack(side=tk.LEFT, padx=(6, 2))

        self.ent_ewcc_beta = tk.Entry(
            ewcc_row, textvariable=self.ewcc_beta_var, width=6, font=self.font_bold, justify="center"
        )
        self.ent_ewcc_beta.pack(side=tk.LEFT, padx=2)
        self.ent_ewcc_beta.bind("<Return>", lambda _: self._on_ewcc_beta_commit())
        self.ent_ewcc_beta.bind("<FocusOut>", lambda _: self._on_ewcc_beta_commit())

        self.btn_beta_up = tk.Button(
            ewcc_row, text="▲", width=2, font=self.font_bold,
            command=lambda: self._step_ewcc_beta(+0.5)
        )
        self.btn_beta_up.pack(side=tk.LEFT, padx=(2, 8))

        self.btn_ewcc_auto = tk.Button(
            ewcc_row, text="Auto-Suggest", command=self._on_ewcc_auto_suggest,
            bg="#1f4e79", fg="white", font=self.font_base, padx=8, pady=1
        )
        self.btn_ewcc_auto.pack(side=tk.LEFT)

        self.lbl_ewcc_info = tk.Label(
            vis_frame,
            text="EWCC inactive (Select Complex Cepstrum and check Exp Weight to enable)",
            font=self.font_italic, fg="#555555", anchor="w"
        )
        self.lbl_ewcc_info.pack(fill=tk.X, padx=(22, 0), pady=(0, 3))

        self.ewcc_widgets = [
            self.cb_ewcc, self.btn_beta_dn, self.ent_ewcc_beta, self.btn_beta_up, self.btn_ewcc_auto
        ]

        upper_view_row = ttk.Frame(vis_frame)
        upper_view_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(upper_view_row, text="Upper Plot View:", font=self.font_bold).pack(side=tk.LEFT)
        self.upper_plot_mode_var = tk.StringVar(value="Magnitude")
        ttk.Radiobutton(
            upper_view_row, text="Magnitude (dB)",
            variable=self.upper_plot_mode_var, value="Magnitude", command=self.update_plots
        ).pack(side=tk.LEFT, padx=(10, 6))
        ttk.Radiobutton(
            upper_view_row, text="Wrapped Phase (deg)",
            variable=self.upper_plot_mode_var, value="Phase_Wrapped", command=self.update_plots
        ).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(
            upper_view_row, text="Unwrapped Phase (deg)",
            variable=self.upper_plot_mode_var, value="Phase_Unwrapped", command=self.update_plots
        ).pack(side=tk.LEFT, padx=6)

        # Log-Frequency Binning & Frequency Range Controls (Upper Plot + FRD Export)
        logbin_row = ttk.Frame(vis_frame)
        logbin_row.pack(fill=tk.X, pady=(4, 2))

        self.log_bin_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            logbin_row, text="Log-Freq Binning",
            variable=self.log_bin_enable_var, command=self._on_log_bin_param_change
        ).pack(side=tk.LEFT)

        ttk.Label(logbin_row, text="PPO:").pack(side=tk.LEFT, padx=(10, 3))
        self.log_bin_ppo_var = tk.IntVar(value=24)
        self.sp_log_ppo = tk.Spinbox(
            logbin_row, textvariable=self.log_bin_ppo_var, from_=12, to=192, increment=12,
            width=4, font=self.font_bold, command=self._on_log_bin_param_change
        )
        self.sp_log_ppo.pack(side=tk.LEFT)
        self.sp_log_ppo.bind("<Return>", lambda _: self._on_log_bin_param_change())
        self.sp_log_ppo.bind("<FocusOut>", lambda _: self._on_log_bin_param_change())

        ttk.Label(logbin_row, text="Min Hz:").pack(side=tk.LEFT, padx=(12, 3))
        self.freq_min_hz_var = tk.DoubleVar(value=10.0)
        self.sp_freq_min = tk.Spinbox(
            logbin_row, textvariable=self.freq_min_hz_var, from_=5.0, to=1000.0, increment=5.0,
            width=6, font=self.font_bold, command=self._on_log_bin_param_change
        )
        self.sp_freq_min.pack(side=tk.LEFT)
        self.sp_freq_min.bind("<Return>", lambda _: self._on_log_bin_param_change())
        self.sp_freq_min.bind("<FocusOut>", lambda _: self._on_log_bin_param_change())

        ttk.Label(logbin_row, text="Max Hz:").pack(side=tk.LEFT, padx=(10, 3))
        self.freq_max_hz_var = tk.DoubleVar(value=20000.0)
        self.sp_freq_max = tk.Spinbox(
            logbin_row, textvariable=self.freq_max_hz_var, from_=200.0, to=48000.0, increment=500.0,
            width=7, font=self.font_bold, command=self._on_log_bin_param_change
        )
        self.sp_freq_max.pack(side=tk.LEFT)
        self.sp_freq_max.bind("<Return>", lambda _: self._on_log_bin_param_change())
        self.sp_freq_max.bind("<FocusOut>", lambda _: self._on_log_bin_param_change())

        opt_row = ttk.Frame(vis_frame)
        opt_row.pack(fill=tk.X, pady=6)
        self.show_ghost_var = tk.BooleanVar(value=True)
        self.cb_ghost = ttk.Checkbutton(
            opt_row, text="Overlay Reflection-Free Cepstrum",
            variable=self.show_ghost_var, command=self.update_plots
        )
        self.cb_ghost.pack(side=tk.LEFT)
        self.model_only_widgets.append(self.cb_ghost)

        self.show_guides_var = tk.BooleanVar(value=True)
        self.cb_guides = ttk.Checkbutton(
            opt_row, text="Show Room Delay Guides",
            variable=self.show_guides_var, command=self.update_plots
        )
        self.cb_guides.pack(side=tk.LEFT, padx=12)
        self.model_only_widgets.append(self.cb_guides)

        self.show_harmonics_var = tk.BooleanVar(value=True)
        self.cb_harmonics = ttk.Checkbutton(
            opt_row, text="2nd/3rd Rahmonics",
            variable=self.show_harmonics_var, command=self.update_plots
        )
        self.cb_harmonics.pack(side=tk.RIGHT)
        self.model_only_widgets.append(self.cb_harmonics)

        scale_row = ttk.Frame(vis_frame)
        scale_row.pack(fill=tk.X, pady=4)
        ttk.Label(scale_row, text="Quefrency Y-Scale:").pack(side=tk.LEFT)
        self.y_scale_var = tk.StringVar(value="Linear (Auto-Zoom >0.5ms)")
        cb_yscale = ttk.Combobox(
            scale_row, textvariable=self.y_scale_var,
            values=["Linear (Auto-Zoom >0.5ms)", "Log (dB)"],
            state="readonly", width=24, font=self.font_base
        )
        cb_yscale.pack(side=tk.LEFT, padx=8)
        self.y_scale_var.trace_add("write", self._on_yscale_change)

        ttk.Label(scale_row, text="Max X (ms):").pack(side=tk.LEFT, padx=(10, 4))
        self.max_q_ms_var = tk.DoubleVar(value=80.0)
        sp_max_q = tk.Spinbox(
            scale_row, textvariable=self.max_q_ms_var, from_=10.0, to=300.0, increment=10.0,
            width=6, font=self.font_bold, command=self.update_plots
        )
        sp_max_q.pack(side=tk.LEFT)
        sp_max_q.bind("<Return>", lambda _: self.update_plots())
        sp_max_q.bind("<FocusOut>", lambda _: self.update_plots())

        # 2. Interactive Multi-Point PCHIP Curve & Zero Notch Editor
        surg_frame = ttk.LabelFrame(
            self.tab_edit,
            text="Interactive PCHIP & Zero Notch (Click: Add/Select | Right-Click: Delete | Arrows: Nudge | Wheel: Zoom)",
            padding=8
        )
        surg_frame.pack(fill=tk.X, pady=6)

        pick_btn_row = ttk.Frame(surg_frame)
        pick_btn_row.pack(fill=tk.X, pady=(2, 6))

        self.btn_begin_insert = tk.Button(
            pick_btn_row,
            text="Begin Edit",
            command=self._on_begin_or_insert_pchip,
            bg="#1f4e79", fg="white", font=self.font_bold, pady=4
        )
        self.btn_begin_insert.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_zero_notch = tk.Button(
            pick_btn_row,
            text="Zero Notch",
            command=self._on_insert_zero_notch,
            bg="#d9534f", fg="white", font=self.font_bold, pady=4, state="disabled"
        )
        self.btn_zero_notch.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_cancel_edit = tk.Button(
            pick_btn_row,
            text="Cancel",
            command=self._cancel_picking_mode,
            bg="#595959", fg="white", font=self.font_base, pady=4, state="disabled"
        )
        self.btn_cancel_edit.pack(side=tk.LEFT)

        opt_edit_row = ttk.Frame(surg_frame)
        opt_edit_row.pack(fill=tk.X, pady=(0, 2))

        self.comb_rahmonics_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opt_edit_row,
            text="Also Notch 2x & 3x Rahmonics (Zero Notch only)",
            variable=self.comb_rahmonics_var
        ).pack(side=tk.LEFT)

        self.lbl_pick_status = ttk.Label(
            surg_frame,
            text="Click 'Begin Edit' to place 2+ points on the lower cepstrum plot.",
            font=self.font_italic
        )
        self.lbl_pick_status.pack(anchor="w", pady=(3, 0))

        # 3. Analytical Exponential Decay Tail (Compact Arrow Steppers)
        tail_frame = ttk.LabelFrame(self.tab_edit, text="Analytical Exponential Tail Substitution", padding=8)
        tail_frame.pack(fill=tk.X, pady=6)

        self.enable_tail_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tail_frame, text="Substitute Analytical Decay Tail Above Anchor Point",
            variable=self.enable_tail_var, command=self.recalculate_all
        ).pack(anchor="w", pady=(0, 4))

        tail_grid = ttk.Frame(tail_frame)
        tail_grid.pack(fill=tk.X, pady=2)

        # Anchor Point (ms): [ down ] [ Entry ] [ up ]
        ttk.Label(tail_grid, text="Anchor Point (ms):", font=self.font_bold).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.tail_anchor_var = tk.StringVar(value="18.0")
        anchor_step = ttk.Frame(tail_grid)
        anchor_step.grid(row=0, column=1, sticky="w", pady=3)
        tk.Button(
            anchor_step, text="\u25bc", width=3, font=self.font_bold,
            command=lambda: self._step_tail_anchor(-0.5)
        ).pack(side=tk.LEFT)
        ent_anchor = tk.Entry(
            anchor_step, textvariable=self.tail_anchor_var, width=8, font=self.font_bold, justify="center"
        )
        ent_anchor.pack(side=tk.LEFT, padx=4)
        ent_anchor.bind("<Return>", lambda _: self._on_tail_entry_commit())
        ent_anchor.bind("<FocusOut>", lambda _: self._on_tail_entry_commit())
        tk.Button(
            anchor_step, text="\u25b2", width=3, font=self.font_bold,
            command=lambda: self._step_tail_anchor(+0.5)
        ).pack(side=tk.LEFT)

        # Decay Rate alpha (1/s): [ down ] [ Entry ] [ up ]
        ttk.Label(tail_grid, text="Decay Rate alpha (1/s):", font=self.font_bold).grid(row=0, column=2, sticky="w", padx=(18, 8), pady=3)
        self.tail_alpha_var = tk.StringVar(value="10.0")
        alpha_step = ttk.Frame(tail_grid)
        alpha_step.grid(row=0, column=3, sticky="w", pady=3)
        tk.Button(
            alpha_step, text="\u25bc", width=3, font=self.font_bold,
            command=lambda: self._step_tail_alpha(-1.0)
        ).pack(side=tk.LEFT)
        ent_alpha = tk.Entry(
            alpha_step, textvariable=self.tail_alpha_var, width=8, font=self.font_bold, justify="center"
        )
        ent_alpha.pack(side=tk.LEFT, padx=4)
        ent_alpha.bind("<Return>", lambda _: self._on_tail_entry_commit())
        ent_alpha.bind("<FocusOut>", lambda _: self._on_tail_entry_commit())
        tk.Button(
            alpha_step, text="\u25b2", width=3, font=self.font_bold,
            command=lambda: self._step_tail_alpha(+1.0)
        ).pack(side=tk.LEFT)

        # 4. Non-Destructive Edit Stack Listbox
        stack_frame = ttk.LabelFrame(self.tab_edit, text="Active Non-Destructive Edit Stack", padding=8)
        stack_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.enable_stack_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            stack_frame, text="Enable Surgical Edit Stack (Uncheck to A/B Compare)",
            variable=self.enable_stack_var, command=self.recalculate_all
        ).pack(anchor="w", pady=2)

        self.listbox_edits = tk.Listbox(stack_frame, height=5, font=self.font_mono)
        self.listbox_edits.pack(fill=tk.BOTH, expand=True, pady=6)

        stack_btns = ttk.Frame(stack_frame)
        stack_btns.pack(fill=tk.X)
        tk.Button(
            stack_btns, text="Remove Selected", command=self.remove_selected_edit,
            bg="#f0ad4e", fg="black", font=self.font_base, pady=3
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            stack_btns, text="Clear All Edits", command=self.clear_all_edits,
            bg="#595959", fg="white", font=self.font_base, pady=3
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # 5. Export Recovered FRD
        tk.Button(
            self.tab_edit, text="Export Reconstructed Response to FRD...", command=self.export_frd_file,
            bg="#2e7d32", fg="white", font=self.font_bold, pady=5
        ).pack(fill=tk.X, pady=(6, 0))

    # -------------------------------------------------------------------------
    # TAB 5: SPL CORRECTIONS (Active in Data Processing Mode Only)
    # -------------------------------------------------------------------------
    def _build_tab_spl_corrections(self):
        # 1. SPL Scaling Mode Selector (Manual Offset vs. Hardware Chain Calibration)
        mode_spl_frame = ttk.LabelFrame(
            self.tab_spl,
            text="SPL Scaling Mode (Data Processing Mode Only)",
            padding=8
        )
        mode_spl_frame.pack(fill=tk.X, pady=(0, 6))

        self.spl_mode_var = tk.StringVar(value="manual")
        rb_manual = ttk.Radiobutton(
            mode_spl_frame,
            text="Manual Offset (dB) - Native WAV Level + User Offset (Default 0.0 dB)",
            variable=self.spl_mode_var,
            value="manual",
            command=self._on_spl_param_change
        )
        rb_manual.pack(anchor="w", pady=2)

        rb_hw = ttk.Radiobutton(
            mode_spl_frame,
            text="Hardware Chain Calibration (V1, V2, Ag, Smic, d -> dB SPL @ 2.83V/1m)",
            variable=self.spl_mode_var,
            value="hardware",
            command=self._on_spl_param_change
        )
        rb_hw.pack(anchor="w", pady=2)

        # Manual Offset Entry Row
        man_row = ttk.Frame(mode_spl_frame)
        man_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(man_row, text="Manual SPL Offset (dB):", font=self.font_bold).pack(side=tk.LEFT, padx=(18, 8))
        self.spl_manual_offset_var = tk.DoubleVar(value=0.0)
        ent_manual = tk.Entry(
            man_row, textvariable=self.spl_manual_offset_var, width=10, font=self.font_bold, justify="right"
        )
        ent_manual.pack(side=tk.LEFT)
        ent_manual.bind("<Return>", lambda _: self._on_spl_param_change())
        ent_manual.bind("<FocusOut>", lambda _: self._on_spl_param_change())

        self.spl_manual_widgets.append(ent_manual)

        # 2. Five Hardware & Geometry Parameters (3-File Ratio Calibration)
        params_frame = ttk.LabelFrame(
            self.tab_spl,
            text="Hardware Chain Parameters (Set All 5 > 0 for dB SPL @ 2.83V/1m)",
            padding=8
        )
        params_frame.pack(fill=tk.X, pady=6)

        self.spl_v1_var = tk.DoubleVar(value=0.0)
        self.spl_v2_var = tk.DoubleVar(value=0.0)
        self.spl_ag_var = tk.DoubleVar(value=0.0)
        self.spl_smic_var = tk.DoubleVar(value=0.0)
        self.spl_dist_var = tk.DoubleVar(value=0.0)

        param_rows = [
            ("V1 (Ch1 Loopback Sensitivity, V @ 0 dBFS):", self.spl_v1_var),
            ("V2 (Ch2 Mic Input Sensitivity, V @ 0 dBFS):", self.spl_v2_var),
            ("Ag (Amp Voltage Gain / Divider Ratio, V/V):", self.spl_ag_var),
            ("Smic (Microphone Sensitivity, mV/Pa):", self.spl_smic_var),
            ("d (Speaker-to-Mic Distance, meters):", self.spl_dist_var),
        ]

        for idx, (label_txt, var) in enumerate(param_rows):
            ttk.Label(params_frame, text=label_txt).grid(row=idx, column=0, sticky="w", pady=3, padx=(0, 10))
            ent = tk.Entry(params_frame, textvariable=var, width=12, font=self.font_bold, justify="right")
            ent.grid(row=idx, column=1, sticky="e", pady=3)
            ent.bind("<Return>", lambda _: self._on_spl_param_change())
            ent.bind("<FocusOut>", lambda _: self._on_spl_param_change())
            self.spl_hardware_widgets.append(ent)

        # Action & Defaults Persistence Buttons
        btn_row_spl = ttk.Frame(params_frame)
        btn_row_spl.grid(row=len(param_rows), column=0, columnspan=2, sticky="ew", pady=(8, 0))

        btn_apply_spl = tk.Button(
            btn_row_spl, text="Apply SPL Settings", command=self._on_spl_param_change,
            bg="#1f4e79", fg="white", font=self.font_bold, pady=3
        )
        btn_apply_spl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_save_def = tk.Button(
            btn_row_spl, text="Save SPL Defaults", command=self.save_spl_defaults,
            bg="#595959", fg="white", font=self.font_base, pady=3
        )
        btn_save_def.pack(side=tk.LEFT, padx=(0, 6))

        btn_load_def = tk.Button(
            btn_row_spl, text="Load SPL Defaults", command=lambda: self._load_spl_defaults(silent=False),
            bg="#595959", fg="white", font=self.font_base, pady=3
        )
        btn_load_def.pack(side=tk.LEFT)

        # 3. Independent Microphone Calibration Curve Section
        cal_frame = ttk.LabelFrame(
            self.tab_spl,
            text="Microphone Calibration Curve (Independent Toggle - Subtracted from SPL)",
            padding=8
        )
        cal_frame.pack(fill=tk.X, pady=6)

        self.mic_cal_enable_var = tk.BooleanVar(value=False)
        cb_mic_cal = ttk.Checkbutton(
            cal_frame,
            text="Apply Microphone Calibration Curve",
            variable=self.mic_cal_enable_var,
            command=self._on_spl_param_change
        )
        cb_mic_cal.pack(anchor="w", pady=(0, 4))

        cal_btn_row = ttk.Frame(cal_frame)
        cal_btn_row.pack(fill=tk.X, pady=2)
        btn_load_cal = tk.Button(
            cal_btn_row, text="Load Mic Cal File (.mic / .cal)...", command=self.load_mic_cal_file,
            bg="#2e7d32", fg="white", font=self.font_bold, pady=3
        )
        btn_load_cal.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_clear_cal = tk.Button(
            cal_btn_row, text="Clear Cal File", command=self.clear_mic_cal_file,
            bg="#595959", fg="white", font=self.font_base, pady=3
        )
        btn_clear_cal.pack(side=tk.LEFT)

        self.lbl_mic_cal_file = ttk.Label(cal_frame, text="Calibration File: None loaded", font=self.font_italic)
        self.lbl_mic_cal_file.pack(anchor="w", pady=(4, 0))

        # 4. Calculated SPL Offset Readout Box
        readout_frame = ttk.LabelFrame(self.tab_spl, text="Calculated SPL Correction Summary", padding=8)
        readout_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.txt_spl_summary = tk.Text(readout_frame, height=5, font=self.font_mono, bg="#f8f9fa", state=tk.DISABLED)
        self.txt_spl_summary.pack(fill=tk.BOTH, expand=True)
        self._set_spl_summary_text(
            "Inactive in Modeling Mode.\n"
            "Switch to Data Processing Mode to configure SPL and microphone calibration."
        )

        self.wav_only_widgets.extend([
            rb_manual, rb_hw, btn_apply_spl, btn_save_def, btn_load_def,
            cb_mic_cal, btn_load_cal, btn_clear_cal
        ] + self.spl_manual_widgets + self.spl_hardware_widgets)

    # =========================================================================
    # MODE SWITCHING & WIDGET STATE MANAGEMENT
    # =========================================================================
    def _on_toggle_operating_mode(self):
        self._sync_mode_widget_states()
        self.recalculate_all()

    def _sync_mode_widget_states(self):
        is_modeling = self.modeling_mode_var.get()

        if is_modeling:
            self.mode_status_var.set("MODELING MODE (Synthetic Loudspeaker + Room Model)")
            self.entry_mode_status.configure(readonlybackground="#1f4e79", fg="white")
        else:
            if self.wav_ready:
                src_label = "Single Impulse WAV" if self.wav_input_type == "single_ir" else "3-File Farina WAV"
                self.mode_status_var.set(f"DATA PROCESSING MODE ({src_label} Active)")
            else:
                self.mode_status_var.set("DATA PROCESSING MODE (Awaiting WAV Import)")
            self.entry_mode_status.configure(readonlybackground="#2e7d32", fg="white")

        state_model = "normal" if is_modeling else "disabled"
        state_wav = "disabled" if is_modeling else "normal"

        for w in self.model_only_widgets:
            try:
                w.configure(state=state_model)
            except Exception:
                pass

        for w in self.wav_only_widgets:
            try:
                w.configure(state=state_wav)
            except Exception:
                pass

        # When xDAT metadata is present, lock the band edge spinboxes to the embedded xDAT values;
        # when xDAT is absent, enable manual adjustment in Data Processing Mode.
        if not is_modeling and self.wav_ready and self.wav_has_xdat:
            self.sp_band_low.configure(state="disabled")
            self.sp_band_high.configure(state="disabled")

        # Interlock Manual Offset vs. Hardware Chain entries in Data Processing Mode
        if not is_modeling:
            is_manual = (self.spl_mode_var.get() == "manual")
            for w in self.spl_manual_widgets:
                w.configure(state="normal" if is_manual else "disabled")
            for w in self.spl_hardware_widgets:
                w.configure(state="disabled" if is_manual else "normal")

        self._compute_spl_post_corrections()
        self._sync_ewcc_widget_states()

    def _set_xdat_text(self, text):
        self.txt_xdat.config(state=tk.NORMAL)
        self.txt_xdat.delete("1.0", tk.END)
        self.txt_xdat.insert(tk.END, text)
        self.txt_xdat.config(state=tk.DISABLED)

    def _set_spl_summary_text(self, text):
        self.txt_spl_summary.config(state=tk.NORMAL)
        self.txt_spl_summary.delete("1.0", tk.END)
        self.txt_spl_summary.insert(tk.END, text)
        self.txt_spl_summary.config(state=tk.DISABLED)

    def save_spl_defaults(self):
        """Saves current SPL hardware/manual parameters and mic cal path to spl_defaults.json."""
        def _sf(var):
            try:
                return float(var.get())
            except Exception:
                return 0.0

        payload = {
            "V1": _sf(self.spl_v1_var),
            "V2": _sf(self.spl_v2_var),
            "Ag": _sf(self.spl_ag_var),
            "Smic": _sf(self.spl_smic_var),
            "d": _sf(self.spl_dist_var),
            "manual_offset_db": _sf(self.spl_manual_offset_var),
            "mic_cal_path": self.mic_cal_path or "",
            "mic_cal_enabled": bool(self.mic_cal_enable_var.get()),
        }
        try:
            with open(self.spl_defaults_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo(
                "SPL Defaults Saved",
                f"SPL calibration defaults saved to:\n{self.spl_defaults_file}"
            )
        except Exception as e:
            messagebox.showerror("Save Defaults Error", f"Could not save SPL defaults:\n{e}")

    def _load_spl_defaults(self, silent=False):
        """Loads saved SPL parameters and mic calibration file from spl_defaults.json."""
        if not os.path.isfile(self.spl_defaults_file):
            if not silent:
                messagebox.showinfo(
                    "No Saved Defaults",
                    f"No saved SPL defaults file found at:\n{self.spl_defaults_file}"
                )
            return
        try:
            with open(self.spl_defaults_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.spl_v1_var.set(float(data.get("V1", 0.0)))
            self.spl_v2_var.set(float(data.get("V2", 0.0)))
            self.spl_ag_var.set(float(data.get("Ag", 0.0)))
            self.spl_smic_var.set(float(data.get("Smic", 0.0)))
            self.spl_dist_var.set(float(data.get("d", 0.0)))
            self.spl_manual_offset_var.set(float(data.get("manual_offset_db", 0.0)))
            self.mic_cal_enable_var.set(bool(data.get("mic_cal_enabled", False)))

            saved_cal_path = data.get("mic_cal_path", "")
            if saved_cal_path and os.path.isfile(saved_cal_path):
                handler = MicCalibrationHandler(saved_cal_path)
                if handler.is_valid:
                    self.mic_cal_path = saved_cal_path
                    self.mic_cal_handler = handler
                    fname = os.path.basename(saved_cal_path)
                    self.lbl_mic_cal_file.config(
                        text=f"Active Cal: {fname} ({handler.f_min:.1f} Hz - {handler.f_max:.0f} Hz, {handler.num_points} pts)"
                    )

            if not silent:
                self._on_spl_param_change()
                messagebox.showinfo("SPL Defaults Loaded", "Saved SPL calibration defaults have been loaded.")
        except Exception as e:
            if not silent:
                messagebox.showerror("Load Defaults Error", f"Could not read SPL defaults:\n{e}")

    def load_mic_cal_file(self):
        path = filedialog.askopenfilename(
            initialdir=self.last_dir,
            title="Select Microphone Calibration File",
            filetypes=[("Mic Calibration Files", "*.mic *.cal *.txt *.frd"), ("All Files", "*.*")]
        )
        if not path:
            return
        handler = MicCalibrationHandler(path)
        if not handler.is_valid:
            messagebox.showerror(
                "Mic Calibration Error",
                "Could not parse valid frequency/dB pairs from the selected calibration file."
            )
            return
        self.mic_cal_path = path
        self.mic_cal_handler = handler
        self.last_dir = os.path.dirname(path)
        fname = os.path.basename(path)
        self.lbl_mic_cal_file.config(
            text=f"Active Cal: {fname} ({handler.f_min:.1f} Hz - {handler.f_max:.0f} Hz, {handler.num_points} pts)"
        )
        self.mic_cal_enable_var.set(True)
        self._on_spl_param_change()

    def clear_mic_cal_file(self):
        self.mic_cal_path = None
        self.mic_cal_handler = None
        self.mic_cal_enable_var.set(False)
        self.lbl_mic_cal_file.config(text="Calibration File: None loaded")
        self._on_spl_param_change()

    def _on_spl_param_change(self):
        self._sync_mode_widget_states()
        if not self.modeling_mode_var.get() and self.wav_ready:
            self.recalculate_all()

    def _compute_spl_post_corrections(self):
        """
        Computes the frequency-independent SPL offset (dB) and updates the summary box:
          - Manual Offset Mode: Native WAV FFT magnitude + manual_offset_db (default 0.0 dB).
          - Hardware Chain Mode: 3-file ratio formula (V1, V2, Ag, Smic, d -> dB SPL @ 2.83V/1m).
        Microphone calibration curve is applied independently via self.mic_cal_enable_var.
        """
        def _safe_float(var):
            try:
                return float(var.get())
            except Exception:
                return 0.0

        is_modeling = self.modeling_mode_var.get()
        if is_modeling:
            self.spl_offset_db = 0.0
            self.spl_factor_val = None
            self.spl_absolute_active = False
            self.spl_mag_ylabel = "Magnitude (dB)"
            self._set_spl_summary_text(
                "Inactive in Modeling Mode.\n"
                "Switch to Data Processing Mode to configure SPL and microphone calibration."
            )
            return

        has_cal_active = (
            self.mic_cal_enable_var.get()
            and self.mic_cal_handler is not None
            and self.mic_cal_handler.is_valid
        )
        cal_status = os.path.basename(self.mic_cal_path) if has_cal_active else "OFF"
        spl_mode = self.spl_mode_var.get()

        if spl_mode == "manual":
            manual_db = _safe_float(self.spl_manual_offset_var)
            self.spl_offset_db = manual_db
            self.spl_factor_val = None
            self.spl_absolute_active = False
            self.spl_mag_ylabel = "SPL [dB, Native + Mic Cal]" if has_cal_active else "SPL [dB, Native]"
            self._set_spl_summary_text(
                f"Mode: Manual Offset (Native WAV Level, No Midband Auto-Normalization)\n"
                f"Manual Offset: {manual_db:+.2f} dB  |  Mic Cal Curve: {cal_status}"
            )
        else:
            v1 = _safe_float(self.spl_v1_var)
            v2 = _safe_float(self.spl_v2_var)
            ag = _safe_float(self.spl_ag_var)
            smic = _safe_float(self.spl_smic_var)
            dist = _safe_float(self.spl_dist_var)

            if all(v > 0.0 for v in (v1, v2, ag, smic, dist)):
                spl_factor = (dist / (0.001 * smic)) * (v2 / (v1 * ag)) * 2.83 * 50000.0
                offset_db = 20.0 * np.log10(spl_factor)
                self.spl_factor_val = spl_factor
                self.spl_offset_db = offset_db
                self.spl_absolute_active = True
                self.spl_mag_ylabel = "SPL [dB re 20 uPa @ 2.83V/1m]"
                self._set_spl_summary_text(
                    f"Mode: Hardware Chain Calibration (V1, V2, Ag, Smic, d)\n"
                    f"SPL_factor: {spl_factor:.4e}  |  Offset: {offset_db:+.2f} dB\n"
                    f"Mic Cal Curve: {cal_status}"
                )
            else:
                self.spl_factor_val = None
                self.spl_offset_db = 0.0
                self.spl_absolute_active = False
                self.spl_mag_ylabel = "Relative SPL [dB, Mic Cal]" if has_cal_active else "Relative SPL [dB]"
                self._set_spl_summary_text(
                    f"Mode: Hardware Chain Calibration (Incomplete: Enter all 5 parameters > 0)\n"
                    f"Fallback Offset: 0.00 dB (Native WAV Level)  |  Mic Cal Curve: {cal_status}"
                )

    def _get_post_spl_mag_db(self, P_complex):
        """
        Converts complex spectrum to dB magnitude and applies SPL offset + independent
        microphone calibration curve subtraction as post-processing in Data Processing Mode.
        """
        mag_db = 20.0 * np.log10(np.maximum(np.abs(P_complex), 1e-12))
        if not self.modeling_mode_var.get():
            mag_db = mag_db + self.spl_offset_db
            if (
                self.mic_cal_enable_var.get()
                and self.mic_cal_handler is not None
                and self.mic_cal_handler.is_valid
            ):
                cal_corr = self.mic_cal_handler.get_corrections(self.freqs)
                mag_db = mag_db - cal_corr
        return mag_db

    def _reset_loudspeaker_model(self):
        self.enclosure_type_var.set("VENTED")
        self.lf_fc_var.set(35.0)
        self.hf_fc_var.set(16000.0)
        self.hf_q_var.set(0.707)
        self.baffle_enable_var.set(True)
        self.baffle_fc_var.set(350.0)
        self.baffle_step_db_var.set(-4.0)

        defaults = [
            (True,   42.0,  +4.5,  8.0),
            (True,   85.0,  -9.0, 12.0),
            (True,  2500.0, -3.0,  6.0),
            (False, 500.0,  +2.0,  4.0),
            (False, 1200.0, -2.5,  5.0),
            (False, 6500.0, +3.0,  7.0),
        ]
        for band, (en, f0, gain, q_val) in zip(self.peq_bands, defaults):
            band["enabled"].set(en)
            band["freq"].set(f0)
            band["gain"].set(gain)
            band["q"].set(q_val)

        self.recalculate_all()

    # =========================================================================
    # WAV SELECTION (SINGLE IMPULSE IR & 3-FILE FARINA) & DECONVOLUTION ENGINE
    # =========================================================================
    def _apply_xdat_band_edges_if_present(self, meta):
        """
        Checks if xDAT sweep frequency bounds exist in metadata.
        If present, populates band_low_hz_var and band_high_hz_var automatically.
        If absent, keeps user-adjustable defaults (20 Hz and 20000 Hz).
        """
        has_xdat = ("f0sw" in meta) or ("f1sw" in meta) or ("fhfw" in meta)
        self.wav_has_xdat = has_xdat
        if has_xdat:
            f0 = float(meta.get("f0sw", 20.0))
            f1 = float(meta.get("f1sw", 20000.0))
            fh = float(meta.get("fhfw", f1))
            f_low = max(5.0, f0)
            f_high = fh if (fh > f_low and fh < self.fs * 0.49) else min(f1, self.fs * 0.45)
            self.band_low_hz_var.set(round(f_low, 2))
            self.band_high_hz_var.set(round(f_high, 1))
        else:
            # Default safe edges for external non-xDAT files (user can freely edit via spinboxes)
            self.band_low_hz_var.set(20.0)
            self.band_high_hz_var.set(min(20000.0, round(self.fs * 0.45, 0)))

    def load_single_impulse_wav(self):
        path = filedialog.askopenfilename(
            initialdir=self.last_dir,
            title="Select Single Pre-Processed Impulse Response WAV File",
            filetypes=[("WAV Files", "*.wav *.WAV"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            fs_ir, data_ir, meta_ir = read_wav_file_plus_info(path)
            self.wav_single_ir_path = path
            self.last_dir = os.path.dirname(path)
            self.lbl_single_ir_wav.config(text=os.path.basename(path))

            if fs_ir != self.fs:
                self.fs = int(fs_ir)
                self._update_dsp_grids()

            self.wav_metadata = meta_ir
            self.ir_meas_raw = data_ir
            self.ir_stim_raw = None
            self.wav_input_type = "single_ir"
            self.wav_ready = True

            self._apply_xdat_band_edges_if_present(meta_ir)

            # Default to Manual Offset SPL mode for Single Impulse WAVs (user can still switch to Hardware)
            self.spl_mode_var.set("manual")

            # Switch automatically into Data Processing Mode (uncheck Modeling Mode)
            self.modeling_mode_var.set(False)
            self._sync_mode_widget_states()
            self.recalculate_all()
        except Exception as e:
            messagebox.showerror("Single Impulse WAV Load Error", f"Failed to load WAV file:\n{e}")

    def select_meas_wav(self):
        path = filedialog.askopenfilename(
            initialdir=self.last_dir, title="Select Recorded Acoustic Measurement WAV File",
            filetypes=[("WAV Files", "*.wav *.WAV"), ("All Files", "*.*")]
        )
        if path:
            self.wav_meas_path = path
            self.last_dir = os.path.dirname(path)
            self.lbl_meas_wav.config(text=os.path.basename(path))

    def select_stim_wav(self):
        path = filedialog.askopenfilename(
            initialdir=self.last_dir, title="Select Measured Stimulus (Loopback/Amplifier) WAV File",
            filetypes=[("WAV Files", "*.wav *.WAV"), ("All Files", "*.*")]
        )
        if path:
            self.wav_stim_path = path
            self.last_dir = os.path.dirname(path)
            self.lbl_stim_wav.config(text=os.path.basename(path))

    def select_inv_wav(self):
        path = filedialog.askopenfilename(
            initialdir=self.last_dir, title="Select Farina Inverse Filter WAV File",
            filetypes=[("WAV Files", "*.wav *.WAV"), ("All Files", "*.*")]
        )
        if path:
            self.wav_inv_path = path
            self.last_dir = os.path.dirname(path)
            self.lbl_inv_wav.config(text=os.path.basename(path))

    def run_three_file_deconvolution(self):
        if not (self.wav_meas_path and self.wav_stim_path and self.wav_inv_path):
            messagebox.showwarning(
                "Missing WAV Files",
                "All 3 WAV files are required:\n"
                "1. Recorded Acoustic Measurement WAV\n"
                "2. Measured Stimulus (Loopback) WAV\n"
                "3. Farina Inverse Filter WAV"
            )
            return

        try:
            fs_m, data_meas, meta_m = read_wav_file_plus_info(self.wav_meas_path)
            fs_s, data_stim, meta_s = read_wav_file_plus_info(self.wav_stim_path)
            fs_i, data_inv, meta_i = read_wav_file_plus_info(self.wav_inv_path)

            if not (fs_m == fs_s == fs_i):
                raise ValueError(f"Sample rate mismatch: Meas={fs_m} Hz, Stim={fs_s} Hz, Inv={fs_i} Hz.")

            if fs_m != self.fs:
                self.fs = int(fs_m)
                self._update_dsp_grids()

            meta = {}
            meta.update(meta_i)
            meta.update(meta_s)
            meta.update(meta_m)
            self.wav_metadata = meta

            self.ir_meas_raw = fftconvolve(data_meas, data_inv, mode="full")
            self.ir_stim_raw = fftconvolve(data_stim, data_inv, mode="full")
            self.wav_input_type = "three_file"
            self.wav_ready = True

            self._apply_xdat_band_edges_if_present(meta)

            # Automatically select Hardware Chain SPL mode for 3-File Farina measurements
            self.spl_mode_var.set("hardware")

            # Switch automatically into Data Processing Mode (uncheck Modeling Mode)
            self.modeling_mode_var.set(False)
            self._sync_mode_widget_states()
            self.recalculate_all()

        except Exception as e:
            messagebox.showerror("3-File Deconvolution Error", f"Failed to process WAV files:\n{e}")

    def _on_wav_window_param_change(self):
        if not self.modeling_mode_var.get() and self.wav_ready:
            self.recalculate_all()

    @staticmethod
    def _detect_rising_edge_onset(ir, fs, rchr_s_per_dec, thresh_db=-45.0):
        """
        Finds the exact rising-edge sample index where the impulse lifts off the baseline,
        safely avoiding Farina negative-time harmonic distortion bursts.
        """
        peak_idx = int(np.argmax(np.abs(ir)))
        peak_val = float(np.abs(ir[peak_idx]))
        if peak_val <= 1e-15:
            return 0, ir

        if rchr_s_per_dec > 0.1:
            h2_offset_samples = int(fs * rchr_s_per_dec * np.log10(2.0))
            quiet_span = min(int(0.040 * fs), max(int(0.005 * fs), h2_offset_samples // 3))
        else:
            quiet_span = int(0.020 * fs)

        q_end = max(0, peak_idx - int(0.003 * fs))
        q_start = max(0, q_end - quiet_span)

        if q_end > q_start + 10:
            dc_bias = float(np.mean(ir[q_start:q_end]))
            ir_zeroed = ir - dc_bias
            noise_rms = float(np.sqrt(np.mean(ir_zeroed[q_start:q_end] ** 2)))
        else:
            ir_zeroed = ir - float(np.mean(ir[:max(1, peak_idx // 2)]))
            noise_rms = 0.0

        thresh_lin = max(5.0 * noise_rms, peak_val * (10.0 ** (thresh_db / 20.0)))

        search_limit = max(0, peak_idx - int(0.015 * fs))
        onset_idx = peak_idx
        for idx in range(peak_idx, search_limit, -1):
            if np.abs(ir_zeroed[idx]) < thresh_lin:
                onset_idx = idx
                break

        onset_idx = max(0, onset_idx - 2)
        return onset_idx, ir_zeroed

    def _build_wav_calibrated_spectrum(self):
        """
        Extracts causal windowed impulse response from ir_meas_raw (and divides out
        ir_stim_raw if in 3-file Farina mode), then applies smooth minimum-phase
        out-of-band tails guided by xDAT or manual Low/High Band Edge controls.
        """
        def _safe_float(var, default):
            try:
                return float(var.get())
            except Exception:
                return default

        def _safe_int(var, default):
            try:
                return int(var.get())
            except Exception:
                return default

        meta = self.wav_metadata
        rchr = float(meta.get("rchr", 0.0))
        fmt_desc = meta.get("_wav_fmt_desc", "Unknown Format")

        thresh_db = _safe_float(self.onset_thresh_db_var, -45.0)
        fine_shift = _safe_int(self.onset_shift_var, 0)
        win_ms = max(20.0, _safe_float(self.ir_win_ms_var, 500.0))
        win_len = min(self.n_fft, int((win_ms / 1000.0) * self.fs))

        # 1. Align Acoustic Measurement / Single Impulse IR at its causal rising edge
        onset_m, ir_m_clean = self._detect_rising_edge_onset(self.ir_meas_raw, self.fs, rchr, thresh_db)
        max_valid_onset = max(0, len(ir_m_clean) - 16)
        onset_m = max(0, min(max_valid_onset, onset_m + fine_shift))
        self.detected_onset_idx = onset_m

        # 2. Right-hand half-Tukey taper
        taper_len = max(8, int(0.15 * win_len))
        right_win = np.ones(win_len, dtype=np.float64)
        right_win[-taper_len:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_len)))
        right_win[0] = 0.0
        if win_len > 1:
            right_win[1] = 0.5

        ir_m_seg = np.zeros(self.n_fft, dtype=np.float64)
        avail_m = min(win_len, len(ir_m_clean) - onset_m)
        ir_m_seg[:avail_m] = ir_m_clean[onset_m : onset_m + avail_m] * right_win[:avail_m]
        H_meas = np.fft.rfft(ir_m_seg)

        # 3. If 3-file Farina mode, divide out stimulus reference; if Single IR mode, use H_meas directly
        if self.wav_input_type == "three_file" and self.ir_stim_raw is not None:
            onset_s, ir_s_clean = self._detect_rising_edge_onset(self.ir_stim_raw, self.fs, rchr, -50.0)
            stim_win_len = min(win_len, int(0.050 * self.fs), len(ir_s_clean) - onset_s)
            stim_win = np.ones(stim_win_len, dtype=np.float64)
            stim_taper = max(4, int(0.20 * stim_win_len))
            stim_win[-stim_taper:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, stim_taper)))
            ir_s_seg = np.zeros(self.n_fft, dtype=np.float64)
            ir_s_seg[:stim_win_len] = ir_s_clean[onset_s : onset_s + stim_win_len] * stim_win

            H_stim = np.fft.rfft(ir_s_seg)
            stim_peak2 = np.max(np.abs(H_stim) ** 2)
            reg_eps = 1e-6 * max(stim_peak2, 1e-24)
            H_cal = H_meas * np.conj(H_stim) / (np.abs(H_stim) ** 2 + reg_eps)
        else:
            H_cal = H_meas

        # 4. Determine active passband edges (from xDAT if present, else from user spinboxes)
        f_low = max(5.0, _safe_float(self.band_low_hz_var, 20.0))
        f_high = max(f_low + 50.0, min(_safe_float(self.band_high_hz_var, 20000.0), self.fs * 0.49))

        # 5. Compute SPL post-corrections (midband auto-normalization bypassed to preserve native WAV/calibrated levels)
        self._compute_spl_post_corrections()

        # 6. Out-of-Band Spectral Conditioning
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H_cal), 1e-12))

        if f_low > 1.0:
            idx_l = int(np.searchsorted(self.freqs, f_low))
            idx_l = max(2, min(len(self.freqs) - 2, idx_l))
            ref_low_db = float(np.median(mag_db[idx_l : min(len(mag_db), idx_l + 5)]))
            low_mask = (self.freqs > 0) & (self.freqs < f_low)
            oct_below = np.log2(np.maximum(self.freqs[low_mask], 0.1) / f_low)
            mag_db[low_mask] = ref_low_db + 24.0 * oct_below
            mag_db[0] = mag_db[1] - 6.0

        if f_high < self.freqs[-1]:
            idx_h = int(np.searchsorted(self.freqs, f_high))
            idx_h = max(2, min(len(self.freqs) - 2, idx_h))
            ref_high_db = float(np.median(mag_db[max(1, idx_h - 5) : idx_h + 1]))
            high_mask = self.freqs > f_high
            oct_above = np.log2(self.freqs[high_mask] / f_high)
            mag_db[high_mask] = ref_high_db - 24.0 * oct_above

        # Update metadata readout box
        if self.wav_has_xdat:
            f0sw = float(meta.get("f0sw", f_low))
            f1sw = float(meta.get("f1sw", f_high))
            fhfw = float(meta.get("fhfw", f_high))
            reps = int(meta.get("reps", 1))
            levl = float(meta.get("levl", 0.0))
            info_text = (
                f"Format: {fmt_desc} @ {self.fs} Hz | xDAT Chunk: PRESENT\n"
                f"f0sw: {f0sw:.2f} Hz | f1sw: {f1sw:.1f} Hz | fhfw: {fhfw:.1f} Hz | rchr: {rchr:.3f} s/dec\n"
                f"reps: {reps:<8d} | levl: {levl:.3e} | Rising-Edge Onset: sample {onset_m} ({onset_m/self.fs*1000:.2f} ms)"
            )
        else:
            info_text = (
                f"Format: {fmt_desc} @ {self.fs} Hz | xDAT Chunk: NOT PRESENT\n"
                f"Using Manual Band Edges: Low = {f_low:.1f} Hz, High = {f_high:.1f} Hz (Editable above)\n"
                f"Rising-Edge Onset: sample {onset_m} ({onset_m/self.fs*1000:.2f} ms) | Window: {win_ms:.0f} ms"
            )
        self._set_xdat_text(info_text)

        P_min_cond, real_ceps_cond, complex_ceps_cond, phase_min_exact = self._compute_minimum_phase_spectrum(mag_db)
        mag_cond_lin = 10.0 ** (mag_db / 20.0)
        excess_meas = np.angle(H_cal * np.exp(-1j * phase_min_exact))

        band_weight = np.ones_like(self.freqs)
        low_m = self.freqs < f_low
        high_m = self.freqs > f_high
        if np.any(low_m) and f_low > 0:
            band_weight[low_m] = 0.5 * (1.0 - np.cos(np.pi * self.freqs[low_m] / f_low))
        if np.any(high_m) and self.freqs[-1] > f_high:
            band_weight[high_m] = 0.5 * (1.0 + np.cos(np.pi * (self.freqs[high_m] - f_high) / (self.freqs[-1] - f_high)))

        P_wav_true = mag_cond_lin * np.exp(1j * (phase_min_exact + excess_meas * band_weight))
        return P_wav_true, real_ceps_cond, complex_ceps_cond, phase_min_exact

    # =========================================================================
    # LOUDSPEAKER & ROOM SYNTHETIC MODELING ENGINE
    # =========================================================================
    def _synthesize_loudspeaker_magnitude_db(self):
        """
        Builds the synthetic loudspeaker log-magnitude response from Tab 2 controls:
          - Sealed (2nd-order, Q=0.707) or Vented (4th-order, Q=0.707) High-Pass
          - 2nd-order High-Frequency Low-Pass (variable fc & Q)
          - 1st-order Baffle-Step Shelf (LF reduction below cutoff fc)
          - 6 Parametric EQ (PEQ) bands
        """
        f_safe = np.maximum(self.freqs, 0.25)

        # Safe float getter for Spinboxes in case user is mid-typing
        def _safe_float(var, default):
            try:
                return float(var.get())
            except Exception:
                return default

        # 1. Low-Frequency Enclosure Roll-Off (Butterworth Q = 0.707 -> (f/fc)^4 / (1 + (f/fc)^4))
        fc_lf = max(5.0, _safe_float(self.lf_fc_var, 35.0))
        ratio_lf4 = (f_safe / fc_lf) ** 4
        if self.enclosure_type_var.get() == "SEALED":
            # 2nd-order Butterworth HP (12 dB/oct, Q = 0.7071)
            mag_lf_db = 10.0 * np.log10(np.maximum(ratio_lf4 / (1.0 + ratio_lf4), 1e-24))
        else:
            # 4th-order Butterworth HP (24 dB/oct, Q = 0.7071)
            mag_lf_db = 20.0 * np.log10(np.maximum(ratio_lf4 / (1.0 + ratio_lf4), 1e-24))

        # 2. High-Frequency 2nd-Order Low-Pass Roll-Off (Variable fc & Q)
        fc_hf = max(200.0, _safe_float(self.hf_fc_var, 16000.0))
        q_hf = max(0.2, _safe_float(self.hf_q_var, 0.707))
        r_hf = f_safe / fc_hf
        denom_hf = (1.0 - r_hf**2)**2 + (r_hf / q_hf)**2
        mag_hf_db = -10.0 * np.log10(np.maximum(denom_hf, 1e-24))

        # 3. 1st-Order Baffle-Step Shelf Filter (LF step-down below cutoff fc_bs)
        mag_bs_db = np.zeros_like(self.freqs)
        if self.baffle_enable_var.get():
            fc_bs = max(10.0, _safe_float(self.baffle_fc_var, 350.0))
            step_db = min(0.0, _safe_float(self.baffle_step_db_var, -4.0))  # Only reduction <= 0 dB
            g_lin = 10.0 ** (step_db / 20.0)
            r_bs2 = (f_safe / fc_bs) ** 2
            # |H_bs(f)|^2 = (g^2 + (f/fc)^2) / (1 + (f/fc)^2) -> 0 dB at HF, step_db at LF
            mag_bs_db = 10.0 * np.log10(np.maximum((g_lin**2 + r_bs2) / (1.0 + r_bs2), 1e-24))

        # 4. 6 Parametric EQ (PEQ) Bands (Standard Audio Biquad Peaking Magnitude Response)
        mag_peq_db = np.zeros_like(self.freqs)
        for band in self.peq_bands:
            if band["enabled"].get():
                f0 = max(5.0, _safe_float(band["freq"], 100.0))
                gain_db = _safe_float(band["gain"], 0.0)
                q_val = max(0.2, _safe_float(band["q"], 1.0))
                if abs(gain_db) > 1e-6:
                    A = 10.0 ** (gain_db / 40.0)
                    w0 = 2.0 * np.pi * f0 / self.fs
                    w = 2.0 * np.pi * f_safe / self.fs
                    alpha = np.sin(w0) / (2.0 * q_val)

                    b0 = 1.0 + alpha * A
                    b1 = -2.0 * np.cos(w0)
                    b2 = 1.0 - alpha * A
                    a0 = 1.0 + alpha / A
                    a1 = -2.0 * np.cos(w0)
                    a2 = 1.0 - alpha / A

                    z1 = np.exp(-1j * w)
                    z2 = np.exp(-2j * w)
                    H_biquad = (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
                    mag_peq_db += 20.0 * np.log10(np.maximum(np.abs(H_biquad), 1e-12))

        total_mag_db = mag_lf_db + mag_hf_db + mag_bs_db + mag_peq_db
        total_mag_db[0] = total_mag_db[1] - 6.0
        return total_mag_db

    @staticmethod
    def _compute_minimum_phase_spectrum(mag_db):
        """
        Computes the exact complex minimum-phase spectrum from log-magnitude
        using the causal cepstral fold (discrete Hilbert transform).
        Returns (P_direct, real_ceps, complex_ceps_clean, phase_direct_exact).
        """
        mag_lin = 10.0 ** (mag_db / 20.0)
        log_mag_half = np.log(np.maximum(mag_lin, 1e-12))
        log_mag_full = np.concatenate([log_mag_half, log_mag_half[-2:0:-1]])

        real_ceps = np.fft.ifft(log_mag_full).real
        n = len(real_ceps)
        w = np.zeros(n)
        w[0] = 1.0
        w[1:n // 2] = 2.0
        w[n // 2] = 1.0

        complex_ceps_clean = real_ceps * w
        analytic_log = np.fft.fft(complex_ceps_clean)
        half_len = len(mag_db)
        phase_direct_exact = analytic_log[:half_len].imag
        return np.exp(analytic_log[:half_len]), real_ceps, complex_ceps_clean, phase_direct_exact

    def _compute_boundary_delays(self):
        """Calculates 1st-order image source delays (ms) and attenuation factors."""
        try:
            Lx, Ly, Lz = [float(x.strip()) for x in self.entry_room.get().split(",")]
            sx, sy, sz = [float(x.strip()) for x in self.entry_spk.get().split(",")]
            mx, my, mz = [float(x.strip()) for x in self.entry_mic.get().split(",")]
        except Exception:
            Lx, Ly, Lz = 30.0, 20.0, 8.0
            sx, sy, sz = 5.0, 10.0, 1.2
            mx, my, mz = 8.0, 10.0, 1.2

        c = 343.0
        r_direct = max(0.1, np.sqrt((mx - sx)**2 + (my - sy)**2 + (mz - sz)**2))

        image_coords = [
            (sx, sy, -sz),           # Floor
            (sx, sy, 2.0 * Lz - sz), # Ceiling
            (sx, -sy, sz),           # Left Wall
            (sx, 2.0 * Ly - sy, sz), # Right Wall
            (-sx, sy, sz),           # Back Wall
            (2.0 * Lx - sx, sy, sz)  # Front Wall
        ]

        boundaries = []
        for ctrl, (ix, iy, iz) in zip(self.surface_controls, image_coords):
            r_img = np.sqrt((mx - ix)**2 + (my - iy)**2 + (mz - iz)**2)
            delay_s = max(0.0, (r_img - r_direct) / c)
            dist_atten = r_direct / max(r_img, 0.1)
            boundaries.append({
                "name": ctrl["name"],
                "enabled": ctrl["enabled"].get(),
                "r0": ctrl["r0"].get(),
                "fc": ctrl["fc"].get(),
                "color": ctrl["color"],
                "delay_s": delay_s,
                "delay_ms": delay_s * 1000.0,
                "dist_atten": dist_atten
            })
        return boundaries

    # =========================================================================
    # EXPONENTIALLY WEIGHTED COMPLEX CEPSTRUM (EWCC) ENGINE & CONTROLS
    # =========================================================================
    def _get_effective_ir_window_samples(self):
        """
        Returns (win_len, T_win_eff_s) clamped to [100 ms, 500 ms] and <= n_fft,
        automatically scaling to any sample rate (44.1k, 48k, 96k, 192k, etc.).
        """
        try:
            win_ms = float(self.ir_win_ms_var.get())
        except Exception:
            win_ms = 500.0
        win_ms = max(100.0, min(500.0, win_ms))
        win_len = min(self.n_fft, max(64, int(round((win_ms / 1000.0) * self.fs))))
        t_win_eff = win_len / float(self.fs)
        return win_len, t_win_eff

    def _get_ewcc_beta(self):
        try:
            val = float(self.ewcc_beta_var.get())
        except Exception:
            val = 0.0
        return max(0.0, min(200.0, val))

    def _sync_ewcc_widget_states(self):
        is_complex = (self.ceps_mode_var.get() == "Complex")
        is_ewcc_on = is_complex and bool(self.ewcc_enable_var.get())

        self.cb_ewcc.configure(state="normal" if is_complex else "disabled")
        sub_state = "normal" if is_ewcc_on else "disabled"
        for w in (self.btn_beta_dn, self.ent_ewcc_beta, self.btn_beta_up, self.btn_ewcc_auto):
            w.configure(state=sub_state)

        self._update_ewcc_info_label()

    def _update_ewcc_info_label(self):
        is_complex = (self.ceps_mode_var.get() == "Complex")
        if not is_complex:
            self.lbl_ewcc_info.configure(
                text="EWCC inactive (Switch to Complex Cepstrum mode to enable)",
                fg="#666666"
            )
            return

        win_len, t_win = self._get_effective_ir_window_samples()
        if not self.ewcc_enable_var.get():
            self.lbl_ewcc_info.configure(
                text=f"EWCC off (Unweighted Complex Cepstrum) | Eff. Window: {t_win*1000.0:.0f} ms @ {self.fs} Hz",
                fg="#555555"
            )
            return

        beta = self._get_ewcc_beta()
        tail_gain_db = 8.685889638 * beta * t_win
        if tail_gain_db > 60.0:
            fg_col = "#c0392b"
            warn = " [HIGH NOISE RISK]"
        elif tail_gain_db > 45.0:
            fg_col = "#d35400"
            warn = " [Caution]"
        else:
            fg_col = "#1f4e79"
            warn = ""

        self.lbl_ewcc_info.configure(
            text=(
                f"Auto Min beta: {self.ewcc_auto_beta_min:.1f} 1/s | "
                f"Tail Gain @ {t_win*1000.0:.0f} ms: +{tail_gain_db:.1f} dB{warn}"
            ),
            fg=fg_col
        )

    def _on_ceps_mode_change(self):
        self._sync_ewcc_widget_states()
        self.recalculate_all()

    def _on_ewcc_toggle(self):
        if self.ewcc_enable_var.get() and self.P_tot is not None:
            # Automatically compute minimum beta when enabling EWCC
            beta_min = self._estimate_auto_min_beta(self.P_tot)
            self.ewcc_beta_var.set(f"{beta_min:.1f}")
        self._sync_ewcc_widget_states()
        self.recalculate_all()

    def _on_ewcc_auto_suggest(self):
        if self.P_tot is None:
            return
        beta_min = self._estimate_auto_min_beta(self.P_tot)
        self.ewcc_beta_var.set(f"{beta_min:.1f}")
        self._sync_ewcc_widget_states()
        self.recalculate_all()

    def _step_ewcc_beta(self, delta):
        val = max(0.0, min(200.0, self._get_ewcc_beta() + delta))
        self.ewcc_beta_var.set(f"{val:.1f}")
        self._update_ewcc_info_label()
        if self.ceps_mode_var.get() == "Complex" and self.ewcc_enable_var.get():
            self.recalculate_all()

    def _on_ewcc_beta_commit(self):
        self.ewcc_beta_var.set(f"{self._get_ewcc_beta():.1f}")
        self._update_ewcc_info_label()
        if self.ceps_mode_var.get() == "Complex" and self.ewcc_enable_var.get():
            self.recalculate_all()

    @staticmethod
    def _compute_min_phase_from_log_mag(log_mag_half):
        """Computes exact Hilbert minimum phase from half-spectrum natural log magnitude."""
        log_mag_full = np.concatenate([log_mag_half, log_mag_half[-2:0:-1]])
        real_ceps = np.fft.ifft(log_mag_full).real
        n = len(real_ceps)
        w = np.zeros(n)
        w[0] = 1.0
        w[1 : n // 2] = 2.0
        w[n // 2] = 1.0
        analytic_log = np.fft.fft(real_ceps * w)
        return analytic_log[:len(log_mag_half)].imag

    def _forward_complex_cepstrum(self, H_half):
        """
        Computes the complex cepstrum of complex half-spectrum H_half using
        continuous unwrapped excess phase with linear-phase detrending.
        Returns (ceps, phase_matched, lin_trend).
        """
        n_half = len(H_half)
        mag = np.maximum(np.abs(H_half), 1e-12)
        log_mag = np.log(mag)
        phase_min = self._compute_min_phase_from_log_mag(log_mag)

        excess_wrapped = np.angle(H_half * np.exp(-1j * phase_min))
        excess_unwrapped = np.unwrap(excess_wrapped)

        k_norm = np.linspace(0.0, 1.0, n_half)
        lin_trend = excess_unwrapped[0] + (excess_unwrapped[-1] - excess_unwrapped[0]) * k_norm
        phase_matched = phase_min + (excess_unwrapped - lin_trend)
        phase_matched[0] = 0.0
        phase_matched[-1] = 0.0

        log_complex_half = log_mag + 1j * phase_matched
        log_complex_full = np.concatenate([log_complex_half, np.conj(log_complex_half[-2:0:-1])])
        ceps = np.fft.ifft(log_complex_full).real
        return ceps, phase_matched, lin_trend

    def _apply_exp_weight_to_spectrum(self, P_half, beta, win_len):
        """Transforms P_half to time domain, applies exp(-beta * t) within win_len, returns weighted rfft."""
        h_raw = np.fft.irfft(P_half, n=self.n_fft)
        t_win = np.arange(win_len, dtype=np.float64) / float(self.fs)
        h_w = np.zeros(self.n_fft, dtype=np.float64)
        h_w[:win_len] = h_raw[:win_len] * np.exp(-beta * t_win)
        return np.fft.rfft(h_w)

    def _estimate_auto_min_beta(self, P_half):
        """
        Automatically finds the smallest beta (in 0.5 1/s steps) within a safe +50 dB
        tail-gain ceiling where the anti-causal complex cepstrum energy collapses.
        """
        win_len, t_win_eff = self._get_effective_ir_window_samples()
        beta_max = 50.0 / (8.685889638 * max(t_win_eff, 0.05))
        self.ewcc_beta_max_safe = beta_max

        h_raw = np.fft.irfft(P_half, n=self.n_fft)
        t_vec = np.arange(win_len, dtype=np.float64) / float(self.fs)

        raw_grid = np.linspace(0.0, beta_max, 16)
        cands = sorted(set(round(x * 2.0) / 2.0 for x in raw_grid))
        if not cands:
            cands = [0.0]

        ac_ratios = []
        half_n = self.n_fft // 2
        for b in cands:
            if b <= 0.0:
                H_w = P_half
            else:
                h_w = np.zeros(self.n_fft, dtype=np.float64)
                h_w[:win_len] = h_raw[:win_len] * np.exp(-b * t_vec)
                H_w = np.fft.rfft(h_w)
            ceps_b, _, _ = self._forward_complex_cepstrum(H_w)
            e_pos = float(np.sum(ceps_b[1:half_n] ** 2))
            e_neg = float(np.sum(ceps_b[half_n + 1 :] ** 2))
            ac_ratios.append(e_neg / max(e_pos + e_neg, 1e-30))

        # 1. If already nearly minimum-phase at beta=0, return 0.0
        if ac_ratios[0] < 0.01:
            chosen = 0.0
        else:
            # 2. Smallest beta where anti-causal energy < 1%
            below = [b for b, r in zip(cands, ac_ratios) if r < 0.01]
            if below:
                chosen = below[0]
            else:
                # 3. Otherwise pick the knee (smallest beta within 25% of minimum anti-causal ratio)
                min_r = min(ac_ratios)
                knee = [b for b, r in zip(cands, ac_ratios) if r <= min_r * 1.25 + 1e-6]
                chosen = knee[0] if knee else cands[int(np.argmin(ac_ratios))]

        self.ewcc_auto_beta_min = float(chosen)
        return float(chosen)

    def recalculate_all(self):
        is_modeling = self.modeling_mode_var.get()
        mode = self.ceps_mode_var.get()

        # If in Data Processing Mode and no WAV data has been imported yet, clear plots to blank
        if not is_modeling and not self.wav_ready:
            self.P_direct = None
            self.P_tot = None
            self.P_edited = None
            self.update_plots()
            return

        if is_modeling:
            # --- PATH A: MODELING MODE (Loudspeaker Model + Room Model) ---
            self.boundaries = self._compute_boundary_delays()
            mag_direct_db = self._synthesize_loudspeaker_magnitude_db()

            self.P_direct, real_ceps_clean, complex_ceps_clean, phase_direct_exact = (
                self._compute_minimum_phase_spectrum(mag_direct_db)
            )

            H_room = np.ones_like(self.P_direct, dtype=np.complex128)
            if self.enable_refl_var.get():
                for b in self.boundaries:
                    if b["enabled"] and b["r0"] > 0.0:
                        lp_filter = 1.0 / (1.0 + 1j * (self.freqs / b["fc"]))
                        R_f = b["r0"] * lp_filter
                        H_room += R_f * b["dist_atten"] * np.exp(-1j * 2.0 * np.pi * self.freqs * b["delay_s"])

            P_tot = self.P_direct * H_room

            if self.enable_noise_var.get():
                np.random.seed(101)
                noise_amp = 10.0 ** (self.noise_db_var.get() / 20.0)
                noise_spec = noise_amp * (
                    np.random.normal(0, 1, len(self.freqs)) + 1j * np.random.normal(0, 1, len(self.freqs))
                ) / np.sqrt(2.0)
                P_tot = P_tot + noise_spec

            self.P_tot = P_tot
        else:
            # --- PATH B: DATA PROCESSING MODE (3-File Farina WAV Measurement) ---
            self.boundaries = []
            P_wav, real_ceps_clean, complex_ceps_clean, phase_direct_exact = self._build_wav_calibrated_spectrum()
            self.P_direct = P_wav
            self.P_tot = P_wav

        # 4. Forward Transform to Cepstral Domain (Real vs. Complex / EWCC)
        mag_tot = np.maximum(np.abs(self.P_tot), 1e-12)
        log_mag_half = np.log(mag_tot)
        win_len, _t_win_eff = self._get_effective_ir_window_samples()
        ewcc_beta = self._get_ewcc_beta() if (mode == "Complex" and self.ewcc_enable_var.get()) else 0.0
        lin_trend_complex = None

        if mode == "Real":
            self.ceps_clean = real_ceps_clean
            log_spec_full = np.concatenate([log_mag_half, log_mag_half[-2:0:-1]])
            self.ceps_raw = np.fft.ifft(log_spec_full).real
            excess_for_plot = np.unwrap(np.angle(self.P_tot * np.exp(-1j * phase_direct_exact)))
            self.phase_tot_rad = phase_direct_exact + excess_for_plot
        else:
            # Complex Cepstrum (with proper phase unwrapping and optional EWCC exp(-beta*t) weighting)
            if ewcc_beta > 0.0:
                H_w = self._apply_exp_weight_to_spectrum(self.P_tot, ewcc_beta, win_len)
                self.ceps_raw, _phase_w, lin_trend_complex = self._forward_complex_cepstrum(H_w)
                if is_modeling:
                    H_clean_w = self._apply_exp_weight_to_spectrum(self.P_direct, ewcc_beta, win_len)
                    self.ceps_clean, _, _ = self._forward_complex_cepstrum(H_clean_w)
                else:
                    self.ceps_clean = complex_ceps_clean
            else:
                self.ceps_clean = complex_ceps_clean
                self.ceps_raw, _phase_w, lin_trend_complex = self._forward_complex_cepstrum(self.P_tot)

            excess_tot_unw = np.unwrap(np.angle(self.P_tot * np.exp(-1j * phase_direct_exact)))
            self.phase_tot_rad = phase_direct_exact + excess_tot_unw

        self.phase_direct_rad = phase_direct_exact

        # 5. Apply Non-Destructive Surgical Edits & Analytical Tail
        self.ceps_edited = np.copy(self.ceps_raw)

        mirror_neg = (mode == "Real")
        if self.enable_stack_var.get():
            for edit in self.edit_stack:
                etype = edit.get("type", "zero")
                if etype == "pchip":
                    self._apply_pchip_edit(
                        self.ceps_edited, edit["points"], edit.get("scale", "linear"), mirror_neg
                    )
                elif etype == "zero":
                    self._apply_zero_notch(self.ceps_edited, edit["start_ms"], edit["end_ms"], mirror_neg)
                    if edit.get("comb", False):
                        w_ms = (edit["end_ms"] - edit["start_ms"]) / 2.0
                        c_ms = (edit["start_ms"] + edit["end_ms"]) / 2.0
                        for mult in (2, 3):
                            self._apply_zero_notch(
                                self.ceps_edited, mult * c_ms - w_ms, mult * c_ms + w_ms, mirror_neg
                            )

        if self.enable_tail_var.get():
            self._apply_analytical_tail(
                self.ceps_edited, self._get_tail_anchor_ms(), self._get_tail_alpha(), mirror_neg
            )

        # 6. Inverse Transform Back to Frequency Domain (with Window-Clamped EWCC Un-Weighting)
        if mode == "Real":
            w = np.zeros(self.n_fft)
            w[0] = 1.0
            w[1:self.n_fft // 2] = 2.0
            w[self.n_fft // 2] = 1.0
            analytic_log = np.fft.fft(self.ceps_edited * w)
            self.P_edited = np.exp(analytic_log[:self.half_n])
            self.phase_edited_rad = analytic_log[:self.half_n].imag
        else:
            spec_full = np.fft.fft(self.ceps_edited)
            H_w_ed = np.exp(spec_full[:self.half_n])
            if lin_trend_complex is not None:
                H_w_ed = H_w_ed * np.exp(1j * lin_trend_complex)

            if ewcc_beta > 0.0:
                h_w_ed = np.fft.irfft(H_w_ed, n=self.n_fft)
                t_win = np.arange(win_len, dtype=np.float64) / float(self.fs)
                # Apply gentle 10% right-hand cosine taper during window-clamped un-weighting
                taper_len = max(8, int(0.10 * win_len))
                post_win = np.ones(win_len, dtype=np.float64)
                post_win[-taper_len:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_len)))
                h_ed = np.zeros(self.n_fft, dtype=np.float64)
                h_ed[:win_len] = h_w_ed[:win_len] * np.exp(ewcc_beta * t_win) * post_win
                self.P_edited = np.fft.rfft(h_ed)
            else:
                self.P_edited = H_w_ed

            excess_ed_unw = np.unwrap(np.angle(self.P_edited * np.exp(-1j * phase_direct_exact)))
            self.phase_edited_rad = phase_direct_exact + excess_ed_unw

        self._update_ewcc_info_label()
        self.update_plots()

    def _get_tail_anchor_ms(self):
        try:
            val = float(self.tail_anchor_var.get())
        except Exception:
            val = 18.0
        return max(0.5, min(300.0, val))

    def _get_tail_alpha(self):
        try:
            val = float(self.tail_alpha_var.get())
        except Exception:
            val = 10.0
        return max(0.0, min(500.0, val))

    def _step_tail_anchor(self, delta):
        val = max(0.5, min(300.0, self._get_tail_anchor_ms() + delta))
        self.tail_anchor_var.set(f"{val:.1f}")
        if self.enable_tail_var.get():
            self.recalculate_all()

    def _step_tail_alpha(self, delta):
        val = max(0.0, min(500.0, self._get_tail_alpha() + delta))
        self.tail_alpha_var.set(f"{val:.1f}")
        if self.enable_tail_var.get():
            self.recalculate_all()

    def _on_tail_entry_commit(self):
        self.tail_anchor_var.set(f"{self._get_tail_anchor_ms():.1f}")
        self.tail_alpha_var.set(f"{self._get_tail_alpha():.1f}")
        if self.enable_tail_var.get():
            self.recalculate_all()

    def _apply_zero_notch(self, ceps_arr, start_ms, end_ms, mirror_neg=True):
        idx_s = max(2, int(round((start_ms / 1000.0) * self.fs)))
        idx_e = min(self.n_fft // 2 - 2, int(round((end_ms / 1000.0) * self.fs)))
        if idx_e <= idx_s:
            return
        ceps_arr[idx_s:idx_e] = 0.0
        if mirror_neg:
            ceps_arr[self.n_fft - idx_e : self.n_fft - idx_s] = 0.0

    def _apply_pchip_edit(self, ceps_arr, points, scale="linear", mirror_neg=True):
        """
        Applies a user-defined multi-point PCHIP curve between the first and last
        selected quefrencies using the exact clicked Y-values.
          - scale == 'linear': Direct signed PCHIP interpolation in linear cepstral amplitude.
          - scale == 'log': Sign-preserving PCHIP envelope interpolation in dB.
        """
        if len(points) < 2:
            return

        pts_sorted = sorted(points, key=lambda p: p[0])
        xs_idx = []
        ys_val = []
        for x_ms, y_v in pts_sorted:
            idx = max(2, min(self.n_fft // 2 - 2, int(round((x_ms / 1000.0) * self.fs))))
            if xs_idx and idx == xs_idx[-1]:
                # Keep the latest clicked value if two clicks map to the exact same sample bin
                ys_val[-1] = float(y_v)
            else:
                xs_idx.append(idx)
                ys_val.append(float(y_v))

        if len(xs_idx) < 2:
            return

        idx_s, idx_e = xs_idx[0], xs_idx[-1]
        if idx_e <= idx_s:
            return

        eval_x = np.arange(idx_s, idx_e + 1, dtype=np.float64)
        pchip = PchipInterpolator(np.array(xs_idx, dtype=np.float64), np.array(ys_val, dtype=np.float64))
        curve = pchip(eval_x)

        pos_slice = slice(idx_s, idx_e + 1)
        neg_slice = slice(self.n_fft - idx_e, self.n_fft - idx_s + 1)

        if scale == "log":
            mag_lin = 10.0 ** (curve / 20.0)
            signs = np.where(ceps_arr[pos_slice] < 0.0, -1.0, 1.0)
            fill = signs * mag_lin
        else:
            fill = curve

        ceps_arr[pos_slice] = fill
        if mirror_neg:
            ceps_arr[neg_slice] = fill[::-1]

    def _apply_analytical_tail(self, ceps_arr, anchor_ms, alpha, mirror_neg=True):
        idx_a = max(2, min(self.n_fft // 2 - 2, int((anchor_ms / 1000.0) * self.fs)))
        half = self.n_fft // 2
        anchor_val = ceps_arr[idx_a]
        t_vec = np.arange(half - idx_a) / self.fs
        tail = anchor_val * np.exp(-alpha * t_vec)
        ceps_arr[idx_a:half] = tail
        if mirror_neg:
            ceps_arr[self.n_fft - half : self.n_fft - idx_a] = tail[::-1]

    # =========================================================================
    # INTERACTIVE MULTI-POINT PCHIP PICKING, WHEEL ZOOM & ARROW NUDGING
    # =========================================================================
    def _disarm_toolbar_tools(self):
        """Automatically turns off Matplotlib's Zoom or Pan tool if currently active."""
        try:
            mode_str = str(getattr(self.toolbar, "mode", "")).lower()
            if "pan" in mode_str:
                self.toolbar.pan()
            elif "zoom" in mode_str:
                self.toolbar.zoom()
        except Exception:
            pass

    def _find_nearest_picked_point(self, event, max_dist_px=10.0):
        """
        Returns the index in self.picked_points of the point closest to the mouse
        click in display (pixel) coordinates, if within max_dist_px; else None.
        """
        if not self.picked_points or event.x is None or event.y is None:
            return None
        best_idx = None
        best_dist = float(max_dist_px)
        for idx, (x_ms, y_val) in enumerate(self.picked_points):
            px, py = self.ax_ceps.transData.transform((x_ms, y_val))
            dist = float(np.hypot(event.x - px, event.y - py))
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _on_ceps_scroll(self, event):
        """Always-on mouse-wheel zoom on the lower cepstrum plot, centered on the cursor."""
        if self.P_direct is None or event.inaxes != self.ax_ceps:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Scroll up ("up" or step > 0) zooms in; scroll down zooms out
        is_zoom_in = (getattr(event, "button", "") == "up") or (getattr(event, "step", 0) > 0)
        scale_factor = 0.80 if is_zoom_in else 1.25

        cur_xlim = self.ax_ceps.get_xlim()
        cur_ylim = self.ax_ceps.get_ylim()
        xdata, ydata = float(event.xdata), float(event.ydata)

        new_x0 = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_x1 = xdata + (cur_xlim[1] - xdata) * scale_factor
        new_y0 = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_y1 = ydata + (cur_ylim[1] - ydata) * scale_factor

        # Clamp X bounds sensibly to [0, max_q]
        max_q = max(5.0, float(self.max_q_ms_var.get()))
        new_x0 = max(0.0, new_x0)
        new_x1 = min(max_q, max(new_x0 + 0.05, new_x1))

        self.ax_ceps.set_xlim(new_x0, new_x1)
        self.ax_ceps.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _on_nudge_key(self, event, dx_dir, dy_dir, fine=False):
        """Nudges the active control point with arrow keys (Shift = 5x finer step)."""
        if not self.picking_mode or self.active_point_idx is None:
            return
        if not (0 <= self.active_point_idx < len(self.picked_points)):
            return

        # Do not hijack arrow keys while typing inside an Entry or Spinbox
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Combobox)):
            return

        xlim = self.ax_ceps.get_xlim()
        ylim = self.ax_ceps.get_ylim()
        frac = 0.002 if fine else 0.01
        dx = dx_dir * (xlim[1] - xlim[0]) * frac
        dy = dy_dir * (ylim[1] - ylim[0]) * frac

        max_q = max(5.0, float(self.max_q_ms_var.get()))
        pt = self.picked_points[self.active_point_idx]
        pt[0] = max(0.05, min(max_q, pt[0] + dx))
        pt[1] = pt[1] + dy

        self._draw_live_pchip_preview()
        return "break"

    def _on_yscale_change(self, *_):
        if self.picking_mode:
            self._cancel_picking_mode()
        self.update_plots()

    def _clear_preview_artists(self):
        for art in self._preview_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._preview_artists.clear()

    def _draw_live_pchip_preview(self):
        """Updates preview control points, active highlight, and PCHIP curve without resetting zoom."""
        self._clear_preview_artists()
        if not self.picking_mode or not self.picked_points:
            self.canvas.draw_idle()
            return

        pts_sorted = sorted(self.picked_points, key=lambda p: p[0])
        xs = np.array([p[0] for p in pts_sorted], dtype=np.float64)
        ys = np.array([p[1] for p in pts_sorted], dtype=np.float64)

        pts_art, = self.ax_ceps.plot(
            xs, ys, linestyle="None", marker="o", markersize=7,
            markerfacecolor="#ffdd00", markeredgecolor="#d9534f", markeredgewidth=1.6, zorder=10
        )
        self._preview_artists.append(pts_art)

        # Highlight the active control point for arrow-key nudging
        if self.active_point_idx is not None and 0 <= self.active_point_idx < len(self.picked_points):
            ax_x, ax_y = self.picked_points[self.active_point_idx]
            active_art, = self.ax_ceps.plot(
                [ax_x], [ax_y], linestyle="None", marker="o", markersize=11,
                markerfacecolor="none", markeredgecolor="#00aa44", markeredgewidth=2.4, zorder=11
            )
            self._preview_artists.append(active_art)

        if len(xs) >= 2 and (xs[-1] > xs[0] + 1e-6):
            u_x, u_y = [], []
            for xv, yv in zip(xs, ys):
                if u_x and abs(xv - u_x[-1]) < 1e-6:
                    u_y[-1] = yv
                else:
                    u_x.append(xv)
                    u_y.append(yv)
            if len(u_x) >= 2:
                x_dense = np.linspace(u_x[0], u_x[-1], 300)
                y_dense = PchipInterpolator(np.array(u_x), np.array(u_y))(x_dense)
                line_art, = self.ax_ceps.plot(
                    x_dense, y_dense, color="#d9534f", linestyle="-", linewidth=2.0, zorder=9
                )
                self._preview_artists.append(line_art)

        self.canvas.draw_idle()

    def _update_picking_buttons(self):
        n_pts = len(self.picked_points)
        if not self.picking_mode:
            self.btn_begin_insert.configure(
                text="Begin Edit", bg="#1f4e79", fg="white", state="normal"
            )
            self.btn_zero_notch.configure(state="disabled")
            self.btn_cancel_edit.configure(state="disabled")
            self.lbl_pick_status.configure(
                text="Click 'Begin Edit' to place 2+ points on the lower plot (Scroll wheel zooms anytime)."
            )
        else:
            self.btn_cancel_edit.configure(state="normal")
            scale_lbl = "Log dB" if self.picking_scale == "log" else "Linear"
            if n_pts >= 2:
                self.btn_begin_insert.configure(
                    text=f"Insert PCHIP ({n_pts} pts)", bg="#2e7d32", fg="white", state="normal"
                )
                self.btn_zero_notch.configure(state="normal")
                self.lbl_pick_status.configure(
                    text=f"Picking ({scale_lbl}, {n_pts} pts): Click to add/select, Arrows (+Shift) to nudge, Right-click to delete."
                )
            else:
                self.btn_begin_insert.configure(
                    text=f"Insert PCHIP ({n_pts} pts)", bg="#888888", fg="white", state="disabled"
                )
                self.btn_zero_notch.configure(state="disabled")
                self.lbl_pick_status.configure(
                    text=f"Picking ({scale_lbl}): Left-click 2+ points (Scroll wheel zooms, Arrows nudge active point)."
                )

    def _on_begin_or_insert_pchip(self):
        if self.P_direct is None:
            messagebox.showwarning("No Data", "Load WAV data or enable Modeling Mode before editing.")
            return

        if not self.picking_mode:
            # Automatically disarm Zoom/Pan toolbar tool so user can place points immediately
            self._disarm_toolbar_tools()
            self.picking_mode = True
            self.picked_points = []
            self.active_point_idx = None
            self.picking_scale = "log" if self.y_scale_var.get().startswith("Log") else "linear"
            self._clear_preview_artists()
            self._update_picking_buttons()
            # Move focus to canvas so arrow keys nudge immediately
            self.canvas.get_tk_widget().focus_set()
            self.canvas.draw_idle()
        else:
            if len(self.picked_points) < 2:
                return
            pts_sorted = sorted(self.picked_points, key=lambda p: p[0])
            if pts_sorted[-1][0] <= pts_sorted[0][0]:
                messagebox.showwarning("Invalid Points", "Select at least two points at different quefrencies.")
                return
            edit_item = {
                "type": "pchip",
                "scale": self.picking_scale,
                "points": [(round(float(x), 3), float(y)) for x, y in pts_sorted],
                "start_ms": round(float(pts_sorted[0][0]), 2),
                "end_ms": round(float(pts_sorted[-1][0]), 2),
            }
            self.edit_stack.append(edit_item)
            self.picking_mode = False
            self.picked_points = []
            self.active_point_idx = None
            self._clear_preview_artists()
            self._update_picking_buttons()
            self._refresh_edit_listbox()
            self.recalculate_all()

    def _on_insert_zero_notch(self):
        if not self.picking_mode or len(self.picked_points) < 2:
            return
        xs = [p[0] for p in self.picked_points]
        s_ms, e_ms = round(float(min(xs)), 2), round(float(max(xs)), 2)
        if e_ms <= s_ms:
            messagebox.showwarning("Invalid Range", "Select two points at different quefrencies.")
            return
        edit_item = {
            "type": "zero",
            "start_ms": s_ms,
            "end_ms": e_ms,
            "comb": bool(self.comb_rahmonics_var.get()),
        }
        self.edit_stack.append(edit_item)
        self.picking_mode = False
        self.picked_points = []
        self.active_point_idx = None
        self._clear_preview_artists()
        self._update_picking_buttons()
        self._refresh_edit_listbox()
        self.recalculate_all()

    def _cancel_picking_mode(self):
        self.picking_mode = False
        self.picked_points = []
        self.active_point_idx = None
        self._clear_preview_artists()
        self._update_picking_buttons()
        self.canvas.draw_idle()

    def _on_ceps_click(self, event):
        """
        Handles mouse clicks on lower cepstrum plot during picking mode:
          - Left-click near an existing point (<= 9 px): selects it as active for arrow-key nudging.
          - Left-click elsewhere: adds a new control point and makes it active.
          - Right-click near an existing point (<= 12 px): deletes that specific point.
          - Right-click in empty space: deletes the most recently added point.
        """
        if not self.picking_mode or self.P_direct is None:
            return
        if self.toolbar.mode != "":
            return
        if event.inaxes != self.ax_ceps or event.xdata is None or event.ydata is None:
            return

        # Ensure canvas has focus so arrow keys immediately nudge the active point
        self.canvas.get_tk_widget().focus_set()

        if event.button == 1:  # Left-click: select existing point or add new point
            hit_idx = self._find_nearest_picked_point(event, max_dist_px=9.0)
            if hit_idx is not None:
                self.active_point_idx = hit_idx
                self._draw_live_pchip_preview()
            else:
                x_ms = max(0.05, float(event.xdata))
                y_val = float(event.ydata)
                self.picked_points.append([x_ms, y_val])
                self.active_point_idx = len(self.picked_points) - 1
                self._update_picking_buttons()
                self._draw_live_pchip_preview()
        elif event.button == 3:  # Right-click: targeted delete (or undo last point if empty space)
            if not self.picked_points:
                return
            hit_idx = self._find_nearest_picked_point(event, max_dist_px=12.0)
            del_idx = hit_idx if hit_idx is not None else (len(self.picked_points) - 1)
            self.picked_points.pop(del_idx)
            if not self.picked_points:
                self.active_point_idx = None
            elif self.active_point_idx is None or self.active_point_idx >= len(self.picked_points):
                self.active_point_idx = len(self.picked_points) - 1
            self._update_picking_buttons()
            self._draw_live_pchip_preview()

    def remove_selected_edit(self):
        sel = self.listbox_edits.curselection()
        if not sel:
            return
        del self.edit_stack[sel[0]]
        self._refresh_edit_listbox()
        self.recalculate_all()

    def clear_all_edits(self):
        self.edit_stack.clear()
        self._refresh_edit_listbox()
        self.recalculate_all()

    def _refresh_edit_listbox(self):
        self.listbox_edits.delete(0, tk.END)
        for i, ed in enumerate(self.edit_stack, start=1):
            if ed.get("type") == "pchip":
                n_pts = len(ed["points"])
                sc_tag = "Log" if ed.get("scale") == "log" else "Lin"
                self.listbox_edits.insert(
                    tk.END,
                    f"[{i}] PCHIP ({n_pts} pts, {sc_tag}): {ed['start_ms']:.2f} - {ed['end_ms']:.2f} ms"
                )
            else:
                comb_tag = " [+2x,3x]" if ed.get("comb", False) else ""
                self.listbox_edits.insert(
                    tk.END,
                    f"[{i}] Zero Notch: {ed['start_ms']:.2f} - {ed['end_ms']:.2f} ms{comb_tag}"
                )

    # =========================================================================
    # LOG-FREQUENCY BINNING (10% OVERLAPPING TUKEY WINDOW) & FREQ RANGE HELPERS
    # =========================================================================
    def _get_log_bin_ppo(self):
        try:
            val = int(round(float(self.log_bin_ppo_var.get())))
        except Exception:
            val = 24
        return max(12, min(192, val))

    def _get_freq_limits(self):
        """Returns sanitized (f_min, f_max) within [5 Hz, 1000 Hz] and [200 Hz, 0.48*fs], with f_min < f_max."""
        try:
            f_min = float(self.freq_min_hz_var.get())
        except Exception:
            f_min = 10.0
        try:
            f_max = float(self.freq_max_hz_var.get())
        except Exception:
            f_max = 20000.0

        nyq_cap = min(48000.0, 0.48 * float(self.fs))
        f_min = max(5.0, min(1000.0, f_min))
        f_max = max(200.0, min(nyq_cap, f_max))
        if f_min >= f_max - 10.0:
            f_min = max(5.0, f_max - 50.0)
        return f_min, f_max

    def _on_log_bin_param_change(self):
        ppo = self._get_log_bin_ppo()
        f_min, f_max = self._get_freq_limits()
        self.log_bin_ppo_var.set(ppo)
        self.freq_min_hz_var.set(round(f_min, 1))
        self.freq_max_hz_var.set(round(f_max, 1))
        self.sp_log_ppo.configure(state="normal" if self.log_bin_enable_var.get() else "disabled")
        self.update_plots()

    @staticmethod
    def _log_bin_tukey10_multi(freqs_lin, y_arrays, ppo, f_min, f_max):
        """
        Re-bins one or more linear-frequency arrays (linear power |H|^2 or unwrapped phase)
        onto a 1000 Hz-anchored log-frequency grid (ppo points/octave) spanning [f_min, f_max]
        using a 10% overlapping Tukey window in log2(f) (exact partition of unity).
        Returns (f_centers, [y_binned_0, y_binned_1, ...]).
        """
        ppo = max(12, min(192, int(round(ppo))))
        f_lo_lim = max(1.0, float(f_min))
        f_hi_lim = min(float(freqs_lin[-1]), float(f_max))
        if f_hi_lim <= f_lo_lim:
            fc = np.array([f_lo_lim], dtype=np.float64)
            return fc, [np.array([float(np.interp(f_lo_lim, freqs_lin, y))], dtype=np.float64) for y in y_arrays]

        k_min = int(np.ceil(ppo * np.log2(f_lo_lim / 1000.0) - 1e-12))
        k_max = int(np.floor(ppo * np.log2(f_hi_lim / 1000.0) + 1e-12))
        if k_max < k_min:
            f_centers = np.array([np.sqrt(f_lo_lim * f_hi_lim)], dtype=np.float64)
        else:
            ks = np.arange(k_min, k_max + 1, dtype=np.float64)
            f_centers = 1000.0 * (2.0 ** (ks / float(ppo)))

        valid = freqs_lin > 0.0
        f_pos = freqs_lin[valid]
        y_pos_list = [np.asarray(y, dtype=np.float64)[valid] for y in y_arrays]
        log2_f_pos = np.log2(f_pos)
        f_pos_min, f_pos_max = float(f_pos[0]), float(f_pos[-1])

        du = 1.0 / float(ppo)
        u_inner = 0.45 * du
        u_outer = 0.55 * du
        taper_w = 0.10 * du

        # 5-point sampling on each cosine ramp ensures accurate integration even when bin < df_FFT
        ramp_offsets = np.linspace(u_inner, u_outer, 5)
        template_u = np.concatenate([-ramp_offsets[::-1], [0.0], ramp_offsets])

        n_bins = len(f_centers)
        n_ch = len(y_pos_list)
        outs = [np.empty(n_bins, dtype=np.float64) for _ in range(n_ch)]
        log2_centers = np.log2(f_centers)

        idx_starts = np.searchsorted(f_pos, f_centers * (2.0 ** (-u_outer)), side="left")
        idx_ends = np.searchsorted(f_pos, f_centers * (2.0 ** (+u_outer)), side="right")

        for i in range(n_bins):
            lc = log2_centers[i]
            f_tmpl = np.clip(2.0 ** (lc + template_u), f_pos_min, f_pos_max)
            u_tmpl = np.log2(f_tmpl)

            i0, i1 = int(idx_starts[i]), int(idx_ends[i])
            if i1 > i0:
                u_int = log2_f_pos[i0:i1]
                u_all = np.concatenate([u_tmpl, u_int])
                order = np.argsort(u_all)
                u_all = u_all[order]
            else:
                u_all = u_tmpl
                order = None

            au = np.abs(u_all - lc)
            w = np.ones_like(au)
            in_taper = (au > u_inner) & (au < u_outer)
            w[in_taper] = 0.5 * (1.0 + np.cos(np.pi * (au[in_taper] - u_inner) / taper_w))
            w[au >= u_outer] = 0.0

            denom = float(_TRAPZ(w, u_all))
            for ch in range(n_ch):
                y_pos = y_pos_list[ch]
                if denom > 1e-18:
                    y_tmpl = np.interp(f_tmpl, f_pos, y_pos)
                    if order is not None:
                        y_all = np.concatenate([y_tmpl, y_pos[i0:i1]])[order]
                    else:
                        y_all = y_tmpl
                    outs[ch][i] = float(_TRAPZ(w * y_all, u_all)) / denom
                else:
                    outs[ch][i] = float(np.interp(f_centers[i], f_pos, y_pos))

        return f_centers, outs

    def update_plots(self):
        self.ax_freq.clear()
        self.ax_ceps.clear()

        f_min, f_max = self._get_freq_limits()

        # Blank state when in Data Processing Mode and no WAV measurement has been loaded yet
        if self.P_direct is None:
            self.ax_freq.set_xscale("log")
            self.ax_freq.set_xlim(f_min, f_max)
            self.ax_freq.set_ylim(-40.0, 10.0)
            self.ax_freq.set_title("Frequency Domain (Awaiting WAV Measurement Import)", fontsize=12, fontweight="bold")
            self.ax_freq.set_xlabel("Frequency (Hz)", fontsize=11)
            self.ax_freq.set_ylabel(self.spl_mag_ylabel, fontsize=11)
            self.ax_freq.grid(True, which="both", linestyle="--", alpha=0.5)
            self.ax_freq.text(
                0.5, 0.5,
                "DATA PROCESSING MODE\nLoad a Single Impulse WAV or 3-File Farina Measurement on Tab 1",
                transform=self.ax_freq.transAxes, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#555555"
            )

            max_q = max(5.0, float(self.max_q_ms_var.get()))
            self.ax_ceps.set_xlim(0.0, max_q)
            if self.y_scale_var.get().startswith("Log"):
                self.ax_ceps.set_ylim(-140.0, 20.0)
                self.ax_ceps.set_ylabel("Cepstral Magnitude (dB)", fontsize=11)
            else:
                self.ax_ceps.set_ylim(-0.05, 0.05)
                self.ax_ceps.set_ylabel("Cepstral Amplitude (Linear)", fontsize=11)
            self.ax_ceps.set_title("Quefrency Domain (Awaiting WAV Measurement Import)", fontsize=12, fontweight="bold")
            self.ax_ceps.set_xlabel("Quefrency (ms)", fontsize=11)
            self.ax_ceps.grid(True, linestyle="--", alpha=0.5)
            self.canvas.draw_idle()
            return

        is_modeling = self.modeling_mode_var.get()
        use_log_bin = bool(self.log_bin_enable_var.get())
        ppo = self._get_log_bin_ppo()
        bin_tag = f" [{ppo} PPO]" if use_log_bin else " [Raw FFT]"

        # --- UPPER PLOT: Frequency Response Comparison (Magnitude or Phase) ---
        upper_mode = self.upper_plot_mode_var.get()

        if upper_mode == "Magnitude":
            mag_dir_lin_db = self._get_post_spl_mag_db(self.P_direct)
            mag_tot_lin_db = self._get_post_spl_mag_db(self.P_tot)
            mag_ed_lin_db = self._get_post_spl_mag_db(self.P_edited)

            if use_log_bin:
                # Integrate linear power (10^(dB/10)) with 10% overlapping Tukey window
                pwr_dir = 10.0 ** (mag_dir_lin_db / 10.0)
                pwr_tot = 10.0 ** (mag_tot_lin_db / 10.0)
                pwr_ed = 10.0 ** (mag_ed_lin_db / 10.0)
                f_plot, [p_dir_b, p_tot_b, p_ed_b] = self._log_bin_tukey10_multi(
                    self.freqs, [pwr_dir, pwr_tot, pwr_ed], ppo, f_min, f_max
                )
                mag_dir_db = 10.0 * np.log10(np.maximum(p_dir_b, 1e-24))
                mag_tot_db = 10.0 * np.log10(np.maximum(p_tot_b, 1e-24))
                mag_ed_db = 10.0 * np.log10(np.maximum(p_ed_b, 1e-24))
            else:
                range_mask = (self.freqs >= f_min) & (self.freqs <= f_max)
                if not np.any(range_mask):
                    range_mask = self.freqs > 0.0
                f_plot = self.freqs[range_mask]
                mag_dir_db = mag_dir_lin_db[range_mask]
                mag_tot_db = mag_tot_lin_db[range_mask]
                mag_ed_db = mag_ed_lin_db[range_mask]

            eval_mask = (f_plot >= max(10.0, f_min)) & (f_plot <= min(1000.0, f_max))
            if not np.any(eval_mask):
                eval_mask = np.ones_like(f_plot, dtype=bool)

            if is_modeling:
                rms_err_raw = np.sqrt(np.mean((mag_tot_db[eval_mask] - mag_dir_db[eval_mask])**2))
                rms_err_ed = np.sqrt(np.mean((mag_ed_db[eval_mask] - mag_dir_db[eval_mask])**2))
                self.ax_freq.plot(
                    f_plot, mag_tot_db, color="#8faadc", alpha=0.55, linewidth=1.0,
                    label=f"Before Processing: Room-Contaminated (10-1kHz RMS Err: {rms_err_raw:.2f} dB)"
                )
                self.ax_freq.plot(
                    f_plot, mag_dir_db, color="#222222", linestyle="--", linewidth=1.5,
                    label="Ground-Truth Loudspeaker Model (Reflection-Free)"
                )
                self.ax_freq.plot(
                    f_plot, mag_ed_db, color="#d9534f", linewidth=1.6,
                    label=f"After Processing: Cepstral Edited (10-1kHz RMS Err: {rms_err_ed:.2f} dB)"
                )
            else:
                self.ax_freq.plot(
                    f_plot, mag_tot_db, color="#8faadc", alpha=0.65, linewidth=1.1,
                    label=("Before Processing: Single Impulse WAV" if self.wav_input_type == "single_ir" else "Before Processing: 3-File Deconvolved WAV")
                )
                self.ax_freq.plot(
                    f_plot, mag_ed_db, color="#d9534f", linewidth=1.7,
                    label="After Processing: Cepstral Edited Result"
                )

            vis_mag_mask = (f_plot >= max(20.0, f_min)) & (f_plot <= min(15000.0, f_max))
            y_peak = np.max(mag_tot_db[vis_mag_mask]) if np.any(vis_mag_mask) else np.max(mag_tot_db)
            y_max = np.ceil((y_peak + 5.0) / 5.0) * 5.0
            self.ax_freq.set_ylim(y_max - 50.0, y_max)
            self.ax_freq.set_title(
                f"Frequency Domain (Magnitude{bin_tag}): Before vs. After Cepstral Processing",
                fontsize=12, fontweight="bold"
            )
            self.ax_freq.set_ylabel(self.spl_mag_ylabel, fontsize=11)
            self.ax_freq.legend(loc="lower left", fontsize=10)
        else:
            if use_log_bin:
                # Integrate continuous unwrapped phase with 10% overlapping Tukey window
                f_plot, [ph_dir_rad, ph_tot_rad, ph_ed_rad] = self._log_bin_tukey10_multi(
                    self.freqs,
                    [self.phase_direct_rad, self.phase_tot_rad, self.phase_edited_rad],
                    ppo, f_min, f_max
                )
            else:
                range_mask = (self.freqs >= f_min) & (self.freqs <= f_max)
                if not np.any(range_mask):
                    range_mask = self.freqs > 0.0
                f_plot = self.freqs[range_mask]
                ph_dir_rad = self.phase_direct_rad[range_mask]
                ph_tot_rad = self.phase_tot_rad[range_mask]
                ph_ed_rad = self.phase_edited_rad[range_mask]

            eval_mask = (f_plot >= max(10.0, f_min)) & (f_plot <= min(1000.0, f_max))
            if not np.any(eval_mask):
                eval_mask = np.ones_like(f_plot, dtype=bool)

            raw_phase_diff_deg = np.degrees(np.angle(np.exp(1j * (ph_tot_rad - ph_dir_rad))))
            ed_phase_diff_deg = np.degrees(np.angle(np.exp(1j * (ph_ed_rad - ph_dir_rad))))
            rms_ph_raw = np.sqrt(np.mean(raw_phase_diff_deg[eval_mask]**2))
            rms_ph_ed = np.sqrt(np.mean(ed_phase_diff_deg[eval_mask]**2))

            if upper_mode == "Phase_Wrapped":
                ph_dir = np.degrees(np.angle(np.exp(1j * ph_dir_rad)))
                ph_tot = np.degrees(np.angle(np.exp(1j * ph_tot_rad)))
                ph_ed = np.degrees(np.angle(np.exp(1j * ph_ed_rad)))
                self.ax_freq.set_ylim(-190.0, 190.0)
                self.ax_freq.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                title_str = f"Frequency Domain (Wrapped Phase{bin_tag}): Before vs. After Cepstral Processing"
                ylabel_str = "Phase (Degrees, Wrapped)"
            else:
                ph_dir = np.degrees(ph_dir_rad)
                ph_tot = np.degrees(ph_tot_rad)
                ph_ed = np.degrees(ph_ed_rad)
                p_min = min(np.min(ph_dir), np.min(ph_tot), np.min(ph_ed)) - 20.0
                p_max = max(np.max(ph_dir), np.max(ph_tot), np.max(ph_ed)) + 20.0
                self.ax_freq.set_ylim(p_min, p_max)
                title_str = f"Frequency Domain (Continuous Unwrapped Phase{bin_tag}): Before vs. After Processing"
                ylabel_str = "Phase (Degrees, Continuous)"

            if is_modeling:
                self.ax_freq.plot(
                    f_plot, ph_tot, color="#8faadc", alpha=0.60, linewidth=1.0,
                    label=f"Before Processing: Room-Contaminated (10-1kHz RMS Phase Err: {rms_ph_raw:.2f} deg)"
                )
                self.ax_freq.plot(
                    f_plot, ph_dir, color="#222222", linestyle="--", linewidth=1.5,
                    label="Ground-Truth Loudspeaker Minimum Phase"
                )
                self.ax_freq.plot(
                    f_plot, ph_ed, color="#d9534f", linewidth=1.6,
                    label=f"After Processing: Cepstral Edited (10-1kHz RMS Phase Err: {rms_ph_ed:.2f} deg)"
                )
            else:
                self.ax_freq.plot(
                    f_plot, ph_tot, color="#8faadc", alpha=0.65, linewidth=1.1,
                    label=("Before Processing: Single Impulse WAV Phase" if self.wav_input_type == "single_ir" else "Before Processing: 3-File Deconvolved WAV Phase")
                )
                self.ax_freq.plot(
                    f_plot, ph_ed, color="#d9534f", linewidth=1.7,
                    label="After Processing: Cepstral Edited Phase"
                )

            self.ax_freq.set_title(title_str, fontsize=12, fontweight="bold")
            self.ax_freq.set_ylabel(ylabel_str, fontsize=11)
            self.ax_freq.legend(loc="lower left", fontsize=10)

        self.ax_freq.set_xscale("log")
        self.ax_freq.set_xlim(f_min, f_max)
        self.ax_freq.set_xlabel("Frequency (Hz)", fontsize=11)
        self.ax_freq.tick_params(axis="both", which="major", labelsize=10)
        self.ax_freq.grid(True, which="both", linestyle="--", alpha=0.5)

        # --- LOWER PLOT: Quefrency Domain (Cepstrum) ---
        max_q = max(5.0, float(self.max_q_ms_var.get()))
        q_mask = (self.quefrency_ms >= 0.0) & (self.quefrency_ms <= max_q)
        q_axis = self.quefrency_ms[q_mask]

        use_log_y = self.y_scale_var.get().startswith("Log")
        if use_log_y:
            ceps_ed_seg = self.ceps_edited[q_mask]
            y_raw = 20.0 * np.log10(np.maximum(np.abs(self.ceps_raw[q_mask]), 1e-9))
            y_ed_all = 20.0 * np.log10(np.maximum(np.abs(ceps_ed_seg), 1e-9))
            y_ed_pos = np.where(ceps_ed_seg >= 0.0, y_ed_all, np.nan)
            y_ed_neg = np.where(ceps_ed_seg < 0.0, y_ed_all, np.nan)
            y_clean = 20.0 * np.log10(np.maximum(np.abs(self.ceps_clean[q_mask]), 1e-9))
            ylabel = "Cepstral Magnitude (dB)"
        else:
            y_raw = self.ceps_raw[q_mask]
            y_ed = self.ceps_edited[q_mask]
            y_clean = self.ceps_clean[q_mask]
            ylabel = "Cepstral Amplitude (Linear)"

        self.ax_ceps.plot(q_axis, y_raw, color="#8faadc", alpha=0.55, linewidth=1.0, label="Unedited Cepstrum")
        if is_modeling and self.show_ghost_var.get():
            self.ax_ceps.plot(
                q_axis, y_clean, color="#2e7d32", linestyle=":", linewidth=1.4,
                label="Reflection-Free Baseline Cepstrum"
            )
        if use_log_y:
            self.ax_ceps.plot(q_axis, y_ed_pos, color="#1f4e79", linewidth=1.5, label="Edited (+ Lobe)")
            self.ax_ceps.plot(q_axis, y_ed_neg, color="#b83a14", linewidth=1.5, label="Edited (- Lobe)")
        else:
            self.ax_ceps.plot(q_axis, y_ed, color="#1f4e79", linewidth=1.4, label="Edited Cepstrum (Active)")

        # Only show 3D Room Reflection Guide Lines when in Modeling Mode
        if is_modeling and self.show_guides_var.get():
            for b in self.boundaries:
                if b["enabled"] and b["r0"] > 0.0 and 0.0 < b["delay_ms"] <= max_q:
                    self.ax_ceps.axvline(
                        b["delay_ms"], color=b["color"], linestyle="--", linewidth=1.4,
                        label=f"{b['name']} ({b['delay_ms']:.1f} ms)"
                    )
                    if self.show_harmonics_var.get():
                        for m in (2, 3):
                            if m * b["delay_ms"] <= max_q:
                                self.ax_ceps.axvline(
                                    m * b["delay_ms"], color=b["color"], linestyle=":", alpha=0.45, linewidth=1.0
                                )

        tail_anchor_ms = self._get_tail_anchor_ms()
        if self.enable_tail_var.get() and tail_anchor_ms <= max_q:
            self.ax_ceps.axvline(
                tail_anchor_ms, color="#d9534f", linestyle="-.", linewidth=1.6,
                label=f"Tail Anchor ({tail_anchor_ms:.1f} ms)"
            )

        self.ax_ceps.set_xlim(0.0, max_q)
        if use_log_y:
            self.ax_ceps.set_ylim(-140.0, 20.0)
        else:
            zoom_mask = q_axis >= 0.5
            max_v = np.max(np.abs(y_raw[zoom_mask])) if np.any(zoom_mask) else 0.05
            self.ax_ceps.set_ylim(-1.2 * max_v, 1.2 * max_v)

        ewcc_tag = (
            f" [EWCC beta={self._get_ewcc_beta():.1f} 1/s]"
            if (self.ceps_mode_var.get() == "Complex" and self.ewcc_enable_var.get() and self._get_ewcc_beta() > 0.0)
            else ""
        )
        ceps_title = (
            f"Quefrency Domain: Cepstrum & Boundary Guides{ewcc_tag} (Use 'Begin Edit' to Pick PCHIP Points)"
            if is_modeling else
            f"Quefrency Domain: Measured Cepstrum{ewcc_tag} (Use 'Begin Edit' to Pick PCHIP Points)"
        )
        self.ax_ceps.set_title(ceps_title, fontsize=12, fontweight="bold")
        self.ax_ceps.set_xlabel("Quefrency (ms)", fontsize=11)
        self.ax_ceps.set_ylabel(ylabel, fontsize=11)
        self.ax_ceps.tick_params(axis="both", which="major", labelsize=10)
        self.ax_ceps.grid(True, linestyle="--", alpha=0.5)
        self.ax_ceps.legend(loc="upper right", fontsize=9.5, ncol=2)

        self._preview_artists.clear()
        if self.picking_mode and self.picked_points:
            self._draw_live_pchip_preview()
        else:
            self.canvas.draw_idle()

    def export_frd_file(self):
        if self.P_edited is None:
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".frd", filetypes=[("FRD Files", "*.frd")])
        if not out_path:
            return
        try:
            f_min, f_max = self._get_freq_limits()
            mag_lin_db = self._get_post_spl_mag_db(self.P_edited)

            if self.log_bin_enable_var.get():
                ppo = self._get_log_bin_ppo()
                pwr_lin = 10.0 ** (mag_lin_db / 10.0)
                f_out, [pwr_b, ph_unw_b] = self._log_bin_tukey10_multi(
                    self.freqs, [pwr_lin, self.phase_edited_rad], ppo, f_min, f_max
                )
                mag_out_db = 10.0 * np.log10(np.maximum(pwr_b, 1e-24))
                phase_out_deg = np.degrees(np.angle(np.exp(1j * ph_unw_b)))
                hdr_note = f"* Log-Frequency Binned ({ppo} PPO, {f_min:.1f} - {f_max:.1f} Hz)\n"
            else:
                mask = (self.freqs >= f_min) & (self.freqs <= f_max)
                f_out = self.freqs[mask]
                mag_out_db = mag_lin_db[mask]
                phase_out_deg = np.degrees(np.angle(self.P_edited[mask]))
                hdr_note = f"* Raw Linear FFT Bins ({f_min:.1f} - {f_max:.1f} Hz, fs={self.fs} Hz, N={self.n_fft})\n"

            with open(out_path, "w") as f:
                f.write(hdr_note)
                f.write("* Frequency (Hz) Magnitude (dB) Phase (deg)\n")
                for freq, m, p in zip(f_out, mag_out_db, phase_out_deg):
                    f.write(f"{freq:<12.4f} {m:<12.4f} {p:<12.4f}\n")
            messagebox.showinfo(
                "Export Complete",
                f"Reconstructed FRD ({len(f_out)} pts) saved to:\n{out_path}"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save FRD:\n{e}")

    def on_closing(self):
        """Cleanly shuts down Matplotlib figures and terminates the Tkinter mainloop."""
        try:
            if hasattr(self, "mpl_click_cid") and self.mpl_click_cid is not None:
                self.canvas.mpl_disconnect(self.mpl_click_cid)
            if hasattr(self, "mpl_scroll_cid") and self.mpl_scroll_cid is not None:
                self.canvas.mpl_disconnect(self.mpl_scroll_cid)
            plt.close("all")
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = CepstralSimulationWorkbench(root)
    root.mainloop()
