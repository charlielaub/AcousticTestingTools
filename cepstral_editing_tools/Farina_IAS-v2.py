"""
Farina_IAS.py - Farina Impulse Alignment & Subtraction
Description: Performs 3-File System-Calibrated Farina Deconvolution with
    stimulus-referenced acoustic delay preservation, sub-sample time alignment
    for the Measure, Move-Move, Measure (MMMM) groundplane technique, and
    optional free-field subtraction using either a constant groundplane
    reflection coefficient (--r_value) or a frequency-dependent FRD file
    (--r_frd) with minimum-phase reconstruction (Equations A1.3 & A1.4).

Author: Charlie Laub
"""

import argparse
import os
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, correlate, fftconvolve, sosfreqz


# ---------------------------------------------------------------------------
# 1. WAV Header Peeking & xDAT Metadata Parsing
# ---------------------------------------------------------------------------

def read_wav_with_xdat(file_path):
    """
    Peeks at the RIFF 'fmt ' chunk (audio_format, bits_per_sample, and 0xFFFE
    WAVE_FORMAT_EXTENSIBLE sub-format) to decode IEEE float (32/64-bit) or
    PCM integer (8/16/24/32-bit) into mono float64, and extracts DaDooRunRun
    'xDAT' metadata tags (f0sw, f1sw, fhfw, rchr, levl, date, reps).

    Returns:
        fs (int): Sampling rate in Hz.
        data (np.ndarray): 1-D float64 audio array (channel 0 if multi-channel).
        xdat (dict): Parsed xDAT metadata tags.
        fmt_info (dict): Parsed WAV format metadata.
    """
    fmt_info = {}
    xdat = {}
    raw_data_bytes = None

    with open(file_path, "rb") as f:
        riff_id = f.read(4)
        if riff_id != b"RIFF":
            raise ValueError(f"'{file_path}' is not a valid RIFF file.")
        _riff_size = struct.unpack("<I", f.read(4))[0]
        wave_id = f.read(4)
        if wave_id != b"WAVE":
            raise ValueError(f"'{file_path}' is not a valid WAVE file.")

        while True:
            chunk_hdr = f.read(8)
            if len(chunk_hdr) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_hdr)

            if chunk_id == b"fmt ":
                chunk_payload = f.read(chunk_size)
                if len(chunk_payload) < 16:
                    raise ValueError(f"Malformed 'fmt ' chunk in '{file_path}'.")
                (
                    audio_format,
                    num_channels,
                    sample_rate,
                    byte_rate,
                    block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", chunk_payload[:16])

                effective_format = audio_format
                if audio_format == 0xFFFE and len(chunk_payload) >= 40:
                    # WAVE_FORMAT_EXTENSIBLE: SubFormat GUID starts at byte 24;
                    # first 2 bytes give the underlying format code (1=PCM, 3=IEEE Float)
                    effective_format = struct.unpack("<H", chunk_payload[24:26])[0]

                fmt_info = {
                    "audio_format": effective_format,
                    "raw_format": audio_format,
                    "num_channels": num_channels,
                    "sample_rate": sample_rate,
                    "byte_rate": byte_rate,
                    "block_align": block_align,
                    "bits_per_sample": bits_per_sample,
                }

            elif chunk_id == b"data":
                raw_data_bytes = f.read(chunk_size)

            elif chunk_id == b"xDAT":
                chunk_payload = f.read(chunk_size)
                # 8 bytes per entry: 4-byte ASCII tag + 4-byte float32 or uint32
                uint_tags = {"date", "reps"}
                for offset in range(0, len(chunk_payload) - 7, 8):
                    tag = chunk_payload[offset : offset + 4].decode("ascii", errors="ignore")
                    val_bytes = chunk_payload[offset + 4 : offset + 8]
                    if tag in uint_tags:
                        xdat[tag] = struct.unpack("<I", val_bytes)[0]
                    else:
                        xdat[tag] = struct.unpack("<f", val_bytes)[0]

            else:
                # Skip unknown chunks without loading them into memory
                f.seek(chunk_size, 1)

            # RIFF chunks are 2-byte word-aligned
            if chunk_size % 2 == 1:
                f.seek(1, 1)

    if not fmt_info or raw_data_bytes is None:
        raise ValueError(f"Missing 'fmt ' or 'data' chunk in '{file_path}'.")

    fs = fmt_info["sample_rate"]
    n_ch = fmt_info["num_channels"]
    bps = fmt_info["bits_per_sample"]
    afmt = fmt_info["audio_format"]

    # Decode raw PCM or IEEE Float bytes into float64
    if afmt == 3:  # IEEE Float
        if bps == 32:
            samples = np.frombuffer(raw_data_bytes, dtype="<f4").astype(np.float64)
        elif bps == 64:
            samples = np.frombuffer(raw_data_bytes, dtype="<f8").astype(np.float64)
        else:
            raise ValueError(f"Unsupported IEEE float bit depth: {bps} in '{file_path}'.")
    elif afmt == 1:  # PCM Integer
        if bps == 8:
            samples = (np.frombuffer(raw_data_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bps == 16:
            samples = np.frombuffer(raw_data_bytes, dtype="<i2").astype(np.float64) / 32768.0
        elif bps == 24:
            n_frames = len(raw_data_bytes) // 3
            raw_u8 = np.frombuffer(raw_data_bytes[: n_frames * 3], dtype=np.uint8).reshape(-1, 3)
            padded = np.zeros((n_frames, 4), dtype=np.uint8)
            padded[:, 1:4] = raw_u8
            samples = (padded.view("<i4").ravel() >> 8).astype(np.float64) / 8388608.0
        elif bps == 32:
            samples = np.frombuffer(raw_data_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bps} in '{file_path}'.")
    else:
        raise ValueError(f"Unsupported WAV audio format code: {hex(afmt)} in '{file_path}'.")

    # Convert multi-channel audio to mono by selecting channel 0
    if n_ch > 1:
        total_frames = len(samples) // n_ch
        samples = samples[: total_frames * n_ch].reshape(total_frames, n_ch)[:, 0]

    return fs, samples, xdat, fmt_info


# ---------------------------------------------------------------------------
# 2-6. 3-File System-Calibrated Farina Deconvolution Pipeline
# ---------------------------------------------------------------------------

def find_harmonic_safe_onset(ir_raw, fs, rchr):
    """
    Locates the main peak n_peak of an impulse response. Using rchr
    (seconds/decade), estimates the quiet pre-arrival baseline safely after
    the 2nd harmonic (delta_n_H2 = fs * rchr * log10(2)) and before main
    arrival, subtracts DC bias, and scans forward from the end of the quiet
    baseline region to n_peak to find the earliest sample exceeding
    max(5 * sigma_noise, A_peak * 10^(-45/20)), stepping back 2 samples as a
    causal guard. Forward scanning prevents premature stops on zero-crossings
    between driver arrivals or signal lobes.
    """
    n_peak = int(np.argmax(np.abs(ir_raw)))
    delta_n_h2 = fs * rchr * np.log10(2.0)

    # Quiet baseline region safely after H2 and before the main impulse arrival
    quiet_start = int(max(0, round(n_peak - 0.75 * delta_n_h2)))
    pre_arrival_guard = int(min(0.015 * fs, 0.15 * delta_n_h2))
    quiet_end = int(min(n_peak, max(quiet_start + 16, n_peak - pre_arrival_guard)))

    if quiet_end > quiet_start:
        quiet_seg = ir_raw[quiet_start:quiet_end]
        dc_bias = float(np.mean(quiet_seg))
        sigma_noise = float(np.std(quiet_seg))
    else:
        dc_bias = 0.0
        sigma_noise = 0.0

    ir_dc = ir_raw - dc_bias
    a_peak = float(np.abs(ir_dc[n_peak]))
    threshold = max(5.0 * sigma_noise, a_peak * (10.0 ** (-45.0 / 20.0)))

    # Scan forward from quiet_end to n_peak for the earliest threshold crossing
    above = np.flatnonzero(np.abs(ir_dc[quiet_end : n_peak + 1]) >= threshold)
    if len(above) > 0:
        first_above = quiet_end + int(above[0])
    else:
        first_above = n_peak

    # Step back 2 samples as a causal guard
    n_onset = max(0, first_above - 2)
    return n_onset, n_peak, ir_dc, dc_bias, sigma_noise


def apply_causal_half_tukey(ir_dc, n_onset, window_samples, taper_fraction=0.10):
    """
    Hard-slices the impulse response starting at n_onset (discarding negative-
    time Farina harmonic bursts), applies a 2-sample micro fade-in at indices
    0..1, and applies a right-hand-only raised-cosine taper over the tail.
    """
    seg = np.zeros(window_samples, dtype=np.float64)
    avail = min(window_samples, len(ir_dc) - n_onset)
    if avail > 0:
        seg[:avail] = ir_dc[n_onset : n_onset + avail]

    # 2-sample micro fade-in at index 0..1
    if window_samples >= 2:
        seg[0] *= 0.0
        seg[1] *= 0.5

    # Right-hand-only raised-cosine (half-Tukey) taper over the tail
    taper_len = max(2, int(round(window_samples * taper_fraction)))
    taper_len = min(taper_len, max(0, window_samples - 2))
    if taper_len >= 2:
        n = np.arange(taper_len, dtype=np.float64)
        fade_out = 0.5 * (1.0 + np.cos(np.pi * n / (taper_len - 1)))
        seg[window_samples - taper_len :] *= fade_out

    return seg


def apply_minimum_phase_band_conditioning(H_cal, fs, f_low, f_high):
    """
    Applies 24 dB/octave (4th-order Butterworth) minimum-phase roll-off
    conditioning outside [f_low, f_high].
    """
    n_bins = len(H_cal)
    n_fft = (n_bins - 1) * 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    nyquist = 0.5 * fs

    H_cond = H_cal.copy()

    # 4th-order Butterworth high-pass (24 dB/oct, minimum-phase)
    if f_low is not None and 0.0 < f_low < nyquist:
        sos_hp = butter(4, f_low, btype="highpass", fs=fs, output="sos")
        _, h_hp = sosfreqz(sos_hp, worN=freqs, fs=fs)
        H_cond *= h_hp

    # 4th-order Butterworth low-pass (24 dB/oct, minimum-phase)
    if f_high is not None and 0.0 < f_high < 0.999 * nyquist:
        sos_lp = butter(4, f_high, btype="lowpass", fs=fs, output="sos")
        _, h_lp = sosfreqz(sos_lp, worN=freqs, fs=fs)
        H_cond *= h_lp

    return H_cond


def deconvolve_calibrated_ir(
    meas_path,
    stim_path,
    inv_path,
    ir_length_ms=500.0,
    stim_window_ms=50.0,
    use_f1sw=False,
):
    """
    Executes the 3-File System-Calibrated Farina Deconvolution pipeline with
    stimulus-referenced acoustic time-of-flight preservation:
      1. WAV Header Peeking & xDAT Parsing
      2. 64-Bit Fast Linear Convolution (fftconvolve)
      3. Harmonic-Safe Rising-Edge Onset Alignment
      4. Causal Slicing, Delay Preservation & Right-Hand Half-Tukey Windowing
      5. Regularized Spectral Division (Tikhonov/Kirkeby)
      6. 24 dB/oct Minimum-Phase Out-of-Band Conditioning & IFFT
    """
    fs_m, data_meas, xdat_m, _ = read_wav_with_xdat(meas_path)
    fs_s, data_stim, xdat_s, _ = read_wav_with_xdat(stim_path)
    fs_i, data_inv, xdat_i, _ = read_wav_with_xdat(inv_path)

    if not (fs_m == fs_s == fs_i):
        raise ValueError(
            f"Sample rate mismatch: meas={fs_m} Hz, stim={fs_s} Hz, inv={fs_i} Hz."
        )
    fs = fs_m

    # Merge xDAT metadata (priority: measurement > stimulus > inverse filter)
    xdat = {**xdat_i, **xdat_s, **xdat_m}
    rchr = float(xdat.get("rchr", 2.0))
    f0sw = float(xdat.get("f0sw", 10.0))
    f1sw = float(xdat.get("f1sw", 0.99 * 0.5 * fs))
    fhfw = float(xdat.get("fhfw", 0.975 * 0.5 * fs))
    f_high = f1sw if use_f1sw else fhfw

    # 2. 64-Bit Fast Linear Convolution
    ir_meas_raw = fftconvolve(data_meas, data_inv, mode="full")
    ir_stim_raw = fftconvolve(data_stim, data_inv, mode="full")

    # 3. Harmonic-Safe Rising-Edge Onset Detection
    n_onset_m, n_peak_m, ir_meas_dc, _, _ = find_harmonic_safe_onset(ir_meas_raw, fs, rchr)
    n_onset_s, n_peak_s, ir_stim_dc, _, _ = find_harmonic_safe_onset(ir_stim_raw, fs, rchr)

    # Preserve exact integer-sample acoustic time-of-flight relative to stimulus
    delta_n_tof = n_onset_m - n_onset_s
    if delta_n_tof < 0:
        print(
            f"Warning: Detected measurement onset ({n_onset_m}) precedes stimulus onset "
            f"({n_onset_s}). Clamping acoustic offset to 0 samples."
        )
        delta_n_tof = 0

    # 4. Causal Slicing & Right-Hand Half-Tukey Windowing
    meas_win_samples = int(round((ir_length_ms / 1000.0) * fs))
    stim_win_samples = int(round((stim_window_ms / 1000.0) * fs))

    ir_m_windowed = apply_causal_half_tukey(
        ir_meas_dc, n_onset_m, meas_win_samples, taper_fraction=0.10
    )
    ir_s_seg = apply_causal_half_tukey(
        ir_stim_dc, n_onset_s, stim_win_samples, taper_fraction=0.20
    )

    # Place causally sliced measurement IR at index delta_n_tof so acoustic latency is preserved
    ir_m_len = delta_n_tof + meas_win_samples
    ir_m_seg = np.zeros(ir_m_len, dtype=np.float64)
    ir_m_seg[delta_n_tof:] = ir_m_windowed

    # Zero-pad both to N_FFT (next power of 2 >= 2 * max segment length to avoid wrap-around)
    min_fft = 2 * max(len(ir_m_seg), len(ir_s_seg))
    n_fft = 1 << (min_fft - 1).bit_length()

    # 5. Regularized Spectral Division (Tikhonov / Kirkeby)
    H_meas = np.fft.rfft(ir_m_seg, n=n_fft)
    H_stim = np.fft.rfft(ir_s_seg, n=n_fft)

    stim_mag2 = np.abs(H_stim) ** 2
    max_stim_mag2 = float(np.max(stim_mag2))
    if max_stim_mag2 == 0.0:
        raise ValueError(f"Stimulus spectrum has zero energy in '{stim_path}'.")
    eps = 1e-6 * max_stim_mag2
    H_cal = (H_meas * np.conj(H_stim)) / (stim_mag2 + eps)

    # 6. 24 dB/octave Minimum-Phase Out-of-Band Conditioning
    H_cal = apply_minimum_phase_band_conditioning(H_cal, fs, f0sw, f_high)

    # Convert back to calibrated time-domain impulse response
    ir_cal = np.fft.irfft(H_cal, n=n_fft)[:ir_m_len]

    info = {
        "fs": fs,
        "xdat": xdat,
        "n_onset_meas": n_onset_m,
        "n_onset_stim": n_onset_s,
        "n_peak_meas": n_peak_m,
        "n_peak_stim": n_peak_s,
        "delta_n_tof": delta_n_tof,
        "tof_ms": ((n_peak_m - n_peak_s) / fs) * 1000.0,
        "n_fft": n_fft,
        "band": (f0sw, f_high),
    }
    return fs, ir_cal, info


# ---------------------------------------------------------------------------
# Groundplane Reflection Transfer Function R(f) & Minimum-Phase Builder
# ---------------------------------------------------------------------------

def load_frd_magnitude(filepath):
    """
    Loads frequency (Hz) and magnitude (dB) from a standard FRD file
    (2-column or 3-column; phase column is ignored since minimum phase is
    reconstructed analytically). Ignores blank lines and comments starting
    with '*', '#', or ';'. Sorts and deduplicates frequencies.
    """
    freqs = []
    mags_db = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", "#", ";")):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                fv = float(parts[0])
                mv = float(parts[1])
            except ValueError:
                continue
            if np.isfinite(fv) and np.isfinite(mv) and fv >= 0.0:
                freqs.append(fv)
                mags_db.append(mv)

    if not freqs:
        raise ValueError(f"No valid frequency/magnitude data found in FRD file '{filepath}'.")

    freqs_arr = np.asarray(freqs, dtype=np.float64)
    mags_arr = np.asarray(mags_db, dtype=np.float64)

    # Sort by ascending frequency and average any duplicate frequency entries
    unique_freqs, inv_idx = np.unique(freqs_arr, return_inverse=True)
    if len(unique_freqs) < len(freqs_arr):
        avg_mags = np.zeros(len(unique_freqs), dtype=np.float64)
        counts = np.zeros(len(unique_freqs), dtype=np.float64)
        np.add.at(avg_mags, inv_idx, mags_arr)
        np.add.at(counts, inv_idx, 1.0)
        mags_arr = avg_mags / counts
        freqs_arr = unique_freqs
    else:
        order = np.argsort(freqs_arr)
        freqs_arr = freqs_arr[order]
        mags_arr = mags_arr[order]

    return freqs_arr, mags_arr


def _evaluate_r_mag_db_on_grid(
    target_freqs, pos_freqs, pos_mags_db, dc_mag_db, pchip_interp, hf_slope_db_per_dec, no_clamp
):
    """Evaluates the interpolated/extrapolated |R(f)| in dB on any frequency grid."""
    f_min = float(pos_freqs[0])
    f_max = float(pos_freqs[-1])
    mag_db_grid = np.zeros_like(target_freqs, dtype=np.float64)

    # 1. DC bin and low-frequency region (0 <= f < f_min)
    mag_db_grid[0] = dc_mag_db
    low_mask = (target_freqs > 0.0) & (target_freqs < f_min)
    if np.any(low_mask):
        mag_db_grid[low_mask] = np.interp(
            target_freqs[low_mask], [0.0, f_min], [dc_mag_db, float(pos_mags_db[0])]
        )

    # 2. Interior region (f_min <= f <= f_max) via shape-preserving PCHIP on log10(f)
    mid_mask = (target_freqs >= f_min) & (target_freqs <= f_max)
    if np.any(mid_mask):
        if pchip_interp is None:
            mag_db_grid[mid_mask] = float(pos_mags_db[0])
        else:
            mag_db_grid[mid_mask] = pchip_interp(np.log10(target_freqs[mid_mask]))

    # 3. High-frequency extrapolation (f > f_max)
    high_mask = target_freqs > f_max
    if np.any(high_mask):
        mag_db_grid[high_mask] = float(pos_mags_db[-1]) + hf_slope_db_per_dec * np.log10(
            target_freqs[high_mask] / f_max
        )

    if not no_clamp:
        mag_db_grid = np.minimum(mag_db_grid, 0.0)

    return mag_db_grid


def build_minimum_phase_r_from_frd(frd_path, fs, n_fft, no_clamp=False):
    """
    Constructs a causal, minimum-phase complex groundplane reflection transfer
    function R(f) on the rfft grid of length (n_fft // 2 + 1):
      - Anchors DC (0 Hz) at 0.0 dB (|R| = 1.0)
      - Interpolates interior [f_min, f_max] via shape-preserving PCHIP on log10(f)
      - Extrapolates above f_max using a linear log10(f) slope fitted above 10 kHz
        (or upper half-octave) clamped to [-40, 0] dB/decade (0 to 2nd-order LP)
      - Clamps magnitude to <= 0.0 dB (|R| <= 1.0) unless no_clamp=True
      - Reconstructs accurate continuous-like minimum phase via a 16x-oversampled
        real-cepstrum (discrete Hilbert) transform to eliminate Nyquist truncation
        phase pull-back, then maps phase onto the target rfft grid.
    """
    if n_fft % 2 != 0:
        raise ValueError("n_fft must be even for minimum-phase cepstral reconstruction.")

    frd_freqs, frd_mags_db = load_frd_magnitude(frd_path)

    # Separate DC point (if present in FRD) from positive frequencies
    has_dc = frd_freqs[0] == 0.0
    dc_mag_db = float(frd_mags_db[0]) if has_dc else 0.0
    pos_mask = frd_freqs > 0.0
    pos_freqs = frd_freqs[pos_mask]
    pos_mags_db = frd_mags_db[pos_mask]

    if len(pos_freqs) == 0:
        raise ValueError(f"FRD file '{frd_path}' contains no positive frequency points.")

    # Hard-clamp measured FRD magnitudes to <= 0.0 dB (|R| <= 1.0) unless disabled
    clamped_count = 0
    if not no_clamp:
        clamped_count = int(np.sum(pos_mags_db > 0.0))
        pos_mags_db = np.minimum(pos_mags_db, 0.0)
        dc_mag_db = min(dc_mag_db, 0.0)

    f_min = float(pos_freqs[0])
    f_max = float(pos_freqs[-1])
    pchip_interp = PchipInterpolator(np.log10(pos_freqs), pos_mags_db) if len(pos_freqs) > 1 else None

    # Fit high-frequency extrapolation slope in dB/decade (clamped to [-40, 0] dB/decade)
    hf_slope_db_per_dec = 0.0
    fit_mask = pos_freqs >= 10000.0
    if np.sum(fit_mask) < 2:
        fit_mask = pos_freqs >= (f_max / np.sqrt(2.0))
    if np.sum(fit_mask) >= 2:
        fit_f = pos_freqs[fit_mask]
        fit_m = pos_mags_db[fit_mask]
    elif len(pos_freqs) >= 2:
        n_pts = min(5, len(pos_freqs))
        fit_f = pos_freqs[-n_pts:]
        fit_m = pos_mags_db[-n_pts:]
    else:
        fit_f = None
        fit_m = None

    if fit_f is not None and len(fit_f) >= 2:
        slope, _ = np.polyfit(np.log10(fit_f), fit_m, 1)
        hf_slope_db_per_dec = float(np.clip(slope, -40.0, 0.0))

    # Evaluate exact magnitude on the target FFT grid [0, fs/2]
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag_db_grid = _evaluate_r_mag_db_on_grid(
        fft_freqs, pos_freqs, pos_mags_db, dc_mag_db, pchip_interp, hf_slope_db_per_dec, no_clamp
    )
    mag_lin = np.maximum(10.0 ** (mag_db_grid / 20.0), 1e-12)

    # 16x-oversampled cepstral Hilbert transform (fixed 262144-pt FFT) so the
    # high-frequency roll-off extends 4 octaves past Nyquist without phase pull-back
    oversample = 16
    n_hilb = 262144
    fs_hilb = fs * oversample
    hilb_freqs = np.fft.rfftfreq(n_hilb, d=1.0 / fs_hilb)
    hilb_mag_db = _evaluate_r_mag_db_on_grid(
        hilb_freqs, pos_freqs, pos_mags_db, dc_mag_db, pchip_interp, hf_slope_db_per_dec, no_clamp
    )
    hilb_log_half = np.log(np.maximum(10.0 ** (hilb_mag_db / 20.0), 1e-12))
    hilb_log_full = np.concatenate([hilb_log_half, hilb_log_half[-2:0:-1]])

    cepstrum = np.fft.ifft(hilb_log_full).real
    cep_min = np.zeros_like(cepstrum)
    cep_min[0] = cepstrum[0]
    cep_min[1 : n_hilb // 2] = 2.0 * cepstrum[1 : n_hilb // 2]
    cep_min[n_hilb // 2] = cepstrum[n_hilb // 2]

    hilb_phase = np.fft.fft(cep_min)[: n_hilb // 2 + 1].imag
    phase_on_grid = np.interp(fft_freqs, hilb_freqs, hilb_phase)

    R_f = mag_lin * np.exp(1j * phase_on_grid)
    # Enforce purely real DC and Nyquist bins for real-valued irfft output
    # while preserving the Nyquist sign (e.g., negative real near -180 deg)
    R_f[0] = mag_lin[0]
    R_f[-1] = float(np.real(R_f[-1]))

    r_info = {
        "f_min": f_min,
        "f_max": f_max,
        "hf_slope_db_per_oct": hf_slope_db_per_dec * np.log10(2.0),
        "clamped_points": clamped_count,
        "min_mag": float(np.min(mag_lin)),
        "max_mag": float(np.max(mag_lin)),
    }
    return R_f, r_info


def compute_reflection_weight(R_f):
    """
    Computes the MMMM reflection weighting transfer function W(f) = R(f) / (1 + R(f))
    (Equation A1.3), with a stability guard if 1 + R(f) approaches zero.
    """
    denom = 1.0 + R_f
    abs_denom = np.abs(denom)
    min_denom = float(np.min(abs_denom))

    if min_denom < 0.1:
        print(
            f"Warning: min|1 + R(f)| = {min_denom:.4f} is very small. "
            "Check the groundplane reflection data for non-physical values."
        )

    if min_denom < 1e-3:
        # Regularized division to prevent numerical blow-up
        W_f = (R_f * np.conj(denom)) / (abs_denom ** 2 + 1e-6)
    else:
        W_f = R_f / denom

    return W_f


# ---------------------------------------------------------------------------
# Sub-Sample Time Alignment (MMMM) & Visualization
# ---------------------------------------------------------------------------

def calculate_subsample_offset(data_master, data_slave, fs, max_window_us=500.0):
    """
    Computes cross-correlation restricted to a specific time window,
    handles polarity inversions, protects boundaries, and applies
    Gaussian (log-quadratic) interpolation (with parabolic fallback) for
    unbiased sub-sample alignment.
    """
    max_lag_samples = int(np.ceil(max_window_us * 1e-6 * fs))
    corr = correlate(data_master, data_slave, mode="full")
    lags = np.arange(-len(data_slave) + 1, len(data_master))

    valid_window_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_window = corr[valid_window_mask]
    lags_window = lags[valid_window_mask]

    peak_idx_in_window = int(np.argmax(np.abs(corr_window)))
    coarse_lag = int(lags_window[peak_idx_in_window])
    global_peak_idx = int(np.where(lags == coarse_lag)[0][0])

    if global_peak_idx <= 0 or global_peak_idx >= len(corr) - 1:
        print("Warning: Correlation peak hit array boundary. Skipping sub-sample interpolation.")
        return float(coarse_lag), coarse_lag, 0.0, max_lag_samples, lags, corr, global_peak_idx

    y1 = corr[global_peak_idx - 1]
    y2 = corr[global_peak_idx]
    y3 = corr[global_peak_idx + 1]

    # Robustness against polarity inversion (log interpolation requires positive values)
    if y2 < 0:
        y1, y2, y3 = -y1, -y2, -y3

    if y1 > 0 and y2 > 0 and y3 > 0:
        # Gaussian (log-quadratic) interpolation for unbiased sub-sample peak estimation
        ln_y1, ln_y2, ln_y3 = np.log(y1), np.log(y2), np.log(y3)
        denominator = ln_y1 - (2.0 * ln_y2) + ln_y3
        fine_offset = 0.0 if denominator == 0.0 else 0.5 * (ln_y1 - ln_y3) / denominator
    else:
        # Fallback to standard parabolic interpolation if narrow peak neighbors cross zero
        print("Note: Non-positive correlation neighbors near peak. Using parabolic interpolation fallback.")
        denominator = y1 - (2.0 * y2) + y3
        fine_offset = 0.0 if denominator == 0.0 else 0.5 * (y1 - y3) / denominator

    fine_offset = float(np.clip(fine_offset, -0.5, 0.5))

    exact_lag_samples = coarse_lag + fine_offset
    return exact_lag_samples, coarse_lag, fine_offset, max_lag_samples, lags, corr, global_peak_idx


def plot_parabolic_fit(lags, corr, global_peak_idx, coarse_lag, fine_offset):
    """Visualizes the cross-correlation peak and sub-sample fit."""
    if global_peak_idx <= 0 or global_peak_idx >= len(corr) - 1:
        print("Skipping plot: peak is at array boundary.")
        return

    idx_points = [global_peak_idx - 1, global_peak_idx, global_peak_idx + 1]
    x_points = lags[idx_points]
    y_points = corr[idx_points]

    y1, y2, y3 = y_points
    a = 0.5 * (y1 - 2.0 * y2 + y3)
    b = 0.5 * (y3 - y1)
    c = y2

    x_fit_rel = np.linspace(-1.5, 1.5, 200)
    y_fit = a * (x_fit_rel ** 2) + b * x_fit_rel + c
    x_fit_abs = x_fit_rel + coarse_lag

    exact_lag = coarse_lag + fine_offset
    peak_value = a * (fine_offset ** 2) + b * fine_offset + c

    plt.rcParams["savefig.dpi"] = 600
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(x_fit_abs, y_fit, "g--", label="Quadratic Fit (Parabola)")
    plt.scatter(x_points, y_points, color="red", zorder=5, label="Discrete Sample Lags")
    plt.axvline(x=coarse_lag, color="gray", linestyle=":", label=f"Coarse Peak ({coarse_lag})")
    plt.scatter(
        [exact_lag],
        [peak_value],
        color="blue",
        marker="*",
        s=150,
        zorder=6,
        label=f"Sub-sample Peak ({exact_lag:.4f})",
    )

    plt.title("Cross-Correlation Peak & Sub-Sample Interpolation")
    plt.xlabel("Lag (Samples)")
    plt.ylabel("Correlation Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.15, right=0.93, top=0.9, bottom=0.21)
    plt.figtext(
        0.5,
        0.05,
        "To close the window, click on the kill button |X| located in the top right corner.",
        ha="center",
        fontsize=12,
        color="blue",
    )
    plt.show()


# ---------------------------------------------------------------------------
# File Resolution & Output Helpers
# ---------------------------------------------------------------------------

def resolve_input_sets(base1, base2):
    """
    Constructs the expected DaDooRunRun filenames (<base>_m.wav, <base>_s.wav,
    <base>_Fif.wav) for Measurement #1 and Measurement #2, with automatic
    fallback if one of the two Farina inverse filter files is missing.
    """
    meas1 = f"{base1}_m.wav"
    stim1 = f"{base1}_s.wav"
    inv1 = f"{base1}_Fif.wav"

    meas2 = f"{base2}_m.wav"
    stim2 = f"{base2}_s.wav"
    inv2 = f"{base2}_Fif.wav"

    for path in (meas1, stim1, meas2, stim2):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input file not found: '{path}'")

    inv1_exists = os.path.isfile(inv1)
    inv2_exists = os.path.isfile(inv2)

    if not inv1_exists and not inv2_exists:
        raise FileNotFoundError(
            f"Farina inverse filter not found for either set: '{inv1}' or '{inv2}'."
        )
    elif not inv2_exists:
        print(f"Note: '{inv2}' not found. Falling back to '{inv1}' for Measurement #2.")
        inv2 = inv1
    elif not inv1_exists:
        print(f"Note: '{inv1}' not found. Falling back to '{inv2}' for Measurement #1.")
        inv1 = inv2

    return (meas1, stim1, inv1), (meas2, stim2, inv2)


def save_float32_wav(output_path, fs, data):
    """Saves a 1-D float64 array as a 32-bit IEEE float WAV file, creating directories if needed."""
    out_p = Path(output_path)
    if out_p.parent and str(out_p.parent) not in ("", "."):
        out_p.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(out_p), int(fs), data.astype(np.float32))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Farina_IAS - Farina Impulse Alignment & Subtraction\n\n"
            "Performs 3-File System-Calibrated Farina Deconvolution on two DaDooRunRun\n"
            "measurement sets collected with the Measure, Move-Move, Measure (MMMM)\n"
            "technique, aligns Measurement #2 to Measurement #1 at sub-sample resolution,\n"
            "and optionally computes the free-field subtraction (Equations A1.3 & A1.4)\n"
            "using either a real constant (--r_value) or an FRD file (--r_frd).\n\n"
            "Input basenames automatically resolve to:\n"
            "  <base>_m.wav   (microphone recording)\n"
            "  <base>_s.wav   (recorded loopback/amplifier stimulus)\n"
            "  <base>_Fif.wav (Farina inverse filter; falls back to the other set if missing)\n\n"
            "Output files derived from --outfile basename (default: 'impulse'):\n"
            "  <outfile>_1.wav   (calibrated Measurement #1 IR)\n"
            "  <outfile>_2.wav   (aligned calibrated Measurement #2 IR)\n"
            "  <outfile>_1-2.wav (free-field subtraction IR, written when --r_value or --r_frd is given)"
        ),
        usage='use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--in1", "--b1", type=str, required=True,
        help="Input basename for Measurement #1 DaDooRunRun files.",
    )
    parser.add_argument(
        "--in2", "--b2", type=str, required=True,
        help="Input basename for Measurement #2 DaDooRunRun files.",
    )
    parser.add_argument(
        "--outfile", "--of", "-o", type=str, default="impulse",
        help="Output basename for generated WAV files (default: 'impulse').",
    )

    # Mutually exclusive groundplane reflection specification: constant or FRD file
    r_group = parser.add_mutually_exclusive_group()
    r_group.add_argument(
        "--r_value", "-r", type=float, default=None,
        help=(
            "Real-valued groundplane reflection coefficient R (e.g., 1.0 for ideal reflector). "
            "Computes p_FF = p_1 - (R / (1 + R)) * p_2_aligned and saves <outfile>_1-2.wav."
        ),
    )
    r_group.add_argument(
        "--r_frd", "-R", type=str, default=None,
        help=(
            "Path to FRD file describing the frequency-dependent groundplane reflection R(f). "
            "Magnitude is interpolated/extrapolated to DC and Nyquist, reconstructed as minimum "
            "phase, applied in the frequency domain, and saved to <outfile>_1-2.wav."
        ),
    )

    parser.add_argument(
        "--no_clamp", action="store_true",
        help="Do not clamp |R| to a maximum of 1.0 (0 dB). By default, |R| is hard-clamped to <= 1.0.",
    )
    parser.add_argument(
        "--ir-length-ms", type=float, default=500.0,
        help="Calibrated measurement IR window length in ms (default: 500.0).",
    )
    parser.add_argument(
        "--stim-window-ms", type=float, default=50.0,
        help="Electrical stimulus window length in ms (default: 50.0).",
    )
    parser.add_argument(
        "--use-f1sw", action="store_true",
        help="Use f1sw instead of fhfw for upper out-of-band conditioning.",
    )
    parser.add_argument(
        "--max-window-us", type=float, default=500.0,
        help="Cross-correlation alignment search window in microseconds (default: 500.0).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Suppress the cross-correlation alignment plot.",
    )

    args = parser.parse_args()

    try:
        # Validate timing/window CLI parameters upfront
        if args.ir_length_ms <= 0.0:
            raise ValueError(f"Invalid --ir-length-ms ({args.ir_length_ms}): must be > 0.")
        if args.stim_window_ms <= 0.0:
            raise ValueError(f"Invalid --stim-window-ms ({args.stim_window_ms}): must be > 0.")
        if args.max_window_us <= 0.0:
            raise ValueError(f"Invalid --max-window-us ({args.max_window_us}): must be > 0.")

        # Validate R inputs early
        r_const = args.r_value
        if r_const is not None:
            if r_const <= -1.0:
                raise ValueError(f"Invalid --r_value ({r_const}): must be > -1.0.")
            if r_const > 1.0 and not args.no_clamp:
                print(
                    f"Note: --r_value ({r_const}) exceeds 1.0. Hard-clamping R to 1.0 "
                    "(use --no_clamp to disable)."
                )
                r_const = 1.0
            elif r_const < 0.0:
                print(f"Warning: --r_value ({r_const}) is negative. Proceeding anyway.")

        if args.r_frd is not None and not os.path.isfile(args.r_frd):
            raise FileNotFoundError(f"FRD file not found: '{args.r_frd}'")

        set1, set2 = resolve_input_sets(args.in1, args.in2)

        print("Deconvolving MMMM Measurement #1...")
        fs1, data_master, info1 = deconvolve_calibrated_ir(
            *set1,
            ir_length_ms=args.ir_length_ms,
            stim_window_ms=args.stim_window_ms,
            use_f1sw=args.use_f1sw,
        )
        print(
            f"  Sample Rate: {fs1} Hz | Acoustic Delay: {info1['tof_ms']:.3f} ms "
            f"(offset={info1['delta_n_tof']} samples) | Band: [{info1['band'][0]:.1f}, {info1['band'][1]:.1f}] Hz"
        )

        print("Deconvolving MMMM Measurement #2...")
        fs2, data_slave, info2 = deconvolve_calibrated_ir(
            *set2,
            ir_length_ms=args.ir_length_ms,
            stim_window_ms=args.stim_window_ms,
            use_f1sw=args.use_f1sw,
        )
        print(
            f"  Sample Rate: {fs2} Hz | Acoustic Delay: {info2['tof_ms']:.3f} ms "
            f"(offset={info2['delta_n_tof']} samples) | Band: [{info2['band'][0]:.1f}, {info2['band'][1]:.1f}] Hz"
        )

        if fs1 != fs2:
            raise ValueError(f"Sampling rate mismatch: Meas1={fs1} Hz, Meas2={fs2} Hz.")
        fs = fs1

        # Match array lengths by zero-padding the tail of the shorter IR
        target_len = max(len(data_master), len(data_slave))
        if len(data_master) < target_len:
            data_master = np.pad(data_master, (0, target_len - len(data_master)))
        if len(data_slave) < target_len:
            data_slave = np.pad(data_slave, (0, target_len - len(data_slave)))

        # Zero-padded FFT length (>= 2 * target_len) to prevent circular wrap-around
        n_fft_sub = 1 << (2 * target_len - 1).bit_length()
        P1 = np.fft.rfft(data_master, n=n_fft_sub)
        P2 = np.fft.rfft(data_slave, n=n_fft_sub)

        # Build W(f) = R(f) / (1 + R(f)) when either --r_value or --r_frd is supplied
        W_f = None
        if r_const is not None:
            R_f = np.full(n_fft_sub // 2 + 1, r_const, dtype=np.complex128)
            W_f = compute_reflection_weight(R_f)
            print(f"\nUsing constant reflection coefficient R = {r_const:.4f} (W = {r_const / (1.0 + r_const):.6f}).")
        elif args.r_frd is not None:
            print(f"\nBuilding minimum-phase R(f) from '{args.r_frd}' (N_FFT={n_fft_sub})...")
            R_f, r_info = build_minimum_phase_r_from_frd(
                args.r_frd, fs, n_fft_sub, no_clamp=args.no_clamp
            )
            W_f = compute_reflection_weight(R_f)
            clamp_status = (
                "disabled (--no_clamp)"
                if args.no_clamp
                else f"enabled ({r_info['clamped_points']} FRD pts clamped <= 0 dB)"
            )
            print(
                f"  FRD Range : [{r_info['f_min']:.1f}, {r_info['f_max']:.1f}] Hz | "
                f"HF Extrap Slope: {r_info['hf_slope_db_per_oct']:.2f} dB/oct"
            )
            print(
                f"  |R(f)|    : [{r_info['min_mag']:.4f}, {r_info['max_mag']:.4f}] | "
                f"Clamping: {clamp_status}"
            )

        # If W(f) is active, apply it before cross-correlation so any spectral tilt
        # or group delay in W(f) matches Measurement #1's reflection during alignment.
        if W_f is not None:
            align_slave_sig = np.fft.irfft(W_f * P2, n=n_fft_sub)[:target_len]
        else:
            align_slave_sig = data_slave

        print(f"\nAnalyzing alignment (Search window restricted to +/-{args.max_window_us} microseconds)...")
        exact_lag, coarse_lag, fine_offset, max_lag, lags, corr, peak_idx = (
            calculate_subsample_offset(
                data_master, align_slave_sig, fs, max_window_us=args.max_window_us
            )
        )

        time_offset_us = (exact_lag / fs) * 1e6
        print("\n--- Alignment Analysis ---")
        print(f"Search Constraints : +/-{max_lag} samples")
        print(f"Coarse Sample Lag  : {coarse_lag} samples")
        print(f"Sub-sample Delta   : {fine_offset:+.4f} samples")
        print(f"Calculated Offset  : {time_offset_us:.3f} microseconds")

        print("\nCorrecting alignment via Fourier phase shift...")
        shift_freqs = np.fft.rfftfreq(n_fft_sub)
        shift_vector = np.exp(-2j * np.pi * shift_freqs * exact_lag)
        P2_aligned = P2 * shift_vector
        aligned_slave = np.fft.irfft(P2_aligned, n=n_fft_sub)[:target_len]

        # Save calibrated Measurement #1 and aligned Measurement #2
        out1_path = f"{args.outfile}_1.wav"
        out2_path = f"{args.outfile}_2.wav"

        save_float32_wav(out1_path, fs, data_master)
        print(f"Saved calibrated Measurement #1 IR to '{out1_path}'.")

        save_float32_wav(out2_path, fs, aligned_slave)
        print(f"Saved aligned Measurement #2 IR to '{out2_path}'.")

        # Optional MMMM Free-Field Subtraction (Equations A1.3 & A1.4) in frequency domain
        if W_f is not None:
            P_ff = P1 - W_f * P2_aligned
            free_field = np.fft.irfft(P_ff, n=n_fft_sub)[:target_len]
            out_ff_path = f"{args.outfile}_1-2.wav"
            save_float32_wav(out_ff_path, fs, free_field)
            print(f"Performed frequency-domain MMMM subtraction and saved free-field IR to '{out_ff_path}'.")

        if not args.no_plot:
            print("\nRendering sub-sample fit visualization...")
            plot_parabolic_fit(lags, corr, peak_idx, coarse_lag, fine_offset)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main()
