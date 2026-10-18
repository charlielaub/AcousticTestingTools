/* ========================================================================== */
/*                                                                            */
/*   DaDooRunRun.cpp                                                          */
/*   Copyright (c) 2026 Charlie Laub                                          */
/*   Licensed under the MIT License. See LICENSE file in the project root     */
/*     for full license information.                                          */
/*                                                                            */
/*   Generates a windowed and filtered chirp stimulus signal. Plays the       */
/*   signal via two output channels while recording from two input channels.  */
/*   The test signal can be repeated N times with silence of a user defined   */
/*   duration in between runs. The N time records are aligned and averaged    */
/*   and the data is saved to WAV files in FLOAT32 format.                    */
/*                                                                            */
/*   Invoke with -h or --help to see a list of all program options.           */
/*                                                                            */
/* ========================================================================== */

#define NOMINMAX
#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <utility>
#include <random>
#include <chrono>
#include <thread>
#include <memory>
#include <portaudio.h>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Conditionally include Windows ASIO specific headers to query low-level buffer parameters
#ifdef _WIN32
#include <windows.h>
#include "pa_asio.h"
#endif

// Struct to handle parsed arguments
struct AppArgs {
    std::string mic_output_file = "mic_data.wav";
    std::string stimulus_output_file = "";
    int sample_rate = 48000;
    float chirp_rate = 2.0f;
    float start_freq = 1.0f;
    int repetitions = 1;
    float warm_up_delay = 3.0f;
    float mean_wait = 1.0f;
    float level = -6.0f; // Default level in dB (must be <= 0.0)
    int api_idx = -1;
    int input_device_idx = -1;
    int output_device_idx = -1;
    int out_channels[2] = {-1, -1};
    int in_channels[2] = {-1, -1};
};


struct AudioContext {
    const std::vector<float>& stimulus;
    std::vector<std::vector<float>>& ref_history;
    std::vector<std::vector<float>>& mic_history;
    
    int current_rep = 0;
    int current_frame = 0;
    int total_samples = 0;
    int pa_in_channels = 2;
    int pa_out_channels = 2;
    int ch_in0 = 0;
    int ch_in1 = 1;
    bool is_warming_up = true;
    bool measurement_complete = false;
};


// --- Helper: Display Usage / Help Information ---
void show_help_info(const char* prog_name) {
    std::cout << "Acoustic Testing Measurement Tool\n\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --mic_output_file, --om <file>   WAV file for mic data (default: mic_data.wav)\n"
              << "  --stimulus_output_file, --os <file> WAV file for stimulus reference\n"
              << "  --level <float>                 Max stimulus level in dB <= 0.0 (default: -6.0)\n"
              << "  --sample_rate <int>             Sampling rate in Hz (default: 48000)\n"
              << "  --chirp_rate <float>            Chirp rate in seconds per decade of frequency (default: 2.0)\n"
              << "  --start_freq <float>            Start frequency in Hz (default: 1.0)\n"
              << "  --repetitions <int>             Number of measurement runs (default: 1)\n"
              << "  --warm_up_delay <float>         Length of DAC warm-up delay in seconds (default: 3.0)\n"
              << "  --mean_wait <float>             Average pause between runs in seconds (default: 3.0)\n"
              << "  --api_idx <int>                 PortAudio host API index\n"
              << "  --input_device_idx <int>        PortAudio hardware input device index\n"
              << "  --output_device_idx <int>       PortAudio hardware output device index\n"
              << "  --input_channels <id1> <id2>    Two physical input channel indices\n"
              << "  --output_channels <id1> <id2>   Two physical output channel indices\n"
              << "  --help, -h                      Display this help message and exit\n\n"
              << "NOTES:\n"
              << "    The command line parameters may be provided in any order.\n"
              << "    The first input channel records the stimulus signal.\n"
              << "    The second input channel records the microphone signal.\n\n";
}


// --- Helper: Manual Command Line Parsing ---
AppArgs parse_arguments(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            show_help_info(argv[0]);
            std::exit(0);
        }
        else if ((arg == "--mic_output_file" || arg == "--om") && i + 1 < argc) args.mic_output_file = argv[++i];
        else if ((arg == "--stimulus_output_file" || arg == "--os") && i + 1 < argc) args.stimulus_output_file = argv[++i];
        else if (arg == "--sample_rate" && i + 1 < argc) args.sample_rate = std::stoi(argv[++i]);
        else if (arg == "--chirp_rate" && i + 1 < argc) args.chirp_rate = std::stof(argv[++i]);
        else if (arg == "--start_freq" && i + 1 < argc) args.start_freq = std::stof(argv[++i]);
        else if (arg == "--repetitions" && i + 1 < argc) args.repetitions = std::stoi(argv[++i]);
        else if (arg == "--warm_up_delay" && i + 1 < argc) args.warm_up_delay = std::stoi(argv[++i]);
        else if (arg == "--mean_wait" && i + 1 < argc) args.mean_wait = std::stof(argv[++i]);
        else if (arg == "--level" && i + 1 < argc) {
            float parsed_level = std::stof(argv[++i]);
            if (parsed_level > 0.0f) {
                std::cerr << "Warning: Level must be 0 dB or less. Restoring default value of -6.0 dB.\n";
                args.level = -6.0f;
            } else {
                args.level = parsed_level;
            }
        }
        else if (arg == "--api_idx" && i + 1 < argc) args.api_idx = std::stoi(argv[++i]);
        else if (arg == "--input_device_idx" && i + 1 < argc) args.input_device_idx = std::stoi(argv[++i]);
        else if (arg == "--output_device_idx" && i + 1 < argc) args.output_device_idx = std::stoi(argv[++i]);
        else if (arg == "--input_channels" && i + 2 < argc) {
            args.in_channels[0] = std::stoi(argv[++i]);
            args.in_channels[1] = std::stoi(argv[++i]);
        }
        else if (arg == "--output_channels" && i + 2 < argc) {
            args.out_channels[0] = std::stoi(argv[++i]);
            args.out_channels[1] = std::stoi(argv[++i]);
        }
    }
    return args;
}


// --- Helper: Write 32-bit Float Mono WAV File ---
void write_float32_wav(const std::string& filename, const std::vector<float>& data, int sample_rate) {
    std::string full_name = filename;
    if (full_name.size() < 4 || full_name.substr(full_name.size() - 4) != ".wav") {
        full_name += ".wav";
    }

    std::ofstream file(full_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << full_name << std::endl;
        return;
    }

    int num_samples = data.size();
    int byte_rate = sample_rate * 4;

    // Write RIFF Header
    file.write("RIFF", 4);
    int chunk_size = 36 + num_samples * 4;
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4);

    // Write fmt Chunk
    file.write("fmt ", 4);
    int subchunk1_size = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    short audio_format = 3; // 3 = IEEE Float
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    short num_channels = 1;
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    short block_align = 4;
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    short bits_per_sample = 32;
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // Write data Chunk
    file.write("data", 4);
    int subchunk2_size = num_samples * 4;
    file.write(reinterpret_cast<const char*>(&subchunk2_size), 4);
    file.write(reinterpret_cast<const char*>(data.data()), subchunk2_size);

    std::cout << " Saved: " << full_name << " (Format: Mono, 32-bit Float)" << std::endl;
}


// --- Step 1 & 2: Hardware Device Management ---
bool check_host_api_and_device(AppArgs& args) {
    std::cout << "\nInitializing and Validating the Audio Device...\n";
    Pa_Initialize();
    int num_apis = Pa_GetHostApiCount();

    if (args.api_idx == -1) {
        std::cout << "\nFATAL ERROR: no API index selected via command line parameter --api_idx\n";
        return false;
    }

    const PaHostApiInfo* api_info = Pa_GetHostApiInfo(args.api_idx);
    std::cout << "Driver API: " << api_info->name << "\n";

    int total_devices = Pa_GetDeviceCount();
    std::vector<int> valid_devices;
    for (int i = 0; i < total_devices; ++i) {
        const PaDeviceInfo* dev = Pa_GetDeviceInfo(i);
        if (dev->hostApi == args.api_idx) {
            valid_devices.push_back(i);
        }
    }

    if (valid_devices.empty()) {
        std::cerr << "ERROR: No hardware devices found under this API.\n";
        return false;
    }

    if (args.input_device_idx == -1) {
        std::cout << "\nFATAL ERROR: no input device was selected via the command line parameter --input_device_idx.\n";
        return false;
    } else {
      const PaDeviceInfo* selected_input_dev = Pa_GetDeviceInfo(args.input_device_idx);
      std::cout << "Input Device: " << selected_input_dev->name << "\n";
    }

    if (args.output_device_idx == -1) {
        std::cout << "\nFATAL ERROR: no output device was selected via the command line parameter --output_device_idx.\n";
        return false;
    } else {
        const PaDeviceInfo* selected_output_dev = Pa_GetDeviceInfo(args.output_device_idx);
        std::cout << "Output Device: " << selected_output_dev->name << "\n";
    }

    if ( args.in_channels[0] == -1 || args.in_channels[1]  == -1 )  {
        std::cout << "\nFATAL ERROR: two input channels were not selected via the command line parameter --input_channels.\n";
    } else {
        std::cout << "  First input channel number: " << args.in_channels[0] << "\n";
        std::cout << "  Second input channel number: " << args.in_channels[1] << "\n";
    }

    if (args.out_channels[0] == -1) {
        std::cout << "\nFATAL ERROR: two output channels were not selected via the command line parameter --output_channels.\n";
    } else {
        std::cout << "  First output channel number: " << args.out_channels[0] << "\n";
        std::cout << "  Second output channel number: " << args.out_channels[1] << "\n";
    }

    return true;
}


// --- Step 3: Math and Stimulus Generation (with padding, filtering, gain, and clipping) ---
std::vector<float> generate_stimulus(int sample_rate, float chirp_rate, float start_freq, float level_dB) {
    // Promote core calculation variables to double precision
    double d_sample_rate = static_cast<double>(sample_rate);
    double d_start_freq  = static_cast<double>(start_freq);
    double d_chirp_rate  = static_cast<double>(chirp_rate);
    double d_level_dB    = static_cast<double>(level_dB);

    double nyquist = 0.5 * d_sample_rate;
    double end_freq = 0.99 * nyquist; // End at 99% of Nyquist frequency

    // Basic input validation
    if (d_start_freq <= 0.0 || end_freq <= d_start_freq || d_chirp_rate <= 0.0 || sample_rate <= 0) {
        throw std::invalid_argument("Invalid parameters for chirp generation.");
    }

    double total_decades = std::log10(end_freq / d_start_freq);
    double duration = d_chirp_rate * total_decades;
    int num_samples = static_cast<int>(d_sample_rate * duration);

    std::vector<float> stimulus(num_samples);

    // Convert decibels to a linear amplitude multiplier in double precision
    double amplitude = std::pow(10.0, d_level_dB / 20.0);

    // Precompute constant terms in double precision outside the loop
    const double ln_ratio = std::log(end_freq / d_start_freq);
    const double phase_factor = d_start_freq * duration / ln_ratio;
    const double ratio = end_freq / d_start_freq;
    const double two_pi = 2.0 * M_PI;

    // 1. Generate Logarithmic Chirp in Double Precision & Cast to Float
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / d_sample_rate;
        double phase = two_pi * phase_factor * (std::pow(ratio, t / duration) - 1.0);
        
        // Cast the high-precision double result down to float upon storage
        stimulus[i] = static_cast<float>(amplitude * std::sin(phase));
    }
 
    // 2. Apply Tapers (Half-Hann windows)
    int start_length = static_cast<int>(sample_rate * (duration * (0.3f / total_decades)));
    int end_length = static_cast<int>(sample_rate * (duration * (0.15f / total_decades)));

    for (int i = 0; i < start_length; ++i) {
        float hann = 0.5f * (1.0f - std::cos(M_PI * i / start_length));
        stimulus[i] *= hann;
    }
    for (int i = 0; i < end_length; ++i) {
        float hann = 0.5f * (1.0f - std::cos(M_PI * i / end_length));
        stimulus[num_samples - end_length + i] *= (1.0f - hann);
    }

    // 3. Add 0.5sec of zero padding at the end
    int end_pad_samples = static_cast<int>(sample_rate * 0.5f);
    stimulus.insert(stimulus.end(), end_pad_samples, 0.0f);

// --- Dynamic Design of a Stable 4th-Order Highpass Bessel Filter ---
    // The corner frequency is 4 times the start frequency
    double corner_freq = 4.0 * static_cast<double>(start_freq);
    double fs = static_cast<double>(sample_rate);

    // 1. Correct Pre-warped analog angular frequency for the cutoff
    double absolute_w0 = 2.0 * fs * std::tan(M_PI * corner_freq / fs); 

    Eigen::MatrixXd sos(2, 6);

    // 4th-order Bessel normalized lowpass prototype parameters (Stage 0 and Stage 1).
    // Using standard coefficients for the two biquad stages (Q and frequency scaling factors):
    // Stage 1: Q1 = 0.805538, frequency multiplier = 1.603358
    // Stage 2: Q2 = 0.521935, frequency multiplier = 1.430172
    // (Values derived from standard analog filter tables / miniDSP / analog design references)
    double q[2]       = { 0.80553828, 0.52193458 };
    double freq_mult[2] = { 1.60335752, 1.43017156 };

    for (int s = 0; s < 2; ++s) {
        // Scale the pre-warped omega for each stage's specific pole pair
        double stage_w0 = absolute_w0 * freq_mult[s];
        double stage_w0_sq = stage_w0 * stage_w0;
        double fs2 = 2.0 * fs;
        double fs2_sq = fs2 * fs2;
        
        // Parameter alpha for bilinear transform of a highpass biquad
        // Denominator coefficients in analog domain: s^2 + (w0/Q)s + w0^2
        double alpha_bessel = stage_w0 / q[s];

        // Bilinear transform equations for Highpass Filter
        double a0 = fs2_sq + alpha_bessel * fs2 + stage_w0_sq;
        double a1 = 2.0 * stage_w0_sq - 2.0 * fs2_sq;
        double a2 = fs2_sq - alpha_bessel * fs2 + stage_w0_sq;

        // Highpass zeros at DC (z = 1)
        double b0 = fs2_sq;
        double b1 = -2.0 * fs2_sq;
        double b2 = fs2_sq;

        sos(s, 0) = b0 / a0; sos(s, 1) = b1 / a0; sos(s, 2) = b2 / a0;
        sos(s, 3) = 1.0;     sos(s, 4) = a1 / a0; sos(s, 5) = a2 / a0;
    }

    // --- Apply the calculated SOS filter using 64-bit State Registers ---
    std::vector<double> filtered(stimulus.begin(), stimulus.end());

    // Process all SOS stages entirely in double precision
    for (int s = 0; s < 2; ++s) {
        double b0 = sos(s, 0), b1 = sos(s, 1), b2 = sos(s, 2);
        double a1 = sos(s, 4), a2 = sos(s, 5); 
        
        double x1 = 0.0, x2 = 0.0; 
        double y1 = 0.0, y2 = 0.0;

        for (size_t i = 0; i < filtered.size(); ++i) {
            double x0 = filtered[i]; 
            double y0 = (b0 * x0) + (b1 * x1) + (b2 * x2) - (a1 * y1) - (a2 * y2);
            
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;
            
            filtered[i] = y0; 
        }
    }

    // Convert the final double vector back to float in a separate step
    std::vector<float> final_output(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        final_output[i] = static_cast<float>(filtered[i]);
    }

    // --- Hard clip individual samples that exceed +/- 1.0 (0 dBFS) ---
    int clipped_count = 0;
    for (float& sample : final_output) {
        if (sample > 1.0f) {
            sample = 1.0f;
            clipped_count++;
        } else if (sample < -1.0f) {
            sample = -1.0f;
            clipped_count++;
        }
    }
    if (clipped_count > 0) {
        std::cout << "Warning: " << clipped_count << " samples exceeded +/- 1.0 and were clipped.\n";
    }

    return final_output;
}


// Fractional delay filter with integer shift separation and zero-padding
std::vector<float> fractional_delay_filter(const std::vector<float>& input, float total_shift) {
    size_t size = input.size();
    std::vector<float> output(size, 0.0f);

    if (size == 0) {
        return output;
    }

    // 1. Separate total shift into integer (gross) and fractional (fine) parts.
    // std::round keeps the fractional part strictly between -0.5 and +0.5.
    int integer_shift = static_cast<int>(std::round(total_shift));
    float fractional_shift = total_shift - static_cast<float>(integer_shift);

    const int num_taps = 31; // Must be odd. Higher = better quality, slightly slower.
    const int half_taps = num_taps / 2;

    for (size_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int t = -half_taps; t <= half_taps; ++t) {
            // 2. Incorporate the integer shift into the sample index lookup
            long long sample_idx = static_cast<long long>(i) - t - integer_shift;

            // Sinc function: sin(pi * x) / (pi * x)
            float x = static_cast<float>(t) - fractional_shift;
            float sinc_val;
            if (std::abs(x) < 1e-5f) {
                sinc_val = 1.0f;
            } else {
                float pi_x = static_cast<float>(M_PI) * x;
                sinc_val = std::sin(pi_x) / pi_x;
            }

            // Hann Window
            float normalized_n = static_cast<float>(t + half_taps) / static_cast<float>(num_taps - 1);
            float hann_window = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * normalized_n));

            float coeff = sinc_val * hann_window;

            // 3. Apply zero padding for out-of-bounds indices
            float sample_val = 0.0f;
            if (sample_idx >= 0 && sample_idx < static_cast<long long>(size)) {
                sample_val = input[sample_idx];
            }

            sum += sample_val * coeff;
            weight_sum += coeff;
        }

        // Normalize weights, defaulting to zero if completely out of bounds
        if (std::abs(weight_sum) > 1e-5f) {
            output[i] = sum / weight_sum;
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}


// Main alignment and averaging function with sub-sample precision
std::pair<std::vector<float>, std::vector<float>> align_and_average(
    const std::vector<std::vector<float>>& ref_history, 
    const std::vector<std::vector<float>>& mic_history,
    float sample_rate) 
{
    size_t num_runs = ref_history.size();
    if (num_runs == 0) return {};
    size_t signal_size = ref_history[0].size();
    
    std::vector<std::vector<float>> aligned_ref = {ref_history[0]};
    std::vector<std::vector<float>> aligned_mic = {mic_history[0]};

    // Hardcoded local static value for max time lag in milliseconds
    static constexpr float max_time_ms = 3.0f;

    // Convert time window to number of samples based on sample rate
    int max_lag = static_cast<int>(std::round((max_time_ms / 1000.0f) * sample_rate));

    if (num_runs > 1) {
        std::cout << "\n--- Performing Sub-Sample Cross-Correlation Alignment ---\n";
        std::cout << "Max lag window: " << max_time_ms << " ms (" << max_lag << " samples at " << sample_rate << " Hz)\n";
    }

    std::vector<float> shifted_ref;
    std::vector<float> shifted_mic;

    for (size_t i = 1; i < num_runs; ++i) {
        float max_corr = -1e20f;
        int best_lag = 0;

        int range_size = 2 * max_lag + 1;
        std::vector<float> corr_scores(range_size, -1e20f);

        // 1. Find best integer lag using normalized cross-correlation
        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            float current_dot = 0.0f;
            float ref0_energy = 0.0f;
            float ref_i_energy = 0.0f;

            for (size_t n = 0; n < signal_size; ++n) {
                long long ref_idx = static_cast<long long>(n) + lag;
                if (ref_idx >= 0 && ref_idx < static_cast<long long>(signal_size)) {
                    float val0 = ref_history[0][n];
                    float vali = ref_history[i][ref_idx];
                    current_dot += val0 * vali;
                    ref0_energy += val0 * val0;
                    ref_i_energy += vali * vali;
                }
            }

            float denominator = std::sqrt(ref0_energy * ref_i_energy);
            float ncc = (denominator > 1e-5f) ? (current_dot / denominator) : 0.0f;

            corr_scores[lag + max_lag] = ncc;

            if (ncc > max_corr) {
                max_corr = ncc;
                best_lag = lag;
            }
        }

        // 2. Refine lag to sub-sample precision using Parabolic Interpolation
        float refined_lag = static_cast<float>(best_lag);
        int best_idx = best_lag + max_lag;

        if (best_lag > -max_lag && best_lag < max_lag) {
            float y_minus = corr_scores[best_idx - 1];
            float y_zero  = corr_scores[best_idx];
            float y_plus  = corr_scores[best_idx + 1];

            float denominator = (y_minus - 2.0f * y_zero + y_plus);
            if (std::abs(denominator) > 1e-5f) {
                float delta = 0.5f * (y_minus - y_plus) / denominator;
                refined_lag += delta;
            }
        }

        std::cout << " Run " << i + 1 << " timing jitter: " << refined_lag << " samples.";
        
        if ( std::abs(refined_lag) < 0.2 ) {
            shifted_ref = ref_history[i];
            shifted_mic = mic_history[i];
        } else {
            std::cout << " Correcting timing jitter for this run.\n";
            // 3. Shift both reference and microphone signals using the fractional delay filter
            shifted_ref = fractional_delay_filter(ref_history[i], refined_lag);
            shifted_mic = fractional_delay_filter(mic_history[i], refined_lag);
        }

        aligned_ref.push_back(shifted_ref);
        aligned_mic.push_back(shifted_mic);

    }

    // 4. Compute average across all runs
    std::vector<float> avg_ref(signal_size, 0.0f);
    std::vector<float> avg_mic(signal_size, 0.0f);
    for (size_t n = 0; n < signal_size; ++n) {
        for (size_t i = 0; i < num_runs; ++i) {
            avg_ref[n] += aligned_ref[i][n];
            avg_mic[n] += aligned_mic[i][n];
        }
        avg_ref[n] /= static_cast<float>(num_runs);
        avg_mic[n] /= static_cast<float>(num_runs);
    }

    return {avg_ref, avg_mic};
}


// Context structure to manage state inside the PortAudio callback
struct MeasurementContext {
    const std::vector<float>& stimulus;
    std::vector<std::vector<float>>& ref_history;
    std::vector<std::vector<float>>& mic_history;
    
    int total_samples = 0;
    int pa_in_channels = 2;
    int pa_out_channels = 2;
    int ch_in0 = 0;
    int ch_in1 = 1;
    
    // State machine flags and counters
    enum Phase { WARMUP, MEASURING, PAUSING, COMPLETE };
    Phase current_phase = WARMUP;
    
    int current_rep = 0;
    int total_reps = 1;
    long long warmup_frames_remaining = 0;
    long long pause_frames_remaining = 0;
    int current_frame = 0;
};


// PortAudio Callback Function (Runs in high-priority audio thread)
static int measurementCallback(const void* inputBuffer, void* outputBuffer,
                               unsigned long framesPerBuffer,
                               const PaStreamCallbackTimeInfo* timeInfo,
                               PaStreamCallbackFlags statusFlags,
                               void* userData) {
    auto* ctx = static_cast<MeasurementContext*>(userData);
    const float* in = static_cast<const float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        float out_sample = 0.0f;

        // State Machine handling
        switch (ctx->current_phase) {
            case MeasurementContext::WARMUP:
                out_sample = 0.0f;
                if (ctx->warmup_frames_remaining > 0) {
                    ctx->warmup_frames_remaining--;
                }
                break;

            case MeasurementContext::MEASURING:
                if (ctx->current_frame < ctx->total_samples) {
                    out_sample = ctx->stimulus[ctx->current_frame];
                } else {
                    out_sample = 0.0f;
                }
                break;

            case MeasurementContext::PAUSING:
                out_sample = 0.0f;
                if (ctx->pause_frames_remaining > 0) {
                    ctx->pause_frames_remaining--;
                }
                break;

            case MeasurementContext::COMPLETE:
                out_sample = 0.0f;
                break;
        }

        // Write output channels
        out[i * ctx->pa_out_channels + 0] = out_sample;
        out[i * ctx->pa_out_channels + 1] = out_sample;

        // Read input channels (if currently measuring)
        if (ctx->current_phase == MeasurementContext::MEASURING) {
            if (ctx->current_frame < ctx->total_samples && in) {
                ctx->ref_history[ctx->current_rep][ctx->current_frame] = in[i * ctx->pa_in_channels + ctx->ch_in0];
                ctx->mic_history[ctx->current_rep][ctx->current_frame] = in[i * ctx->pa_in_channels + ctx->ch_in1];
            }
            ctx->current_frame++;

            // Check if this repetition is finished
            if (ctx->current_frame >= ctx->total_samples) {
                ctx->current_rep++;
                ctx->current_frame = 0;
                
                if (ctx->current_rep >= ctx->total_reps) {
                    ctx->current_phase = MeasurementContext::COMPLETE;
                } else {
                    ctx->current_phase = MeasurementContext::PAUSING; // Will be initialized by main thread
                }
            }
        }
    }

    return (ctx->current_phase == MeasurementContext::COMPLETE) ? paComplete : paContinue;
}


std::pair<std::vector<float>, std::vector<float>> perform_acoustic_measurement(AppArgs& args, const std::vector<float>& stimulus) {
    int total_samples = stimulus.size();
    
    // 1. Detect Host API type
    const PaDeviceInfo* inputDevInfo = Pa_GetDeviceInfo(args.input_device_idx);
    const PaDeviceInfo* outputDevInfo = Pa_GetDeviceInfo(args.output_device_idx);
    bool isAsio = false;
    #ifdef _WIN32
    if (inputDevInfo && Pa_GetHostApiInfo(inputDevInfo->hostApi)->type == paASIO) isAsio = true;
    #endif

    int pa_out_channels = isAsio ? 2 : std::max(args.out_channels[0], args.out_channels[1]) + 1;
    int pa_in_channels  = isAsio ? 2 : std::max(args.in_channels[0],  args.in_channels[1])  + 1;
    if (pa_out_channels < 2) pa_out_channels = 2;
    if (pa_in_channels < 2)  pa_in_channels  = 2;

    // 2. Allocate histories
    std::vector<std::vector<float>> ref_history(args.repetitions, std::vector<float>(total_samples, 0.0f));
    std::vector<std::vector<float>> mic_history(args.repetitions, std::vector<float>(total_samples, 0.0f));

    // 3. Initialize Context
    MeasurementContext ctx{
        stimulus, ref_history, mic_history,
        total_samples, pa_in_channels, pa_out_channels,
        isAsio ? 0 : args.in_channels[0],
        isAsio ? 1 : args.in_channels[1],
        MeasurementContext::WARMUP,
        0, args.repetitions,
        static_cast<long long>(args.warm_up_delay * args.sample_rate),
        0, 0
    };

    // 4. Configure Stream Parameters
    PaStreamParameters inputParameters{}, outputParameters{};
    inputParameters.device = args.input_device_idx;
    inputParameters.channelCount = pa_in_channels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = inputDevInfo ? inputDevInfo->defaultHighInputLatency : 0.1;

    outputParameters.device = args.output_device_idx;
    outputParameters.channelCount = pa_out_channels;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = outputDevInfo ? outputDevInfo->defaultHighOutputLatency : 0.1;

    #ifdef _WIN32
    PaAsioStreamInfo asioIn{}, asioOut{};
    static int inSel[2], outSel[2];
    if (isAsio) {
        inSel[0] = args.in_channels[0]; inSel[1] = args.in_channels[1];
        asioIn.size = sizeof(PaAsioStreamInfo); asioIn.hostApiType = paASIO; asioIn.version = 1;
        asioIn.flags = paAsioUseChannelSelectors; asioIn.channelSelectors = inSel;
        inputParameters.hostApiSpecificStreamInfo = &asioIn;

        outSel[0] = args.out_channels[0]; outSel[1] = args.out_channels[1];
        asioOut.size = sizeof(PaAsioStreamInfo); asioOut.hostApiType = paASIO; asioOut.version = 1;
        asioOut.flags = paAsioUseChannelSelectors; asioOut.channelSelectors = outSel;
        outputParameters.hostApiSpecificStreamInfo = &asioOut;
    }
    #endif

    std::cout << "\nStarting " << args.repetitions << " sequential measurements using " << args.sample_rate << " Hz sampling rate...\n";
    std::cout << "Opening hardware stream with " << pa_in_channels << " inputs and " << pa_out_channels << " outputs.\n";

    PaStream* stream = nullptr;
    PaError err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, args.sample_rate, 
                                paFramesPerBufferUnspecified, paNoFlag, measurementCallback, &ctx);
    if (err != paNoError) {
        std::cerr << "PortAudio Error opening stream: " << Pa_GetErrorText(err) << "\n";
        return {};
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio Error starting stream: " << Pa_GetErrorText(err) << "\n";
        Pa_CloseStream(stream);
        return {};
    }

    // 5. Execute Warm-up Phase with Console Output
    std::cout << "Warming up the output device for " << args.warm_up_delay << " seconds with silence...\n";
    while (ctx.warmup_frames_remaining > 0 && Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(20);
    }

    // 6. Measurement Repetitions Loop with Console Logs & Pauses
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int r = 0; r < args.repetitions; ++r) {
        // Log repetition start
        std::cout << "Capturing measurement #" << (r + 1) << " of " << args.repetitions << "...\n";

        // Switch state to measuring for this repetition
        ctx.current_rep = r;
        ctx.current_frame = 0;
        ctx.current_phase = MeasurementContext::MEASURING;

        // Wait until this repetition completes
        while (ctx.current_phase == MeasurementContext::MEASURING && Pa_IsStreamActive(stream) == 1) {
            Pa_Sleep(20);
        }

        // Handle Inter-measurement Pause (if not the last repetition)
        if (r < args.repetitions - 1) {
            float max_variation = args.mean_wait * 0.10f;
            std::uniform_real_distribution<float> dis(-max_variation, max_variation);
            float actual_wait = args.mean_wait + dis(gen);
            
            std::cout << "    Entering the inter-measurement pause. Waiting " << actual_wait << " seconds ...\n";
            
            long long pause_samples = static_cast<long long>(actual_wait * args.sample_rate);
            ctx.pause_frames_remaining = pause_samples;
            ctx.current_phase = MeasurementContext::PAUSING;

            // Wait until the pause silence finishes feeding
            while (ctx.pause_frames_remaining > 0 && Pa_IsStreamActive(stream) == 1) {
                Pa_Sleep(20);
            }
        }
    }

    // 7. Cleanup
    Pa_StopStream(stream);
    Pa_CloseStream(stream);

    // 8. Return aligned and averaged results
    return align_and_average(ref_history, mic_history, args.sample_rate);
}




int main(int argc, char* argv[]) {
    // Show usage/help info automatically if no arguments are provided
    if (argc == 1) {
        show_help_info(argv[0]);
        return 0;
    }

    AppArgs args = parse_arguments(argc, argv);

    if (!check_host_api_and_device(args)) {
        return -1;
    }

    std::vector<float> stimulus = generate_stimulus(args.sample_rate, args.chirp_rate, args.start_freq, args.level);

    auto [final_avg_ref, final_avg_mic] = perform_acoustic_measurement(args, stimulus);

    // Check if the unpacked vectors are empty (indicating failure)
    if (final_avg_ref.empty() && final_avg_mic.empty()) {
        std::cerr << "Error: The measurement aborted due to stream failure.\n";
        Pa_Terminate(); // Don't forget to clean up PortAudio if initialized here
        return EXIT_FAILURE; 
    }

    std::cout << "\n--- Saving Data Files ---\n";
    if (!args.stimulus_output_file.empty()) {
        write_float32_wav(args.stimulus_output_file, final_avg_ref, args.sample_rate);
    } else {
        std::cout << " Stimulus output file pathname omitted by user - stimulus data not saved.\n";
    }

    write_float32_wav(args.mic_output_file, final_avg_mic, args.sample_rate);
    std::cout << " Microphone data file saved.\n\nDone with all measurement tasks.\n";

    Pa_Terminate();
    return 0;
}
