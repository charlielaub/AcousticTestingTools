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

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <memory>
#include <portaudio.h>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Struct to handle parsed arguments
struct AppArgs {
    std::string mic_output_file = "mic_data.wav";
    std::string stimulus_output_file = "";
    int sample_rate = 48000;
    float chirp_rate = 2.0f;
    float start_freq = 1.0f;
    int repetitions = 1;
    float mean_wait = 3.0f;
    float level = -6.0f; // Default level in dB (must be <= 0.0)
    int api_idx = -1;
    int device_idx = -1;
    int out_channels[2] = {-1, -1};
    int in_channels[2] = {-1, -1};
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
              << "  --mean_wait <float>             Average pause between runs in seconds (default: 3.0)\n"
              << "  --api_idx <int>                 PortAudio host API index\n"
              << "  --device_idx <int>              PortAudio hardware device index\n"
              << "  --output_channels <id1> <id2>   Two physical output channel indices\n"
              << "  --input_channels <id1> <id2>    Two physical input channel indices\n"
              << "  --help, -h                      Display this help message and exit\n\n"
              << "NOTE: the command line parameters may be provided in any order.";
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
        else if (arg == "--device_idx" && i + 1 < argc) args.device_idx = std::stoi(argv[++i]);
        else if (arg == "--output_channels" && i + 2 < argc) {
            args.out_channels[0] = std::stoi(argv[++i]);
            args.out_channels[1] = std::stoi(argv[++i]);
        }
        else if (arg == "--input_channels" && i + 2 < argc) {
            args.in_channels[0] = std::stoi(argv[++i]);
            args.in_channels[1] = std::stoi(argv[++i]);
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
bool select_host_api_and_device(AppArgs& args) {
    Pa_Initialize();
    int num_apis = Pa_GetHostApiCount();

    if (args.api_idx == -1) {
        std::cout << "\n--- [Step 1] Select Audio Driver / API Backend ---\n";
        for (int i = 0; i < num_apis; ++i) {
            std::cout << "[" << i << "] " << Pa_GetHostApiInfo(i)->name << "\n";
        }
        std::cout << "Select driver API index: ";
        std::cin >> args.api_idx;
    }

    const PaHostApiInfo* api_info = Pa_GetHostApiInfo(args.api_idx);
    std::cout << "Active Driver API: " << api_info->name << "\n";

    int total_devices = Pa_GetDeviceCount();
    std::vector<int> valid_devices;
    std::cout << "\n--- [Step 2] Select Audio Hardware Device for " << api_info->name << " ---\n";
    for (int i = 0; i < total_devices; ++i) {
        const PaDeviceInfo* dev = Pa_GetDeviceInfo(i);
        if (dev->hostApi == args.api_idx) {
            valid_devices.push_back(i);
            std::cout << "[" << i << "] " << dev->name 
                      << " (Max Inputs: " << dev->maxInputChannels 
                      << ", Max Outputs: " << dev->maxOutputChannels << ")\n";
        }
    }

    if (valid_devices.empty()) {
        std::cerr << "No hardware devices found under this API.\n";
        return false;
    }

    if (args.device_idx == -1) {
        std::cout << "\nSelect hardware device index: ";
        std::cin >> args.device_idx;
    }

    const PaDeviceInfo* selected_dev = Pa_GetDeviceInfo(args.device_idx);
    std::cout << "Active Device: " << selected_dev->name << "\n";

    if (args.out_channels[0] == -1) {
        std::cout << "\n--- [Step 3] Configure Output Channels (Available: 0 to " << selected_dev->maxOutputChannels - 1 << ") ---\n";
        std::cout << "Enter physical output channel index for Stimulus: ";
        std::cin >> args.out_channels[0];
        std::cout << "Enter secondary physical output channel index: ";
        std::cin >> args.out_channels[1];
    }

    if (args.in_channels[0] == -1) {
        std::cout << "\n--- [Step 4] Configure Input Channels (Available: 0 to " << selected_dev->maxInputChannels - 1 << ") ---\n";
        std::cout << "Enter physical input channel index for Reference Loopback: ";
        std::cin >> args.in_channels[0];
        std::cout << "Enter physical input channel index for Microphone: ";
        std::cin >> args.in_channels[1];
    }

    return true;
}

// --- Step 3: Math and Stimulus Generation (with padding, filtering, gain, and clipping) ---
std::vector<float> generate_stimulus(int sample_rate, float chirp_rate, float start_freq, float level_dB) {
    float end_freq = 0.99f * sample_rate;
    float total_decades = std::log10(end_freq / start_freq);
    float duration = chirp_rate * total_decades;
    int num_samples = static_cast<int>(sample_rate * duration);

    std::vector<float> stimulus(num_samples);

    // 1. Generate Logarithmic Chirp
    for (int i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        stimulus[i] = std::sin(2.0f * M_PI * start_freq * duration / std::log(end_freq / start_freq) * (std::pow(end_freq / start_freq, t / duration) - 1.0f));
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

    // 3. Add 50 ms of padding at the start and 0.5 seconds at the end
    int start_pad_samples = static_cast<int>(sample_rate * 0.05f);
    int end_pad_samples = static_cast<int>(sample_rate * 0.5f);
    
    stimulus.insert(stimulus.begin(), start_pad_samples, 0.0f);
    stimulus.insert(stimulus.end(), end_pad_samples, 0.0f);


    // --- Dynamic Design of a Stable 4th-Order Highpass Butterworth Filter ---
    // The corner frequency is 4 times the start frequency
    double corner_freq = 4.0 * static_cast<double>(start_freq); // 4.0 Hz
    double fs = static_cast<double>(sample_rate);

    // 1. Correct Pre-warped analog angular frequency
    double absolute_w0 = 2.0 * fs * std::tan(M_PI * corner_freq / fs); 

    Eigen::MatrixXd sos(2, 6);
    // Butterworth pole angles for a 4th-order filter
    double angles[2] = { 0.3826834323650897, 0.9238795325112867 }; 

    for (int s = 0; s < 2; ++s) {
        double alpha = absolute_w0 * angles[s];
        double w0_sq = absolute_w0 * absolute_w0;
        double fs2 = 2.0 * fs;
        double fs2_sq = fs2 * fs2;

        // Standard bilinear transform equations using pre-warped w0
        double a0 = fs2_sq + 2.0 * alpha * fs2 + w0_sq;
        double a1 = 2.0 * w0_sq - 2.0 * fs2_sq;
        double a2 = fs2_sq - 2.0 * alpha * fs2 + w0_sq;

        double b0 = fs2_sq;
        double b1 = -2.0 * fs2_sq;
        double b2 = fs2_sq;

        sos(s, 0) = b0 / a0; sos(s, 1) = b1 / a0; sos(s, 2) = b2 / a0;
        sos(s, 3) = 1.0;    sos(s, 4) = a1 / a0; sos(s, 5) = a2 / a0;
    }

    // --- Apply the calculated SOS filter using 64-bit State Registers ---
    std::vector<float> filtered = stimulus; 
/*    
    for (int s = 0; s < 2; ++s) {
        double b0 = sos(s, 0), b1 = sos(s, 1), b2 = sos(s, 2);
        double a1 = sos(s, 4), a2 = sos(s, 5);
        
        double x1 = 0.0, x2 = 0.0; 
        double y1 = 0.0, y2 = 0.0;

        for (size_t i = 0; i < filtered.size(); ++i) {
            double x0 = static_cast<double>(filtered[i]);
            double y0 = (b0 * x0) + (b1 * x1) + (b2 * x2) - (a1 * y1) - (a2 * y2);
            
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;
            
            filtered[i] = static_cast<float>(y0);
        }
    } 
*/
    // --- Apply Decibel Level Scaling ---
    float voltage_gain = static_cast<float>(std::pow(10.0, level_dB / 20.0));
    for (float& sample : filtered) {
        sample *= voltage_gain;
    }

    // --- Hard clip individual samples that exceed +/- 1.0 (0 dBFS) ---
    int clipped_count = 0;
    for (float& sample : filtered) {
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

    return filtered;
}

// --- Step 4: Window-Constrained Cross Correlation Alignment ---
std::pair<std::vector<float>, std::vector<float>> align_and_average(
    const std::vector<std::vector<float>>& ref_history, 
    const std::vector<std::vector<float>>& mic_history) 
{
    size_t num_runs = ref_history.size();
    size_t signal_size = ref_history[0].size();
    
    std::vector<std::vector<float>> aligned_ref = {ref_history[0]};
    std::vector<std::vector<float>> aligned_mic = {mic_history[0]};

    int max_lag = 500; 

    std::cout << "\n--- Performing Window-Constrained Cross-Correlation Alignment ---\n";
    for (size_t i = 1; i < num_runs; ++i) {
        float max_corr = -1e20f;
        int best_lag = 0;

        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            float current_dot = 0.0f;
            for (size_t n = 0; n < signal_size; ++n) {
                long long ref_idx = static_cast<long long>(n) + lag;
                if (ref_idx >= 0 && ref_idx < static_cast<long long>(signal_size)) {
                    current_dot += ref_history[0][n] * ref_history[i][ref_idx];
                }
            }
            if (current_dot > max_corr) {
                max_corr = current_dot;
                best_lag = lag;
            }
        }

        std::cout << " Run " << i + 1 << " timing lag relative to baseline: " << best_lag << " samples\n";

        std::vector<float> shifted_ref(signal_size, 0.0f);
        std::vector<float> shifted_mic(signal_size, 0.0f);

        if (best_lag > 0) {
            std::copy(ref_history[i].begin(), ref_history[i].end() - best_lag, shifted_ref.begin() + best_lag);
            std::copy(mic_history[i].begin(), mic_history[i].end() - best_lag, shifted_mic.begin() + best_lag);
        } else if (best_lag < 0) {
            int abs_lag = std::abs(best_lag);
            std::copy(ref_history[i].begin() + abs_lag, ref_history[i].end(), shifted_ref.begin());
            std::copy(mic_history[i].begin() + abs_lag, mic_history[i].end(), shifted_mic.begin());
        } else {
            shifted_ref = ref_history[i];
            shifted_mic = mic_history[i];
        }

        aligned_ref.push_back(shifted_ref);
        aligned_mic.push_back(shifted_mic);
    }

    std::vector<float> avg_ref(signal_size, 0.0f);
    std::vector<float> avg_mic(signal_size, 0.0f);
    for (size_t n = 0; n < signal_size; ++n) {
        for (size_t i = 0; i < num_runs; ++i) {
            avg_ref[n] += aligned_ref[i][n];
            avg_mic[n] += aligned_mic[i][n];
        }
        avg_ref[n] /= num_runs;
        avg_mic[n] /= num_runs;
    }

    return {avg_ref, avg_mic};
}

// --- Step 5: Full-Duplex Audio Run Execution (Stream Stays Open Across Repetitions) ---
std::pair<std::vector<float>, std::vector<float>> perform_acoustic_measurement(AppArgs& args, const std::vector<float>& stimulus) {
    int block_size = 256;
    if (args.sample_rate >= 192000) block_size = 1024;
    else if (args.sample_rate >= 88200) block_size = 512;

    int total_samples = stimulus.size();

    int pa_out_channels = std::max(args.out_channels[0], args.out_channels[1]) + 1;
    int pa_in_channels  = std::max(args.in_channels[0],  args.in_channels[1])  + 1;

    if (pa_out_channels < 2) pa_out_channels = 2;
    if (pa_in_channels < 2)  pa_in_channels  = 2;

    std::vector<float> output_buffer(total_samples * pa_out_channels, 0.0f);
    for (int i = 0; i < total_samples; ++i) {
        int frame_offset = i * pa_out_channels;
        output_buffer[frame_offset + args.out_channels[0]] = stimulus[i];
    }

    std::vector<std::vector<float>> ref_history(args.repetitions, std::vector<float>(total_samples));
    std::vector<std::vector<float>> mic_history(args.repetitions, std::vector<float>(total_samples));

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\nStarting " << args.repetitions << " sequential measurements using " << args.sample_rate << "Hz sampling rate...\n";
    std::cout << "Opening hardware stream with: " << pa_in_channels << " inputs and " << pa_out_channels << " outputs (Stream remains open across all repetitions).\n";

    PaStreamParameters inputParameters, outputParameters;
    inputParameters.device = args.device_idx;
    inputParameters.channelCount = pa_in_channels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(args.device_idx)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    outputParameters.device = args.device_idx;
    outputParameters.channelCount = pa_out_channels;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(args.device_idx)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream;
    PaError err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, args.sample_rate, block_size, paNoFlag, nullptr, nullptr);
    if (err != paNoError) {
        std::cerr << "PortAudio Error opening stream: " << Pa_GetErrorText(err) << "\n";
        return {};
    }

    Pa_StartStream(stream);

    for (int r = 0; r < args.repetitions; ++r) {
        std::cout << "Capturing measurement #" << r + 1 << " of " << args.repetitions << "...\n";

        std::vector<float> input_buffer(total_samples * pa_in_channels, 0.0f);
        int chunks = total_samples / block_size;

        // Execution Loop (Full-Duplex I/O)
        for (int c = 0; c < chunks; ++c) {
            int out_offset = c * block_size * pa_out_channels;
            int in_offset  = c * block_size * pa_in_channels;

            Pa_WriteStream(stream, &output_buffer[out_offset], block_size);
            Pa_ReadStream(stream, &input_buffer[in_offset], block_size);
        }

        // Software Demultiplexing
        for (int i = 0; i < total_samples; ++i) {
            int frame_offset = i * pa_in_channels;
            ref_history[r][i] = input_buffer[frame_offset + args.in_channels[0]];
            mic_history[r][i] = input_buffer[frame_offset + args.in_channels[1]];
        }

        // Inter-measurement pause (feed stream silence chunks to avoid buffer underruns)
        if (r < args.repetitions - 1) {
            float max_variation = args.mean_wait * 0.10f;
            std::uniform_real_distribution<float> dis(-max_variation, max_variation);
            float actual_wait = args.mean_wait + dis(gen);
            std::cout << "    Entering the inter-measurement pause. Waiting " << actual_wait << " seconds ...\n";
            
            int silence_samples = static_cast<int>(actual_wait * args.sample_rate);
            int silence_chunks = silence_samples / block_size;
            
            std::vector<float> silence_block(block_size * pa_out_channels, 0.0f);
            std::vector<float> dummy_in_block(block_size * pa_in_channels, 0.0f);

            for (int sc = 0; sc < silence_chunks; ++sc) {
                Pa_WriteStream(stream, silence_block.data(), block_size);
                Pa_ReadStream(stream, dummy_in_block.data(), block_size);
            }
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);

    return align_and_average(ref_history, mic_history);
}


int main(int argc, char* argv[]) {
    // Show usage/help info automatically if no arguments are provided
    if (argc == 1) {
        show_help_info(argv[0]);
        return 0;
    }

    AppArgs args = parse_arguments(argc, argv);

    if (!select_host_api_and_device(args)) {
        return -1;
    }

    std::vector<float> stimulus = generate_stimulus(args.sample_rate, args.chirp_rate, args.start_freq, args.level);

    auto [final_avg_ref, final_avg_mic] = perform_acoustic_measurement(args, stimulus);

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
