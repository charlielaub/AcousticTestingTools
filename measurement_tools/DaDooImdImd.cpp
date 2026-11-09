/* ========================================================================== */
/*                                                                            */
/*   DaDooImdImd.cpp                                                          */
/*   Copyright (c) 2026 Charlie Laub                                          */
/*                                                                            */
/*   Generates a crest-factor-optimized multi-tone stimulus for IMD           */
/*   measurement. Plays the signal continuously in a cyclic buffer while      */
/*   recording. Uses a Zero-Lag NCC check to reject USB clock slips or        */
/*   transient noise, performing on-the-fly Time Synchronous Averaging (TSA). */
/*                                                                            */
/* ========================================================================== */

#define NOMINMAX
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cctype>
#include <unordered_set>
#include <filesystem>
#include <atomic>

#include <portaudio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <windows.h>
#include "pa_asio.h"
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif


// -----------------------------------------------------------------------------
// Program Parameters
// -----------------------------------------------------------------------------
struct AppArgs {
    std::string basefilename = "imd_results";
    int sample_rate = 48000;
    int fft_exp = 16;              // e.g. 16 = 65536 bins
    float start_freq = 20.0f;
    float end_freq = 20000.0f;
    float ppo = 3.0f;              // Points (tones) per octave
    int num_reps = 16;          // Number of successful blocks to average
    int warmup_blocks = 2;         // Blocks to play before recording starts
    float level = -10.0f;          // dB FS
    int api_idx = -1;
    int input_device_idx = -1;
    int output_device_idx = -1;
    int out_channels[2] = {-1, -1};
    int in_channels[2] = {-1, -1};
    uint32_t timestamp;

    void print() const {
        const int name_width = 20;
        std::cout << "\n============ IMD Parameters ============\n";
        std::cout << std::right;
        std::cout << std::setw(name_width) << "base file name"    << " : " << basefilename << "\n";
        std::cout << std::setw(name_width) << "sample rate"       << " : " << sample_rate << " Hz\n";
        std::cout << std::setw(name_width) << "FFT exponent"      << " : " << fft_exp << " (" << (1 << fft_exp) << " bins)\n";
        std::cout << std::setw(name_width) << "start freq"        << " : " << start_freq << " Hz\n";
        std::cout << std::setw(name_width) << "end freq"          << " : " << end_freq << " Hz\n";
        std::cout << std::setw(name_width) << "points per octave" << " : " << ppo << "\n";
        std::cout << std::setw(name_width) << "num_reps"          << " : " << num_reps << "\n";
        std::cout << std::setw(name_width) << "warmup blocks"     << " : " << warmup_blocks << "\n";
        std::cout << std::setw(name_width) << "level"             << " : " << level << " dB\n";
        std::cout << "========================================\n\n";
    }
};

void show_help_info(const char* prog_name) {
    std::cout << "Multi-Tone IMD Measurement Tool\n\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --basefilename, -b <file>           Base file name (default: imd_results)\n"
              << "  --level, -l <float>                 RMS stimulus level in dB <= 0.0 (default: -10.0)\n"
              << "  --sample_rate, -r <int>             Sampling rate in Hz (default: 48000)\n"
              << "  --fft_exp, -e <int>                 FFT Size as power of 2 (default: 16 -> 65536)\n"
              << "  --start_freq, -s <float>            Start frequency in Hz (default: 20.0)\n"
              << "  --end_freq, -f <float>              End frequency in Hz (default: 20000.0)\n"
              << "  --ppo, -p <float>                   Tones per octave (default: 3.0)\n"
              << "  --num_reps, -n <int>                Number of valid runs to average (default: 16)\n"
              << "  --warmup_blocks, -w <int>           Blocks to play before recording (default: 2)\n"
              << "  --api_idx, -a <int>                 PortAudio host API index\n"
              << "  --input_device_idx, -i <int>        PortAudio hardware input device index\n"
              << "  --input_channels, -I <id1> <id2>    Two physical input channel indices\n"
              << "  --output_device_idx, -o <int>       PortAudio hardware output device index\n"
              << "  --output_channels, -O <id1> <id2>   Two physical output channel indices\n"
              << "  --help, -h                          Display this help message and exit\n\n";
}

bool parseAndValidateInt(const std::string& argName, const char* strValue, int& outValue, int minAllowed) {
    try {
        size_t idx = 0;
        int val = std::stoi(strValue, &idx);
        if (strValue[idx] != '\0') return false;
        if (val < minAllowed) return false;
        outValue = val;
        return true;
    } catch (...) { return false; }
}

bool parseAndValidateFloat(const std::string& argName, const char* strValue, float& outValue, bool mustBePositive = true) {
    try {
        size_t idx = 0;
        float val = std::stof(strValue, &idx);
        if (strValue[idx] != '\0') return false;
        if (mustBePositive && val <= 0.0f) return false;
        outValue = val;
        return true;
    } catch (...) { return false; }
}

AppArgs parse_arguments(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            show_help_info(argv[0]);
            std::exit(0);
        } else if (arg == "--basefilename" || arg == "-b") {
            args.basefilename = argv[++i];
        } else if (arg == "--sample_rate" || arg == "-r") {
            parseAndValidateInt(arg, argv[++i], args.sample_rate, 1);
        } else if (arg == "--fft_exp" || arg == "-e") {
            parseAndValidateInt(arg, argv[++i], args.fft_exp, 10); // Min 1024
        } else if (arg == "--start_freq" || arg == "-s") {
            parseAndValidateFloat(arg, argv[++i], args.start_freq, true);
        } else if (arg == "--end_freq" || arg == "-f") {
            parseAndValidateFloat(arg, argv[++i], args.end_freq, true);
        } else if (arg == "--ppo" || arg == "-p") {
            parseAndValidateFloat(arg, argv[++i], args.ppo, true);
        } else if (arg == "--num_reps" || arg == "-n") {
            parseAndValidateInt(arg, argv[++i], args.num_reps, 1);
        } else if (arg == "--warmup_blocks" || arg == "-w") {
            parseAndValidateInt(arg, argv[++i], args.warmup_blocks, 0);
        } else if (arg == "--level" || arg == "-l") {
            parseAndValidateFloat(arg, argv[++i], args.level, false);
            if (args.level > 0.0f) args.level = -10.0f;
        } else if (arg == "--api_idx" || arg == "-a") {
            parseAndValidateInt(arg, argv[++i], args.api_idx, 0);
        } else if (arg == "--input_device_idx" || arg == "-i") {
            parseAndValidateInt(arg, argv[++i], args.input_device_idx, 0);
        } else if (arg == "--output_device_idx" || arg == "-o") {
            parseAndValidateInt(arg, argv[++i], args.output_device_idx, 0);
        } else if (arg == "--input_channels" || arg == "-I") {
            parseAndValidateInt(arg, argv[++i], args.in_channels[0], 0);
            parseAndValidateInt(arg, argv[++i], args.in_channels[1], 0);
        } else if (arg == "--output_channels" || arg == "-O") {
            parseAndValidateInt(arg, argv[++i], args.out_channels[0], 0);
            parseAndValidateInt(arg, argv[++i], args.out_channels[1], 0);
        }
    }
    return args;
}


// -----------------------------------------------------------------------------
// Core Math & Signal Generation
// -----------------------------------------------------------------------------
double calculate_crest_factor(const std::vector<double>& signal) {
    double peak = 0.0;
    double sum_sq = 0.0;
    for (double val : signal) {
        double abs_val = std::abs(val);
        if (abs_val > peak) peak = abs_val;
        sum_sq += val * val;
    }
    double rms = std::sqrt(sum_sq / signal.size());
    if (rms == 0) return 0.0;
    return 20.0 * std::log10(peak / rms);
}

std::vector<float> generate_multitone(const AppArgs& args) {
    int fft_size = 1 << args.fft_exp;
    double freq_resolution = (double)args.sample_rate / fft_size;
    
    std::vector<int> active_bins;
    double octaves = std::log2(args.end_freq / args.start_freq);
    int num_points = static_cast<int>(std::ceil(octaves * args.ppo));
    
    for (int i = 0; i <= num_points; ++i) {
        double target_hz = args.start_freq * std::pow(2.0, i / args.ppo);
        if (target_hz > args.end_freq) break;
        int bin = static_cast<int>(std::round(target_hz / freq_resolution));
        if (active_bins.empty() || bin > active_bins.back()) {
            active_bins.push_back(bin);
        }
    }

    std::cout << "Generating multi-tone array with " << active_bins.size() << " perfectly aligned tones...\n";

    // Fixed seed guarantees identical stimulus every time the program runs
    std::mt19937 gen(1337); 
    std::uniform_real_distribution<double> phase_dist(-M_PI, M_PI);

    std::vector<double> best_signal(fft_size, 0.0);
    double best_crest_factor = 999.0;
    std::vector<double> temp_signal(fft_size);
    std::vector<double> temp_phases(active_bins.size());

    // Monte Carlo Phase Optimization
    for (int iter = 0; iter < 500; ++iter) {
        std::fill(temp_signal.begin(), temp_signal.end(), 0.0);
        for (size_t i = 0; i < active_bins.size(); ++i) {
            temp_phases[i] = phase_dist(gen);
        }
        for (size_t i = 0; i < active_bins.size(); ++i) {
            int k = active_bins[i];
            double phase = temp_phases[i];
            for (int n = 0; n < fft_size; ++n) {
                temp_signal[n] += std::cos(2.0 * M_PI * k * n / fft_size + phase);
            }
        }
        double cf = calculate_crest_factor(temp_signal);
        if (cf < best_crest_factor) {
            best_crest_factor = cf;
            best_signal = temp_signal;
        }
    }

    std::cout << "Optimized Crest Factor: " << best_crest_factor << " dB\n";

    // Normalize and scale to user dB level
    double max_val = 0.0;
    for (double val : best_signal) {
        if (std::abs(val) > max_val) max_val = std::abs(val);
    }
    double amplitude = std::pow(10.0, args.level / 20.0);
    
    std::vector<float> final_signal(fft_size);
    for (int n = 0; n < fft_size; ++n) {
        final_signal[n] = static_cast<float>((best_signal[n] / max_val) * amplitude);
    }
    return final_signal;
}


// -----------------------------------------------------------------------------
// Real-Time Callback Context
// -----------------------------------------------------------------------------
struct IMDContext {
    const std::vector<float>& stimulus;
    int fft_size;
    
    int pa_in_channels, pa_out_channels, ch_in0, ch_in1;
    
    enum Phase { WARMUP, RECORD_ANCHOR, RECORD_TSA, COMPLETE };
    Phase current_phase = WARMUP;

    int blocks_remaining;
    int successful_reps = 0;
    int target_reps;
    int playback_idx = 0;
    int record_idx = 0;

    std::vector<float> anchor_block;
    std::vector<float> temp_block;
    
    // Double precision prevents floating-point drift over many iterations
    std::vector<double> tsa_accumulator; 
    
    // Thread-safe feedback for console
    std::atomic<int> slips_detected{0};
};

static int measurementCallback(const void* inputBuffer, void* outputBuffer,
                               unsigned long framesPerBuffer,
                               const PaStreamCallbackTimeInfo* timeInfo,
                               PaStreamCallbackFlags statusFlags,
                               void* userData) {
    auto* ctx = static_cast<IMDContext*>(userData);
    const float* in = static_cast<const float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        // 1. Continuous Cyclic Playback
        float out_sample = ctx->stimulus[ctx->playback_idx];
        out[i * ctx->pa_out_channels + 0] = out_sample;
        out[i * ctx->pa_out_channels + 1] = out_sample;
        
        ctx->playback_idx++;
        if (ctx->playback_idx >= ctx->fft_size) ctx->playback_idx = 0;

        // 2. State Machine Recording
        float mic_sample = in ? in[i * ctx->pa_in_channels + ctx->ch_in1] : 0.0f;

        switch (ctx->current_phase) {
            case IMDContext::WARMUP:
                ctx->record_idx++;
                if (ctx->record_idx >= ctx->fft_size) {
                    ctx->record_idx = 0;
                    ctx->blocks_remaining--;
                    if (ctx->blocks_remaining <= 0) {
                        ctx->current_phase = IMDContext::RECORD_ANCHOR;
                    }
                }
                break;

            case IMDContext::RECORD_ANCHOR:
                ctx->anchor_block[ctx->record_idx] = mic_sample;
                ctx->record_idx++;
                if (ctx->record_idx >= ctx->fft_size) {
                    ctx->record_idx = 0;
                    ctx->current_phase = IMDContext::RECORD_TSA;
                }
                break;

            case IMDContext::RECORD_TSA:
                ctx->temp_block[ctx->record_idx] = mic_sample;
                ctx->record_idx++;
                
                // End of block: perform Zero-Lag QA inline
                if (ctx->record_idx >= ctx->fft_size) {
                    ctx->record_idx = 0;
                    
                    double dot = 0.0, e_anchor = 0.0, e_temp = 0.0;
                    for (int n = 0; n < ctx->fft_size; ++n) {
                        float a = ctx->anchor_block[n];
                        float t = ctx->temp_block[n];
                        dot += a * t;
                        e_anchor += a * a;
                        e_temp += t * t;
                    }
                    
                    double denominator = std::sqrt(e_anchor * e_temp);
                    double ncc = (denominator > 1e-9) ? (dot / denominator) : 0.0;

                    // Strict threshold to reject USB slips or transient noise
                    if (ncc >= 0.98) {
                        for (int n = 0; n < ctx->fft_size; ++n) {
                            ctx->tsa_accumulator[n] += ctx->temp_block[n];
                        }
                        ctx->successful_reps++;
                        if (ctx->successful_reps >= ctx->target_reps) {
                            ctx->current_phase = IMDContext::COMPLETE;
                        }
                    } else {
                        // Slip detected: ignore temp_block and try again next loop
                        ctx->slips_detected++;
                    }
                }
                break;
                
            case IMDContext::COMPLETE:
                break;
        }
    }

    return (ctx->current_phase == IMDContext::COMPLETE) ? paComplete : paContinue;
}


// -----------------------------------------------------------------------------
// File I/O
// -----------------------------------------------------------------------------
bool write_wav_file_plus_info(const AppArgs& args, const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t sample_rate = args.sample_rate;
    uint32_t num_samples = data.size();
    uint32_t data_bytes = num_samples * sizeof(float);
    uint32_t xdat_payload_size = 56; 
    uint32_t riff_chunk_size = 4 + 24 + (8 + data_bytes) + (8 + xdat_payload_size);

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    
    uint32_t fmt_chunk_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
    
    uint16_t audio_format = 3; 
    uint16_t num_channels = 1;
    uint32_t byte_rate = sample_rate * sizeof(float);
    uint16_t block_align = sizeof(float);
    uint16_t bits_per_sample = 32;

    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_bytes), 4);
    if (num_samples > 0) file.write(reinterpret_cast<const char*>(data.data()), data_bytes);

    // IMD-specific metadata Chunk
    file.write("xDAT", 4);
    file.write(reinterpret_cast<const char*>(&xdat_payload_size), 4);

    auto write_f = [&](const char tag[4], float val) {
        file.write(tag, 4); file.write(reinterpret_cast<const char*>(&val), 4);
    };
    auto write_u = [&](const char tag[4], uint32_t val) {
        file.write(tag, 4); file.write(reinterpret_cast<const char*>(&val), 4);
    };

    write_f("f0sw", args.start_freq);
    write_f("f1sw", args.end_freq);
    write_f("ppo ", args.ppo);
    write_u("ffte", static_cast<uint32_t>(args.fft_exp));
    write_f("levl", args.level);
    write_u("date", static_cast<uint32_t>(std::time(nullptr)));
    write_u("reps", static_cast<uint32_t>(args.num_reps));

    file.close();
    return file.good();
}


// -----------------------------------------------------------------------------
// MAIN ENTRY
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc == 1) {
        show_help_info(argv[0]);
        return 0;
    }

    AppArgs args = parse_arguments(argc, argv);
    args.print();

    // 1. Initialize Audio
    Pa_Initialize();
    int pa_in_channels = 2;
    int pa_out_channels = 2;

    std::vector<float> stimulus = generate_multitone(args);
    int fft_size = stimulus.size();

    // 2. Setup Context
    IMDContext ctx{
        stimulus, fft_size,
        pa_in_channels, pa_out_channels, args.in_channels[0], args.in_channels[1],
        IMDContext::WARMUP, args.warmup_blocks, 0, args.num_reps, 0, 0,
        std::vector<float>(fft_size, 0.0f),
        std::vector<float>(fft_size, 0.0f),
        std::vector<double>(fft_size, 0.0)
    };

    PaStreamParameters inParams{}, outParams{};
    
    // --- INPUT SETUP ---
    inParams.device = args.input_device_idx;
    inParams.channelCount = pa_in_channels;
    inParams.sampleFormat = paFloat32;
    
    // Request safe ALSA buffer geometry
    const PaDeviceInfo* inDeviceInfo = Pa_GetDeviceInfo(inParams.device);
    if (inDeviceInfo) {
        inParams.suggestedLatency = inDeviceInfo->defaultHighInputLatency;
    } else {
        inParams.suggestedLatency = 0.1; // 100ms fallback
    }

    // --- OUTPUT SETUP ---
    outParams.device = args.output_device_idx;
    outParams.channelCount = pa_out_channels;
    outParams.sampleFormat = paFloat32;
    
    // Request safe ALSA buffer geometry
    const PaDeviceInfo* outDeviceInfo = Pa_GetDeviceInfo(outParams.device);
    if (outDeviceInfo) {
        outParams.suggestedLatency = outDeviceInfo->defaultHighOutputLatency;
    } else {
        outParams.suggestedLatency = 0.1; // 100ms fallback
    }

    // Explicitly define a safe buffer size rather than leaving it up to ALSA
    unsigned long framesPerBuffer = 2048; 

    PaStream* stream = nullptr;
    PaError err = Pa_OpenStream(&stream, &inParams, &outParams, args.sample_rate, 
                                framesPerBuffer, paNoFlag, measurementCallback, &ctx);
    if (err != paNoError) return -1;

    // 3. Collect Data
    std::cout << "\nStarting Continuous IMD Data Collection...\n";
    Pa_StartStream(stream);

    int last_reported_reps = -1;
    int last_reported_slips = -1;

    while (ctx.current_phase != IMDContext::COMPLETE && Pa_IsStreamActive(stream) == 1) {
        if (ctx.successful_reps != last_reported_reps) {
            std::cout << "\rAveraging Block: " << ctx.successful_reps << " / " << args.num_reps << std::flush;
            last_reported_reps = ctx.successful_reps;
        }
        int current_slips = ctx.slips_detected.load();
        if (current_slips != last_reported_slips && current_slips > 0) {
            std::cout << "\n  [Warning] Timing slip or noise detected. Block rejected. Retrying...\n";
            last_reported_slips = current_slips;
        }
        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "\n\nData Collection Complete.\n";

    // 4. Finalize TSA Average
    std::vector<float> final_mic(fft_size);
    for (int n = 0; n < fft_size; ++n) {
        final_mic[n] = static_cast<float>(ctx.tsa_accumulator[n] / args.num_reps);
    }

    // 5. Save Files
    std::string filename_mic = args.basefilename + "_IMD_MIC.wav";
    write_wav_file_plus_info(args, filename_mic, final_mic);
    std::cout << "Saved TSA Mic Data to: " << filename_mic << "\n";

    std::string filename_stim = args.basefilename + "_IMD_STIM.wav";
    write_wav_file_plus_info(args, filename_stim, stimulus);
    std::cout << "Saved Ideal Stimulus to: " << filename_stim << "\n\n";

    return 0;
}