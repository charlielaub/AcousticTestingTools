/* ========================================================================== */
/*                                                                            */
/*    ProbeAudio.cpp                                                          */
/*    Copyright (c) 2026 Charlie Laub                                         */
/*    Licensed under the MIT License. See LICENSE file in the project root    */
/*      for full license information.                                         */
/*                                                                            */
/*    Probes the audio cards and devices available to the system. Lists       */
/*    output and input channels. These values can be use when invoking        */
/*    other audio processing programs.                                        */
/*                                                                            */
/*    Usage: ProbeAudioDevs -c or --choose_from_list, or omit to see all      */
/*                                                                            */
/* ========================================================================== */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <portaudio.h>

// Conditionally include Windows ASIO specific headers to query low-level buffer parameters
#ifdef _WIN32
#include <windows.h>
#include "pa_asio.h"
#endif

void check_error(PaError err) {
    if (err != paNoError) {
        std::cerr << "PortAudio Error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }
}

// Helper function to extract a clean base name depending on the platform/API convention
// Windows: "Out 3-4 (MOTU M Series)" -> "MOTU M Series" (Name is inside parentheses)
// Linux ALSA: "HDA Intel PCH: ALC892 Analog (hw:0,0)" -> "HDA Intel PCH: ALC892 Analog" (Name is before parentheses)
std::string get_base_device_name(const std::string& raw_name, bool is_alsa) {
    if (is_alsa) {
        size_t open_paren = raw_name.find("(hw:");
        if (open_paren == std::string::npos) {
            open_paren = raw_name.rfind('(');
        }
        if (open_paren != std::string::npos && open_paren > 0) {
            std::string base = raw_name.substr(0, open_paren);
            // Trim trailing whitespace
            while (!base.empty() && base.back() == ' ') {
                base.pop_back();
            }
            if (!base.empty()) {
                return base;
            }
        }
        return raw_name;
    } else {
        size_t open_paren = raw_name.rfind('(');
        size_t close_paren = raw_name.rfind(')');
        if (open_paren != std::string::npos && close_paren != std::string::npos && close_paren > open_paren) {
            return raw_name.substr(open_paren + 1, close_paren - open_paren - 1);
        }
        return raw_name;
    }
}

// Helper function to print detailed diagnostics for a specific device index
void print_device_details(int i) {
    const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
    if (!dev_info) return;

    const PaHostApiInfo* parent_api = Pa_GetHostApiInfo(dev_info->hostApi);
    std::string api_name = parent_api ? parent_api->name : "Unknown API";
    bool is_asio = (parent_api && parent_api->type == paASIO);

    std::cout << "\n---------------------------------------------------------\n";
    std::cout << "Device Index [" << i << "]: " << dev_info->name << "\n";
    std::cout << "  * Managed by Driver API  : [" << dev_info->hostApi << "] (" << api_name << ")\n";
    std::cout << "  * Max Input Channels     : " << dev_info->maxInputChannels << "\n";
    std::cout << "  * Max Output Channels    : " << dev_info->maxOutputChannels << "\n";
    std::cout << "  * Default Sample Rate    : " << dev_info->defaultSampleRate << " Hz\n";
    std::cout << "  * Low I/O Latency Line   : In: " << dev_info->defaultLowInputLatency * 1000.0 << " ms | Out: " << dev_info->defaultLowOutputLatency * 1000.0 << " ms\n";

    // EXTRA ASIO DIAGNOSTICS: Extract internal hardware buffer step values
    #ifdef _WIN32
    if (is_asio) {
        long minBufferSize, maxBufferSize, preferredBufferSize, granularity;
        PaError asio_err = PaAsio_GetAvailableBufferSizes(i, &minBufferSize, &maxBufferSize, &preferredBufferSize, &granularity);
        if (asio_err == paNoError) {
            std::cout << "  * ASIO Hardware Buffers  : Preferred: " << preferredBufferSize 
                      << " | Range: [" << minBufferSize << " -> " << maxBufferSize 
                      << "] | Step: " << granularity << " samples\n";
        }
    }
    #endif

    // Query common sample rates safely
    std::vector<double> test_rates = { 44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0 };
    std::cout << "  * Supported Sample Rates : ";
    bool first = true;
    
    for (double rate : test_rates) {
        PaStreamParameters in_params, out_params;
        
        in_params.device = i;
        in_params.channelCount = dev_info->maxInputChannels;
        in_params.sampleFormat = paFloat32;
        in_params.suggestedLatency = dev_info->defaultLowInputLatency;
        in_params.hostApiSpecificStreamInfo = nullptr;

        out_params.device = i;
        out_params.channelCount = dev_info->maxOutputChannels;
        out_params.sampleFormat = paFloat32;
        out_params.suggestedLatency = dev_info->defaultLowOutputLatency;
        out_params.hostApiSpecificStreamInfo = nullptr;

        const PaStreamParameters* in_ptr = (dev_info->maxInputChannels > 0) ? &in_params : nullptr;
        const PaStreamParameters* out_ptr = (dev_info->maxOutputChannels > 0) ? &out_params : nullptr;

        bool input_supported = !in_ptr || (Pa_IsFormatSupported(in_ptr, nullptr, rate) == paFormatIsSupported);
        bool output_supported = !out_ptr || (Pa_IsFormatSupported(nullptr, out_ptr, rate) == paFormatIsSupported);
        
        if (input_supported && output_supported) {
            if (!first) std::cout << ", ";
            std::cout << rate / 1000.0 << "kHz";
            first = false;
        }
    }
    if (first) std::cout << "None detected via fallback query constraints.";
    std::cout << "\n---------------------------------------------------------\n";
}

int main(int argc, char* argv[]) {
    // Check for command-line arguments (-c or --choose_from_list)
    bool choose_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" || arg == "--choose_from_list") {
            choose_mode = true;
            break;
        }
    }

    std::cout << "=========================================================\n";
    std::cout << "     ProbeAudio - An Audio Hardware Diagnostic Tool      \n";
    std::cout << "=========================================================\n";

    // Initialize PortAudio System
    PaError err = Pa_Initialize();
    check_error(err);

    int num_apis = Pa_GetHostApiCount();
    if (num_apis < 0) {
        std::cerr << "Error fetching API count.\n";
        Pa_Terminate();
        return -1;
    }

    if (choose_mode) {
        // --- INTERACTIVE MODE ---
        std::cout << "\n[Interactive Mode] Detected System Driver APIs:\n";
        std::cout << "---------------------------------------------------------\n";
        int default_api = Pa_GetDefaultHostApi();
        for (int i = 0; i < num_apis; ++i) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(i);
            if (!api_info) continue;
            
            std::cout << " API Index [" << i << "] : " << api_info->name;
            if (i == default_api) std::cout << " (SYSTEM DEFAULT)";
            if (api_info->type == paASIO) std::cout << " <-- [ASIO]";
            std::cout << " (" << api_info->deviceCount << " devices)\n";
        }

        int chosen_api = -1;
        std::cout << "\nEnter API Index to investigate: ";
        std::cin >> chosen_api;

        if (std::cin.fail() || chosen_api < 0 || chosen_api >= num_apis) {
            std::cerr << "Invalid API index selection.\n";
            Pa_Terminate();
            return -1;
        }

        const PaHostApiInfo* chosen_api_info = Pa_GetHostApiInfo(chosen_api);
        bool is_alsa = (chosen_api_info->type == paALSA);

        std::cout << "\n[Interactive Mode] Grouped Hardware Names under API: " << chosen_api_info->name << "\n";
        std::cout << "---------------------------------------------------------\n";

        int total_devices = Pa_GetDeviceCount();
        std::vector<std::string> unique_base_names;
        std::map<std::string, std::vector<int>> base_name_to_indices;

        for (int i = 0; i < total_devices; ++i) {
            const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
            if (dev_info && dev_info->hostApi == chosen_api) {
                std::string base_name = get_base_device_name(dev_info->name, is_alsa);
                if (base_name_to_indices.find(base_name) == base_name_to_indices.end()) {
                    unique_base_names.push_back(base_name);
                }
                base_name_to_indices[base_name].push_back(i);
            }
        }

        if (unique_base_names.empty()) {
            std::cout << "No devices found for this API.\n";
            Pa_Terminate();
            return 0;
        }

        for (size_t u = 0; u < unique_base_names.size(); ++u) {
            std::cout << " [" << u << "] " << unique_base_names[u] 
                      << " (" << base_name_to_indices[unique_base_names[u]].size() << " endpoint entries)\n";
        }

        size_t chosen_name_idx = 0;
        std::cout << "\nEnter Device Name Index to inspect: ";
        std::cin >> chosen_name_idx;

        if (std::cin.fail() || chosen_name_idx >= unique_base_names.size()) {
            std::cerr << "Invalid device name selection.\n";
            Pa_Terminate();
            return -1;
        }

        std::string selected_base_name = unique_base_names[chosen_name_idx];
        const std::vector<int>& matching_indices = base_name_to_indices[selected_base_name];

        std::cout << "\n[Results] Showing all " << matching_indices.size() 
                  << " endpoint entry/entries for: \"" << selected_base_name << "\"\n";
        for (int idx : matching_indices) {
            print_device_details(idx);
        }

    } else {
        // --- DEFAULT FULL LISTING MODE ---
        std::cout << "\n[Step 1] Detected System Driver APIs / Backends:\n";
        std::cout << "---------------------------------------------------------\n";
        int default_api = Pa_GetDefaultHostApi();
        for (int i = 0; i < num_apis; ++i) {
            const PaHostApiInfo* api_info = Pa_GetHostApiInfo(i);
            if (!api_info) continue;
            
            std::cout << " API Index [" << i << "] : " << api_info->name;
            if (i == default_api) {
                std::cout << " (SYSTEM DEFAULT)";
            }
            if (api_info->type == paASIO) {
                std::cout << " <-- [NATIVE LOW-LATENCY ASIO DETECTED]";
            }
            std::cout << "\n    Total Hardware Devices managed: " << api_info->deviceCount << "\n\n";
        }

        int total_devices = Pa_GetDeviceCount();
        if (total_devices < 0) {
            std::cerr << "Error fetching hardware device count.\n";
            Pa_Terminate();
            return -1;
        }

        std::cout << "[Step 2] Comprehensive Physical Device & Channel Maps:\n";
        std::cout << "---------------------------------------------------------\n";
        
        for (int i = 0; i < total_devices; ++i) {
            print_device_details(i);
            std::cout << "\n";
        }
    }

    // Always clean up resources before closing context strings
    Pa_Terminate();
    std::cout << "Diagnostic run complete.\n";
    return 0;
}