/* ========================================================================== */
/*                                                                            */
/*   ProbeAudio.cpp                                                           */
/*   Copyright (c) 2026 Charlie Laub                                          */
/*   Licensed under the MIT License. See LICENSE file in the project root     */
/*     for full license information.                                          */
/*                                                                            */
/*   Probes the audio cards and devices available to the system. Lists        */
/*   output and input channels. These values can be use when invoking         */
/*   other audio processing programs.                                         */
/*                                                                            */
/* ========================================================================== */

#include <iostream>
#include <vector>
#include <portaudio.h>

void check_error(PaError err) {
    if (err != paNoError) {
        std::cerr << "PortAudio Error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        exit(EXIT_FAILURE);
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "     ProbeAudio - An Audio Hardware Diagnostic Tool      \n";
    std::cout << "=========================================================\n";

    // Initialize PortAudio System
    PaError err = Pa_Initialize();
    check_error(err);

    // 1. Enumerate Available Audio APIs (Drivers)
    int num_apis = Pa_GetHostApiCount();
    if (num_apis < 0) {
        std::cerr << "Error fetching API count.\n";
        return -1;
    }

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
        std::cout << "\n   Total Hardware Devices managed: " << api_info->deviceCount << "\n\n";
    }

    // 2. Enumerate Discrete Audio Hardware Devices
    int total_devices = Pa_GetDeviceCount();
    if (total_devices < 0) {
        std::cerr << "Error fetching hardware device count.\n";
        return -1;
    }

    std::cout << "[Step 2] Comprehensive Physical Device & Channel Maps:\n";
    std::cout << "---------------------------------------------------------\n";
    
    for (int i = 0; i < total_devices; ++i) {
        const PaDeviceInfo* dev_info = Pa_GetDeviceInfo(i);
        if (!dev_info) continue;

        const PaHostApiInfo* parent_api = Pa_GetHostApiInfo(dev_info->hostApi);
        std::string api_name = parent_api ? parent_api->name : "Unknown API";

        std::cout << "Device Index [" << i << "]: " << dev_info->name << "\n";
        std::cout << "  * Managed by Driver API  : [" << dev_info->hostApi << "] (" << api_name << ")\n";
        std::cout << "  * Max Input Channels     : " << dev_info->maxInputChannels << "\n";
        std::cout << "  * Max Output Channels    : " << dev_info->maxOutputChannels << "\n";
        std::cout << "  * Default Sample Rate    : " << dev_info->defaultSampleRate << " Hz\n";

        // Query common sample rates to verify duplex feasibility
        std::vector<double> test_rates = { 44100.0, 48000.0, 96000.0, 192000.0 };
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

            // Check if full capability configuration works
            const PaStreamParameters* in_ptr = (dev_info->maxInputChannels > 0) ? &in_params : nullptr;
            const PaStreamParameters* out_ptr = (dev_info->maxOutputChannels > 0) ? &out_params : nullptr;

            PaError format_supported = Pa_IsFormatSupported(in_ptr, out_ptr, rate);
            if (format_supported == paFormatIsSupported) {
                if (!first) std::cout << ", ";
                std::cout << rate / 1000.0 << "kHz";
                first = false;
            }
        }
        if (first) std::cout << "None detected via blocking query structure.";
        std::cout << "\n\n---------------------------------------------------------\n";
    }

    // Always clean up resources before closing context strings
    Pa_Terminate();
    std::cout << "Diagnostic run complete.\n";
    return 0;
}
