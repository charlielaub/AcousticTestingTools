/* ========================================================================== */
/*                                                                            */
/*   DaDooRunRun.cpp                                                          */
/*   Copyright (c) 2026 Charlie Laub                                          */
/*   Licensed under the MIT License. See LICENSE file in the project root     */
/*     for full license information.                                          */
/*                                                                            */
/*   The program generates a windowed log chirp stimulus signal. Plays the    */
/*   signal via two output channels while recording from two input channels.  */
/*   The test signal can be repeated N times with silence of a user defined   */
/*   duration in between runs. The N time records are aligned and averaged    */
/*   and the data is saved to WAV files in FLOAT32 format. The Farina method  */
/*   inverse filter can also be saved to a WAV file for post-processing.      */
/*                                                                            */
/*   Invoke with -h or --help to see a list of all program options.           */
/*                                                                            */
/* ========================================================================== */

#define NOMINMAX
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <complex>
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
#include <stdexcept>
#include <cstdint> // Required for uint32_t
#include <cstddef>
#include <cctype>
#include <unordered_set>

#include <portaudio.h>
#include "pocketfft_hdronly.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Conditionally include Windows ASIO specific headers to query low-level buffer parameters
#ifdef _WIN32
#include <windows.h>
#include "pa_asio.h"
#endif

// These headers needed to silence unwanted terminal output
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif


//Structure to hold stimulus and measurement data sets
struct DataContainer {
    std::vector<std::vector<float>> stimulus_data;
    std::vector<std::vector<float>> measurement_data;
};


// Struct to handle parsed arguments
struct AppArgs {
    std::string basefilename = "results";
    int sample_rate = 48000;
    float chirp_rate = 2.0f;
    float start_freq = 1.0f;
    float end_freq = 0.99f * 0.5 * sample_rate;
    float HF_window_startF = 0.975f * end_freq;
    int repetitions = 1;
    float alignment_window = 5.0f;
    float warm_up_delay = 3.0f;
    float rt60 = 3.0f;
    float mean_wait = 1.0f;
    float level = -10.0f; // Default level in dB (must be <= 0.0)
    int api_idx = -1;
    int input_device_idx = -1;
    int output_device_idx = -1;
    int out_channels[2] = {-1, -1};
    int in_channels[2] = {-1, -1};
    uint32_t timestamp;
    bool only_save_measurement = false;

    // Helper method to print all parameters and their values with aligned colons
    void print() const {
        const int name_width = 20; // Matches length of "stimulus_output_file"

        std::cout << "\n";
        std::cout << "============ Program Parameters ============\n";
        std::cout << std::right; // Ensure right-alignment
        
        std::cout << std::setw(name_width) << "base file name"       << " : " << basefilename << "\n";
        std::cout << std::setw(name_width) << "sample rate"          << " : " << sample_rate << " Hz\n";
        std::cout << std::setw(name_width) << "chirp rate"           << " : " << chirp_rate << " seconds per decade Hz\n";
        std::cout << std::setw(name_width) << "start freq"           << " : " << start_freq << " Hz\n";
        std::cout << std::setw(name_width) << "repetitions"          << " : " << repetitions << "\n";
        std::cout << std::setw(name_width) << "alignment window"     << " : " << alignment_window << " milliseconds\n";
        std::cout << std::setw(name_width) << "warm up delay"        << " : " << warm_up_delay << " seconds\n";
        std::cout << std::setw(name_width) << "room decay time"      << " : " << rt60 << " seconds\n";
        std::cout << std::setw(name_width) << "mean wait"            << " : " << mean_wait << " seconds\n";
        std::cout << std::setw(name_width) << "level"                << " : " << level << " dB\n";
        std::cout << std::setw(name_width) << "api idx"              << " : " << api_idx << "\n";
        std::cout << std::setw(name_width) << "input device idx"     << " : " << input_device_idx << "\n";
        std::cout << std::setw(name_width) << "input channels"       << " : [" << in_channels[0] << ", " << in_channels[1] << "]\n";
        std::cout << std::setw(name_width) << "output device idx"    << " : " << output_device_idx << "\n";
        std::cout << std::setw(name_width) << "output channels"      << " : [" << out_channels[0] << ", " << out_channels[1] << "]\n";
        std::cout << std::setw(name_width) << "min_outfiles"         << " : [" << std::boolalpha << only_save_measurement << std::noboolalpha << "]\n";
        
        std::cout << "============================================\n";
        std::cout << "\n";
    }
};



// Display Usage / Help Information
void show_help_info(const char* prog_name) {
    std::cout << "Acoustic Testing Measurement Tool\n\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --basefilename, -b <file>           base file name for output files (default: results)\n"
              << "  --level, -l <float>                 RMS stimulus level in dB <= 0.0 (default: -10.0)\n"
              << "  --sample_rate, -r <int>             Sampling rate in Hz (default: 48000)\n"
              << "  --chirp_rate, -c <float>            Chirp rate in seconds per decade of frequency (default: 2.0)\n"
              << "  --start_freq, -s <float>            Start frequency in Hz (default: 1.0)\n"
              << "  --repetitions, -n <int>             Number of measurement runs (default: 1)\n"
              << "  --alignment_window <float>          Cross-Correlation Alignment Window in milliseconds (default: 5.0)\n"
              << "  --warm_up_delay <float>             Length of DAC warm-up delay in seconds (default: 3.0)\n"
              << "  --rt60 <float>                      Room decay time in seconds (default: 3.0)\n"
              << "  --mean_wait, -w <float>             Average pause between runs in seconds (default: 3.0)\n"
              << "  --api_idx, -a <int>                 PortAudio host API index\n"
              << "  --input_device_idx, -i <int>        PortAudio hardware input device index\n"
              << "  --output_device_idx, -o <int>       PortAudio hardware output device index\n"
              << "  --input_channels, -I <id1> <id2>    Two physical input channel indices\n"
              << "  --output_channels, -O <id1> <id2>   Two physical output channel indices\n"
              << "  --min_outfiles, -m                  Do not save the Farina inverse filter or Stimulus.\n"
              << "  --help, -h                          Display this help message and exit\n\n"
              << "NOTES:\n"
              << "    Both output channels produce the same stimulus.\n"
              << "    The first input channel records the stimulus signal.\n"
              << "    The second input channel records the microphone signal.\n"
              << "    Command line parameters may be provided in any order.\n\n";
}



// Helper to validate and convert an integer
// minAllowed allows us to enforce either > 0 (strictly positive) or >= 0 (zero or greater)
bool parseAndValidateInt(const std::string& argName, const char* strValue, int& outValue, int minAllowed) {
    try {
        size_t idx = 0;
        int val = std::stoi(strValue, &idx);
        if (strValue[idx] != '\0') {
            std::cerr << "Error: Trailing characters in value for " << argName << ": '" << strValue << "'\n";
            return false;
        }
        if (val < minAllowed) {
            std::cerr << "Error: " << argName << " must be an integer " 
                      << (minAllowed == 0 ? "zero or greater" : "greater than 0") 
                      << ". Got: " << val << "\n";
            return false;
        }
        outValue = val;
        return true;
    } catch (const std::exception&) {
        std::cerr << "Error: Invalid integer format for " << argName << ": '" << strValue << "'\n";
        return false;
    }
}

// Helper to validate and convert a float
bool parseAndValidateFloat(const std::string& argName, const char* strValue, float& outValue, bool mustBePositive = true) {
    try {
        size_t idx = 0;
        float val = std::stof(strValue, &idx);
        if (strValue[idx] != '\0') {
            std::cerr << "Error: Trailing characters in value for " << argName << ": '" << strValue << "'\n";
            return false;
        }
        if (mustBePositive && val <= 0.0f) {
            std::cerr << "Error: " << argName << " must be a positive number greater than 0.0. Got: " << val << "\n";
            return false;
        }
        outValue = val;
        return true;
    } catch (const std::exception&) {
        std::cerr << "Error: Invalid numeric format for " << argName << ": '" << strValue << "'\n";
        return false;
    }
}

AppArgs parse_arguments(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help_info(argv[0]);
            std::exit(0);
        }
        // --- String outputs ---
        else if (arg == "--basefilename" || arg == "-b") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            args.basefilename = argv[++i];
        }
        
        // --- Strictly positive integers (> 0) ---
        else if (arg == "--sample_rate" || arg == "-r") { // Fixed -h conflict with --help
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.sample_rate, 1)) std::exit(1);
        }
        else if (arg == "--repetitions" || arg == "-n") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.repetitions, 1)) std::exit(1);
        }
        
        // --- Strictly positive floats (> 0.0) ---
        else if (arg == "--chirp_rate" || arg == "-c") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.chirp_rate, true)) std::exit(1);
        }
        else if (arg == "--start_freq" || arg == "-s") { // Fixed malformed string literal syntax
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.start_freq, true)) std::exit(1);
        }
        else if (arg == "--alignment_window") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.alignment_window, true)) std::exit(1);
        }
        else if (arg == "--warm_up_delay") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.warm_up_delay, true)) std::exit(1);
        }
        else if (arg == "--rt60") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.rt60, true)) std::exit(1);
        }
        else if (arg == "--mean_wait" || arg == "-w") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.mean_wait, true)) std::exit(1);
        }
        
        // --- Integers zero or greater (>= 0) ---
        else if (arg == "--api_idx" || arg == "-a") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.api_idx, 0)) std::exit(1);
        }
        else if (arg == "--input_device_idx" || arg == "-i") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.input_device_idx, 0)) std::exit(1);
        }
        else if (arg == "--output_device_idx" || arg == "-o") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.output_device_idx, 0)) std::exit(1);
        }
        else if (arg == "--input_channels" || arg == "-I") {
            if (i + 2 >= argc) { std::cerr << "Error: --input_channels requires 2 integer values.\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.in_channels[0], 0)) std::exit(1);
            if (!parseAndValidateInt(arg, argv[++i], args.in_channels[1], 0)) std::exit(1);
        }
        else if (arg == "--output_channels" || arg == "-O") {
            if (i + 2 >= argc) { std::cerr << "Error: --output_channels requires 2 integer values.\n"; std::exit(1); }
            if (!parseAndValidateInt(arg, argv[++i], args.out_channels[0], 0)) std::exit(1);
            if (!parseAndValidateInt(arg, argv[++i], args.out_channels[1], 0)) std::exit(1);
        }

        // --- Decibel level parameter (<= 0.0) ---
        else if (arg == "--level") {
            if (i + 1 >= argc) { std::cerr << "Error: Missing value for " << arg << "\n"; std::exit(1); }
            if (!parseAndValidateFloat(arg, argv[++i], args.level, false)) std::exit(1);
            if (args.level > 0.0f) {
                std::cerr << "Warning: Level must be 0 dB or less. Restoring default value of -10.0 dB.\n";
                args.level = -10.0f;
            }
        }
        
        // --- Boolean Flag ---
        else if (arg == "--min_outfiles" || arg == "-m") {
            args.only_save_measurement = true;
        }

        // --- Catch-All Error Case ---
        else {
            std::cerr << "Error: Unknown or unrecognized command line option: '" << arg << "'\n";
            show_help_info(argv[0]);
            std::exit(1);
        }
    }
    args.end_freq = 0.99f * 0.5f * args.sample_rate;
    args.HF_window_startF = 0.975f * 0.5f * args.sample_rate;
    
    return args;
}




std::string sanitizeFilename(const std::string& input, const std::string& defaultName) {
    if (input.empty()) {
        return defaultName;
    }

    std::string result;
    result.reserve(input.size());

    bool lastWasSpace = false;

    for (char c : input) {
        unsigned char uc = static_cast<unsigned char>(c);
        
        // 1. Replace forbidden characters with '_'
        if (uc < 32 || c == '<' || c == '>' || c == ':' || c == '"' || 
            c == '/' || c == '\\' || c == '|' || c == '?' || c == '*') {
            result.push_back('_');
            lastWasSpace = false;
        } 
        // 2. Handle spaces: collapse multiple interior spaces into a single '_'
        else if (std::isspace(uc)) {
            // Only add an underscore if we aren't at the very beginning and the last char wasn't already a space/underscore converter
            if (!result.empty() && !lastWasSpace) {
                result.push_back('_');
            }
            lastWasSpace = true;
        } 
        else {
            result.push_back(c);
            lastWasSpace = false;
        }
    }

    // 3. Trim leading and trailing spaces, dots, and stray underscores
    size_t start = result.find_first_not_of(" ._");
    if (start == std::string::npos) {
        return defaultName; // String consisted only of spaces, dots, or invalid chars
    }
    size_t end = result.find_last_not_of(" ._");
    result = result.substr(start, end - start + 1);

    if (result.empty()) {
        return defaultName;
    }

    // 4. Handle Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    std::string upperResult = result;
    for (char& c : upperResult) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    
    size_t dotPos = upperResult.find('.');
    std::string baseName = (dotPos == std::string::npos) ? upperResult : upperResult.substr(0, dotPos);

    const std::unordered_set<std::string> reservedNames = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    };

    if (reservedNames.find(baseName) != reservedNames.end()) {
        result = "_" + result; // Prefix with underscore to neutralize reserved status
    }

    return result;
}



// Returns the current Unix timestamp (seconds since Jan 1, 1970) as a 32-bit unsigned int
uint32_t get_current_timestamp_32() {
    auto now = std::chrono::system_clock::now();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    return static_cast<uint32_t>(seconds.count());
}



/* Write 32-bit Float Mono WAV File, ending with a custom chunk containing several
   parameters used to peform the measurement:
      identifier <> variable
            f0sw <> args.start_freq);
            f1sw <> args.end_freq);
            fhfw <> args.HF_window_startF);
            rchr <> args.chirp_rate);
            levl <> args.level);
            date <> args.timestamp);
*/    
bool write_wav_file_plus_info(AppArgs& args, const std::string& filetype, const std::vector<float>& data) {
    // Construct filename using basefilename and the provided filetype suffix
    std::string filename = args.basefilename + "_" + filetype + ".wav";
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return false;
    }

    uint32_t sample_rate = static_cast<uint32_t>(args.sample_rate);
    uint32_t num_samples = static_cast<uint32_t>(data.size());
    uint32_t data_bytes = num_samples * sizeof(float);

    // Custom chunk payload size: 7 fields, each consisting of a 4-byte tag + 4-byte value = 56 bytes
    uint32_t xdat_payload_size = 56; 

    // RIFF chunk size = Total file size - 8 bytes (excludes 'RIFF' tag and 4-byte size field itself)
    // Structure: 4 (WAVE) + 24 (fmt chunk) + (8 + data_bytes) (data chunk) + (8 + xdat_payload_size) (xDAT chunk)
    uint32_t riff_chunk_size = 4 + 24 + (8 + data_bytes) + (8 + xdat_payload_size);

    // 1. Write RIFF Header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
    file.write("WAVE", 4);

    // 2. Write fmt Chunk (PCM IEEE Float = format 3, Mono = 1 channel, 32-bit)
    file.write("fmt ", 4);
    uint32_t fmt_chunk_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
    
    uint16_t audio_format = 3; // IEEE float
    uint16_t num_channels = 1; // Single channel
    uint32_t byte_rate = sample_rate * num_channels * sizeof(float);
    uint16_t block_align = num_channels * sizeof(float);
    uint16_t bits_per_sample = 32;

    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // 3. Write data Chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_bytes), 4);
    if (num_samples > 0) {
        file.write(reinterpret_cast<const char*>(data.data()), data_bytes);
    }

    // 4. Write Custom "xDAT" Chunk
    file.write("xDAT", 4);
    file.write(reinterpret_cast<const char*>(&xdat_payload_size), 4);

    // Helper lambda to write a 4-byte tag followed by its 4-byte value
    auto write_field = [&](const char tag[4], float val) {
        file.write(tag, 4);
        file.write(reinterpret_cast<const char*>(&val), sizeof(float));
    };

    auto write_field_u32 = [&](const char tag[4], uint32_t val) {
        file.write(tag, 4);
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
    };

    // Write parameters in the requested mapping
    write_field("f0sw", args.start_freq);
    write_field("f1sw", args.end_freq);
    write_field("fhfw", args.HF_window_startF);
    write_field("rchr", args.chirp_rate);
    write_field("levl", args.level);
    write_field_u32("date", args.timestamp);
    write_field_u32("reps", static_cast<uint32_t>(args.repetitions));

    file.close();
    return file.good();
}



// Hardware Device Management
bool check_host_api_and_device(AppArgs& args) {
    std::cout << "\nInitializing and Validating the Audio Device...\n";

    // Silence terminal output
    #if defined(_WIN32)
    int saved_stderr = _dup(_fileno(stderr));
    int devnull;
    if (_sopen_s(&devnull, "NUL", _O_WRONLY, _SH_DENYNO, 0) == 0) {
        _dup2(devnull, _fileno(stderr));
        _close(devnull);
    }
    #else
    int saved_stderr = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_RDWR);
    if (devnull != -1) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    #endif

    // Run the PortAudio code
    Pa_Initialize();
    int num_apis = Pa_GetHostApiCount();

    // Restore terminal output
    #if defined(_WIN32)
    _dup2(saved_stderr, _fileno(stderr));
    _close(saved_stderr);
    #else
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    #endif


    if (args.api_idx == -1) {
        std::cout << "\nFATAL ERROR: no API index selected via command line parameter --api_idx\n";
        return false;
    }

    const PaHostApiInfo* api_info = Pa_GetHostApiInfo(args.api_idx);
    std::cout << "\nDriver API: " << api_info->name << "\n";

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




// Log Chirp Generation including windowing, padding, and gain
std::vector<float> generate_log_chirp(const AppArgs& args) {
    // Promote core calculation variables to double precision from AppArgs
    double d_sample_rate = static_cast<double>(args.sample_rate);
    double d_start_freq  = static_cast<double>(args.start_freq);
    double d_end_freq    = static_cast<double>(args.end_freq);
    double d_chirp_rate  = static_cast<double>(args.chirp_rate);
    double d_level_dB    = static_cast<double>(args.level);
    double room_decay_time = static_cast<double>(args.rt60);

    // Basic input validation
    if (d_start_freq <= 0.0 || d_end_freq <= d_start_freq || d_chirp_rate <= 0.0 || args.sample_rate <= 0) {
        throw std::invalid_argument("Invalid parameters for chirp generation.");
    }

    double total_decades = std::log10(d_end_freq / d_start_freq);
    double duration = d_chirp_rate * total_decades;
    int num_samples = static_cast<int>(d_sample_rate * duration);

    std::vector<float> stimulus(num_samples);

    // Convert decibels to a linear amplitude multiplier in double precision
    // amplitude must be 1.0 or less.
    double amplitude = std::min(1.0, std::pow(10.0, d_level_dB / 20.0));

    // Precompute constant terms in double precision outside the loop
    const double ln_ratio = std::log(d_end_freq / d_start_freq);
    const double phase_factor = d_start_freq * duration / ln_ratio;
    const double ratio = d_end_freq / d_start_freq;
    const double two_pi = 2.0 * M_PI;

    // Generate Logarithmic Chirp in Double Precision & Cast to Float
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / d_sample_rate;
        double phase = two_pi * phase_factor * (std::pow(ratio, t / duration) - 1.0);
        
        // Cast the high-precision double result down to float upon storage
        stimulus[i] = static_cast<float>(amplitude * std::sin(phase));
    }
 
    // Generate half-Hann windows:
    // Calculate taper for low frequency (start) end of chirp:
    //   Define taper width in decades of frequency; 0.3 ~ one octave
    const float width_in_decades = 0.3f;  
    int start_length = static_cast<int>(args.sample_rate * (duration * (width_in_decades / total_decades)));    
    // Taper width for high frequency end of chirp is fixed at 220 samples. 
    //   This corresponds to 5 ms at a 44.1 kHz sample rate. In terms of
    //   frequency, it is approximately 0.975 times the Nyquist frequency 
    //   regardless of sample rate. 
    int end_length = 220;

    // Apply Tapers:
    for (int i = 0; i < start_length; ++i) {
        double hann = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / start_length));
        stimulus[i] *= static_cast<float>(hann);
    }
    for (int i = 0; i < end_length; ++i) {
        double hann = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / end_length));
        stimulus[num_samples - end_length + i] *= static_cast<float>(1.0 - hann);
    }

    // Add zero padding at the end equal to the room_decay_time + 0.5 sec
    int end_pad_samples = static_cast<int>(args.sample_rate * (0.5 + room_decay_time));
    stimulus.insert(stimulus.end(), end_pad_samples, 0.0f);

    return stimulus;
}




// Generate the analytical Farina inverse filter with an integrated high-
//    frequency fade-out taper to suppress ringing near the Nyquist frequency

std::vector<float> generate_Farina_inverse_filter(const AppArgs& args) {
    // 1. Setup core duration and sampling variables (ignoring end padding)
    double d_sample_rate = static_cast<double>(args.sample_rate);
    double d_start_freq  = static_cast<double>(args.start_freq);
    double d_end_freq    = static_cast<double>(args.end_freq);
    double total_decades = std::log10(d_end_freq / d_start_freq);
    double duration      = static_cast<double>(args.chirp_rate) * total_decades;
    int num_samples      = static_cast<int>(d_sample_rate * duration);

    std::vector<float> final_inv_filter(num_samples);

    double ln_ratio = std::log(d_end_freq / d_start_freq);
    double phase_factor = d_start_freq * duration / ln_ratio;
    double ratio = d_end_freq / d_start_freq;
    double two_pi = 2.0 * M_PI;
    double sweep_rate_L = duration / ln_ratio;

    // 2. Generate the exact sweep at 0 dB (amplitude = 1.0)
    std::vector<double> ideal_sweep(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / d_sample_rate;
        double phase = two_pi * phase_factor * (std::pow(ratio, t / duration) - 1.0);
        ideal_sweep[i] = std::sin(phase);
    }

    // 3. Apply the EXACT SAME fades as the stimulus generator
    const float width_in_decades = 0.3f;  
    int start_length = static_cast<int>(d_sample_rate * (duration * (width_in_decades / total_decades)));    
    int end_length = 220;

    for (int i = 0; i < start_length; ++i) {
        double hann = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / start_length));
        ideal_sweep[i] *= hann;
    }
    for (int i = 0; i < end_length; ++i) {
        double hann = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / end_length));
        ideal_sweep[num_samples - end_length + i] *= (1.0 - hann);
    }

    // 4. Amplitude scalar adjusted for discrete sample-rate normalization
    double normalization_scaler = (d_start_freq / sweep_rate_L) / d_sample_rate;

    // 5. Construct Farina Inverse Filter (Reverse, apply exp(-t/L), and scale)
    for (int i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / d_sample_rate;
        int rev_idx = num_samples - 1 - i;
        
        double envelope = std::exp(-t / sweep_rate_L);
        final_inv_filter[i] = static_cast<float>(ideal_sweep[rev_idx] * envelope * normalization_scaler);
    }

    return final_inv_filter;
}



// Fractional delay filter with integer shift separation and zero-padding
std::vector<float> fractional_delay_filter(const std::vector<float>& input, float total_shift) {
    size_t size = input.size();
    std::vector<float> output(size, 0.0f);

    if (size == 0) {
        return output;
    }

    int integer_shift = static_cast<int>(std::round(total_shift));
    float fractional_shift = total_shift - static_cast<float>(integer_shift);

    const int num_taps = 31; // Must be odd
    const int half_taps = num_taps / 2;

    for (size_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int t = -half_taps; t <= half_taps; ++t) {
            long long sample_idx = static_cast<long long>(i) - t - integer_shift;

            float x = static_cast<float>(t) - fractional_shift;
            float sinc_val;
            if (std::abs(x) < 1e-5f) {
                sinc_val = 1.0f;
            } else {
                float pi_x = static_cast<float>(M_PI) * x;
                sinc_val = std::sin(pi_x) / pi_x;
            }

            float normalized_n = static_cast<float>(t + half_taps) / static_cast<float>(num_taps - 1);
            float hann_window = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * normalized_n));

            float coeff = sinc_val * hann_window;

            float sample_val = 0.0f;
            if (sample_idx >= 0 && sample_idx < static_cast<long long>(size)) {
                sample_val = input[sample_idx];
            }

            sum += sample_val * coeff;
            weight_sum += coeff;
        }

        if (std::abs(weight_sum) > 1e-5f) {
            output[i] = sum / weight_sum;
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}



#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <iostream>

// Assuming DataContainer and pocketfft are defined in your environment
DataContainer perform_QA_and_time_alignment(
    const DataContainer& raw_data,
    float sample_rate,
    float alignment_window_msec) 
{
    const auto& ref_history = raw_data.stimulus_data;
    const auto& mic_history = raw_data.measurement_data;
    
    size_t num_runs = mic_history.size();
    if (num_runs == 0 || ref_history.size() != num_runs) return {};
    size_t signal_size = mic_history[0].size();
    
    // Convert time window to number of samples based on sample rate
    int max_lag = static_cast<int>(std::round((alignment_window_msec / 1000.0f) * sample_rate));

    // Determine efficient FFT size for linear cross-correlation (next power of 2 >= 2*N - 1)
    size_t fft_size = 1;
    size_t min_fft_size = 2 * signal_size - 1;
    while (fft_size < min_fft_size) {
        fft_size *= 2;
    }
    size_t num_freq_bins = fft_size / 2 + 1;
    float fft_norm = 1.0f / static_cast<float>(fft_size);

    pocketfft::shape_t shape_fwd = { fft_size };
    pocketfft::stride_t stride_in_fwd = { sizeof(float) };
    pocketfft::stride_t stride_out_fwd = { sizeof(std::complex<float>) };
    pocketfft::shape_t axes = { 0 };

    pocketfft::shape_t shape_bwd = { fft_size };
    pocketfft::stride_t stride_in_bwd = { sizeof(std::complex<float>) };
    pocketfft::stride_t stride_out_bwd = { sizeof(float) };

    // --- AUTOMATIC CONSENSUS ANCHOR SELECTION ---
    size_t best_anchor_idx = 0;
    if (num_runs > 1) {
        std::cout << "\n--- Evaluating Measurement Consistency & Selecting Consensus Anchor ---\n";
        
        std::vector<float> anchor_scores(num_runs, 0.0f);

        // Precompute unscaled FFTs (fct = 1.0f) and energy prefix sums for all measurement runs
        std::vector<std::vector<std::complex<float>>> all_MIC_f(num_runs, std::vector<std::complex<float>>(num_freq_bins));
        std::vector<std::vector<float>> all_pref_2(num_runs, std::vector<float>(signal_size + 1, 0.0f));

        for (size_t i = 0; i < num_runs; ++i) {
            std::vector<float> mic_padded(fft_size, 0.0f);
            for (size_t n = 0; n < signal_size; ++n) {
                float val = mic_history[i][n];
                mic_padded[n] = val;
                all_pref_2[i][n + 1] = all_pref_2[i][n] + val * val;
            }
            pocketfft::r2c(shape_fwd, stride_in_fwd, stride_out_fwd, axes, pocketfft::FORWARD,
                           mic_padded.data(), all_MIC_f[i].data(), 1.0f);
        }

        auto get_energy = [&](size_t run_idx, int start, int end) {
            if (start > end) return 0.0f;
            return all_pref_2[run_idx][end + 1] - all_pref_2[run_idx][start];
        };

        // Score each run based on its average correlation agreement with all other measurement runs
        for (size_t i = 0; i < num_runs; ++i) {
            float total_agreement = 0.0f;
            for (size_t j = 0; j < num_runs; ++j) {
                if (i == j) continue;

                // Cross-spectrum between measurement run i and run j (unscaled)
                std::vector<std::complex<float>> Z_f(num_freq_bins);
                for (size_t f = 0; f < num_freq_bins; ++f) {
                    Z_f[f] = std::conj(all_MIC_f[i][f]) * all_MIC_f[j][f];
                }

                std::vector<float> xcorr(fft_size);
                pocketfft::c2r(shape_bwd, stride_in_bwd, stride_out_bwd, axes, pocketfft::BACKWARD,
                               Z_f.data(), xcorr.data(), fft_norm); // Apply fft_norm here

                // Find peak NCC within max_lag for this pair
                float max_pair_ncc = -1.0f;
                for (int lag = -max_lag; lag <= max_lag; ++lag) {
                    size_t xcorr_idx = (lag >= 0) ? static_cast<size_t>(lag) : static_cast<size_t>(fft_size + lag);
                    float current_dot = xcorr[xcorr_idx];

                    int n_start = std::max(0, -lag);
                    int n_end = std::min(static_cast<int>(signal_size - 1), static_cast<int>(signal_size - 1 - lag));

                    float energy_i = get_energy(i, n_start, n_end);
                    float energy_j = get_energy(j, n_start + lag, n_end + lag);

                    float denom = std::sqrt(energy_i * energy_j);
                    float ncc = (denom > 1e-5f) ? (current_dot / denom) : 0.0f;
                    if (ncc > max_pair_ncc) max_pair_ncc = ncc;
                }
                total_agreement += max_pair_ncc;
            }
            anchor_scores[i] = total_agreement / static_cast<float>(num_runs - 1);
        }

        // Select the measurement run with the highest average agreement score as our robust anchor
        auto max_it = std::max_element(anchor_scores.begin(), anchor_scores.end());
        best_anchor_idx = std::distance(anchor_scores.begin(), max_it);
        
        std::cout << "  --> Selected MEASUREMENT RUN " << best_anchor_idx + 1 
                  << " as the consensus reference anchor (Avg Agreement Score: " << *max_it << ").\n\n";
    }

    // --- ALIGNMENT & QUALITY CONTROL PHASE ---
    DataContainer output;
    output.stimulus_data.reserve(num_runs);
    output.measurement_data.reserve(num_runs);

    // The winning consensus run is always accepted first as our baseline
    output.stimulus_data.push_back(ref_history[best_anchor_idx]);
    output.measurement_data.push_back(mic_history[best_anchor_idx]);

    if (num_runs > 1) {
        std::cout << "--- Performing FFT-Optimized Alignment & Quality Control ---\n";
        std::cout << "Searching within a window of +/- " << alignment_window_msec 
            << " ms (" << max_lag << " samples at " << sample_rate 
            << " Hz) for the best correlation with MEASUREMENT RUN " << best_anchor_idx + 1 << ":\n\n";
    }

    static constexpr float min_ncc_threshold = 0.95f;
    int lag_threshold = static_cast<int>(0.95f * max_lag);

    // Precompute forward FFT (unscaled) and energy prefix sums for the chosen anchor
    std::vector<float> anchor_mic_padded(fft_size, 0.0f);
    for (size_t n = 0; n < signal_size; ++n) {
        anchor_mic_padded[n] = mic_history[best_anchor_idx][n];
    }
    std::vector<std::complex<float>> MIC_anchor_f(num_freq_bins);
    pocketfft::r2c(shape_fwd, stride_in_fwd, stride_out_fwd, axes, pocketfft::FORWARD,
                   anchor_mic_padded.data(), MIC_anchor_f.data(), 1.0f);

    std::vector<float> pref_anchor_2(signal_size + 1, 0.0f);
    for (size_t n = 0; n < signal_size; ++n) {
        float val = mic_history[best_anchor_idx][n];
        pref_anchor_2[n + 1] = pref_anchor_2[n] + val * val;
    }
    auto get_anchor_energy = [&](int start, int end) {
        if (start > end) return 0.0f;
        return pref_anchor_2[end + 1] - pref_anchor_2[start];
    };

    // Loop through all other runs and align/filter them against the robust measurement anchor
    for (size_t i = 0; i < num_runs; ++i) {
        if (i == best_anchor_idx) continue;

        std::vector<float> pref_mic_i_2(signal_size + 1, 0.0f);
        for (size_t n = 0; n < signal_size; ++n) {
            float val = mic_history[i][n];
            pref_mic_i_2[n + 1] = pref_mic_i_2[n] + val * val;
        }
        auto get_mic_i_energy = [&](int start, int end) {
            if (start > end) return 0.0f;
            return pref_mic_i_2[end + 1] - pref_mic_i_2[start];
        };

        std::vector<float> mic_i_padded(fft_size, 0.0f);
        for (size_t n = 0; n < signal_size; ++n) {
            mic_i_padded[n] = mic_history[i][n];
        }
        std::vector<std::complex<float>> MIC_I_f(num_freq_bins);
        pocketfft::r2c(shape_fwd, stride_in_fwd, stride_out_fwd, axes, pocketfft::FORWARD,
                       mic_i_padded.data(), MIC_I_f.data(), 1.0f);

        // Multiply cross-spectrum (unscaled)
        std::vector<std::complex<float>> Z_f(num_freq_bins);
        for (size_t f = 0; f < num_freq_bins; ++f) {
            Z_f[f] = std::conj(MIC_anchor_f[f]) * MIC_I_f[f];
        }

        // Inverse FFT with proper normalization
        std::vector<float> xcorr(fft_size);
        pocketfft::c2r(shape_bwd, stride_in_bwd, stride_out_bwd, axes, pocketfft::BACKWARD,
                       Z_f.data(), xcorr.data(), fft_norm);

        int range_size = 2 * max_lag + 1;
        std::vector<float> corr_scores(range_size, -1e20f);
        float max_corr = -1e20f;
        int best_lag = 0;

        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            size_t xcorr_idx = (lag >= 0) ? static_cast<size_t>(lag) : static_cast<size_t>(fft_size + lag);
            float current_dot = xcorr[xcorr_idx];

            int n_start = std::max(0, -lag);
            int n_end = std::min(static_cast<int>(signal_size - 1), static_cast<int>(signal_size - 1 - lag));

            float anchor_energy = get_anchor_energy(n_start, n_end);
            float mic_i_energy = get_mic_i_energy(n_start + lag, n_end + lag);

            float denominator = std::sqrt(anchor_energy * mic_i_energy);
            float ncc = (denominator > 1e-5f) ? (current_dot / denominator) : 0.0f;

            int score_idx = lag + max_lag;
            corr_scores[score_idx] = ncc;

            if (ncc > max_corr) {
                max_corr = ncc;
                best_lag = lag;
            }
        }

        // Refine lag to sub-sample precision using Parabolic Interpolation (with bounds protection)
        float refined_lag = static_cast<float>(best_lag);
        int best_idx = best_lag + max_lag;

        if (best_lag > -max_lag && best_lag < max_lag) {
            if (best_idx > 0 && best_idx < range_size - 1) {
                float y_minus = corr_scores[best_idx - 1];
                float y_zero  = corr_scores[best_idx];
                float y_plus  = corr_scores[best_idx + 1];

                float denominator = (y_minus - 2.0f * y_zero + y_plus);
                if (std::abs(denominator) > 1e-5f) {
                    float delta = 0.5f * (y_minus - y_plus) / denominator;
                    refined_lag += delta;
                }
            }
        }

        std::cout << "RUN " << i + 1 << " has a timing offset of " << refined_lag 
                  << " samples and a quality score of " << max_corr << ".\n";
        
        std::vector<float> shifted_ref;
        std::vector<float> shifted_mic;

        if (std::abs(refined_lag) > lag_threshold) {
            std::cout << "  --> The timing error exceeded the threshold. RUN REJECTED.\n";
            continue;
        } 
        else if (max_corr < min_ncc_threshold) {
            std::cout << "  --> The quality score is below the cutoff (" 
                      << min_ncc_threshold << "). RUN REJECTED (likely corrupted).\n";
            continue;
        } 
        else if (std::abs(refined_lag) < 0.2f) {
            std::cout << "  --> RUN ACCEPTED for further processing.\n";
            shifted_ref = ref_history[i];
            shifted_mic = mic_history[i];
        } 
        else {
            std::cout << "  --> Correcting the timing error for this run. RUN ACCEPTED for further processing.\n";
            shifted_ref = fractional_delay_filter(ref_history[i], -refined_lag);
            shifted_mic = fractional_delay_filter(mic_history[i], -refined_lag);
        }

        output.stimulus_data.push_back(shifted_ref);
        output.measurement_data.push_back(shifted_mic);
    }

    return output;
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
    
    enum Phase { WARMUP, PROBING, MEASURING, PAUSING, COMPLETE };
    Phase current_phase = WARMUP;
    
    // Probe phase data references
    const std::vector<float>& probe_stimulus;
    std::vector<float>& probe_ref_record;
    std::vector<float>& probe_mic_record;
    int probe_total_samples = 0;

    int current_rep = 0;
    int total_reps = 1;
    long long warmup_frames_remaining = 0;
    long long pause_frames_remaining = 0;
    int current_frame = 0;
};




// Generate a short Hann-windowed sinc pulse for latency probing
std::vector<float> generate_probe_signal(float sample_rate, float duration_ms = 1.0f, float dB_level = -6.0 ) {
    int num_samples = static_cast<int>(std::round((duration_ms / 1000.0f) * sample_rate));
    if (num_samples % 2 == 0) num_samples += 1; // Ensure odd length for symmetry
    
    std::vector<float> probe(num_samples, 0.0f);
    int center = num_samples / 2;
    float cutoff_freq = 0.25f; // Fraction of Nyquist
    const float amplitude = std::pow(10.0f, dB_level / 20.0f ); 

    for (int i = 0; i < num_samples; ++i) {
        float x = static_cast<float>(i - center);
        float sinc_val = 0.0f;
        
        if (std::abs(x) < 1e-5f) {
            sinc_val = 1.0f;
        } else {
            float pi_x = static_cast<float>(M_PI) * cutoff_freq * x;
            sinc_val = std::sin(pi_x) / pi_x;
        }

        // Hann window
        float normalized_n = static_cast<float>(i) / static_cast<float>(num_samples - 1);
        float hann = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * normalized_n));

        probe[i] = amplitude * sinc_val * hann;
    }

    return probe;
}



// Estimates the baseline hardware latency by cross-correlating the recorded probe
int estimate_baseline_delay(
    const std::vector<float>& recorded_probe, 
    const std::vector<float>& pure_probe,
    int max_expected_latency_samples = 96000) // e.g., max 1 second at 96kHz
{
    if (recorded_probe.empty() || pure_probe.empty()) {
        return 0;
    }

    size_t rec_size = recorded_probe.size();
    size_t probe_size = pure_probe.size();

    float max_corr = -1e20f;
    int best_lag = 0;

    // Search from index 0 up to the maximum expected hardware latency buffer size
    int search_limit = static_cast<int>(rec_size) - static_cast<int>(probe_size);
    search_limit = std::min(search_limit, max_expected_latency_samples);

    if (search_limit <= 0) {
        std::cerr << "Error: Recorded probe buffer is too short for the probe signal size.\n";
        return 0;
    }

    for (int lag = 0; lag <= search_limit; ++lag) {
        float current_dot = 0.0f;
        float probe_energy = 0.0f;
        float rec_energy = 0.0f;

        for (size_t n = 0; n < probe_size; ++n) {
            float val_probe = pure_probe[n];
            float val_rec = recorded_probe[n + lag];

            current_dot += val_probe * val_rec;
            probe_energy += val_probe * val_probe;
            rec_energy += val_rec * val_rec;
        }

        float denominator = std::sqrt(probe_energy * rec_energy);
        float ncc = (denominator > 1e-5f) ? (current_dot / denominator) : 0.0f;

        if (ncc > max_corr) {
            max_corr = ncc;
            best_lag = lag;
        }
    }

    std::cout << "Probe cross-correlation peak found at index: " << best_lag 
              << " (NCC score: " << max_corr << ").\n";

    return best_lag;
}





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
                if (in) {
                    volatile float dummy_ref = in[i * ctx->pa_in_channels + ctx->ch_in0];
                    volatile float dummy_mic = in[i * ctx->pa_in_channels + ctx->ch_in1];
                    (void)dummy_ref;
                    (void)dummy_mic;
                }
                if (ctx->warmup_frames_remaining > 0) {
                    ctx->warmup_frames_remaining--;
                }
                break;

            case MeasurementContext::PROBING:
                if (ctx->current_frame < ctx->probe_total_samples) {
                    out_sample = ctx->probe_stimulus[ctx->current_frame];
                } else {
                    out_sample = 0.0f; // Padding for tail recording
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
                if (in) {
                    volatile float dummy_ref = in[i * ctx->pa_in_channels + ctx->ch_in0];
                    volatile float dummy_mic = in[i * ctx->pa_in_channels + ctx->ch_in1];
                    (void)dummy_ref;
                    (void)dummy_mic;
                }
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

        // Read input channels based strictly on the phase active at this specific frame
        if (ctx->current_phase == MeasurementContext::PROBING) {
            size_t record_idx = static_cast<size_t>(ctx->current_frame);
            if (record_idx < ctx->probe_ref_record.size() && in) {
                ctx->probe_ref_record[record_idx] = in[i * ctx->pa_in_channels + ctx->ch_in0];
                ctx->probe_mic_record[record_idx] = in[i * ctx->pa_in_channels + ctx->ch_in1];
            }
            ctx->current_frame++;

            // If probe recording finishes mid-buffer, zero out remainder of buffer and transition cleanly
            if (ctx->current_frame >= static_cast<int>(ctx->probe_ref_record.size())) {
                ctx->current_phase = MeasurementContext::MEASURING;
                ctx->current_frame = 0;
                ctx->current_rep = 0;
                break; // Exit the loop early so remaining buffer frames don't bleed into MEASURING
            }
        }
        else if (ctx->current_phase == MeasurementContext::MEASURING) {
            if (ctx->current_frame < ctx->total_samples && in) {
                ctx->ref_history[ctx->current_rep][ctx->current_frame] = in[i * ctx->pa_in_channels + ctx->ch_in0];
                ctx->mic_history[ctx->current_rep][ctx->current_frame] = in[i * ctx->pa_in_channels + ctx->ch_in1];
            }
            ctx->current_frame++;

            if (ctx->current_frame >= ctx->total_samples) {
                ctx->current_rep++;
                ctx->current_frame = 0;
                
                if (ctx->current_rep >= ctx->total_reps) {
                    ctx->current_phase = MeasurementContext::COMPLETE;
                } else {
                    ctx->current_phase = MeasurementContext::PAUSING;
                }
                break; // Exit early on phase completion
            }
        }
    }

    return (ctx->current_phase == MeasurementContext::COMPLETE) ? paComplete : paContinue;
}


DataContainer perform_acoustic_measurement(AppArgs& args, const std::vector<float>& stimulus) {
    int total_samples = stimulus.size();
    
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

    std::vector<std::vector<float>> ref_history(args.repetitions, std::vector<float>(total_samples, 0.0f));
    std::vector<std::vector<float>> mic_history(args.repetitions, std::vector<float>(total_samples, 0.0f));

    // Generate probe signal & allocate recording buffers for latency calibration
    std::vector<float> pure_probe = generate_probe_signal(args.sample_rate, 5.0f, args.level);
    size_t probe_tail_padding = static_cast<size_t>(args.sample_rate * 0.05f); // 50ms tail
    size_t total_probe_record_size = pure_probe.size() + probe_tail_padding;
    
    std::vector<float> probe_ref_record(total_probe_record_size, 0.0f);
    std::vector<float> probe_mic_record(total_probe_record_size, 0.0f);

    MeasurementContext ctx{
        stimulus, ref_history, mic_history,
        total_samples, pa_in_channels, pa_out_channels,
        isAsio ? 0 : args.in_channels[0],
        isAsio ? 1 : args.in_channels[1],
        MeasurementContext::WARMUP,
        pure_probe, probe_ref_record, probe_mic_record, static_cast<int>(pure_probe.size()),
        0, args.repetitions,
        static_cast<long long>(args.warm_up_delay * args.sample_rate),
        0, 0
    };

    PaStreamParameters inputParameters{}, outputParameters{};
    inputParameters.device = args.input_device_idx;
    inputParameters.channelCount = pa_in_channels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = inputDevInfo ? inputDevInfo->defaultLowInputLatency : 0.05;

    outputParameters.device = args.output_device_idx;
    outputParameters.channelCount = pa_out_channels;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = outputDevInfo ? outputDevInfo->defaultLowOutputLatency : 0.05;

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

    std::cout << "\nStarting acoustic measurement session at " << args.sample_rate << " Hz...\n";
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

    // Duplex I/O Warm-up
    std::cout << "Warming up the duplex hardware stream for " << args.warm_up_delay << " seconds...\n";
    while (ctx.warmup_frames_remaining > 0 && Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(20);
    }

    // Get stimulus measurement I/O latency
    std::cout << "\nProbing baseline hardware latency...\n";
    ctx.current_frame = 0;
    ctx.current_phase = MeasurementContext::PROBING;

    while (ctx.current_phase == MeasurementContext::PROBING && Pa_IsStreamActive(stream) == 1) {
        Pa_Sleep(20);
    }

    const int max_expected_latency = static_cast<int>( 0.3f * args.sample_rate );
    int baseline_delay_samples = estimate_baseline_delay(probe_ref_record, pure_probe, max_expected_latency);

    std::cout << "    An I/O Delay of " 
        << 1000.0f * baseline_delay_samples / args.sample_rate 
        << " milliseconds (" << baseline_delay_samples << " samples) was found.\n\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());

    // Measurement Repetition Loop with Inter-measurement Pauses
    std::cout << "Begin Collecting Data:\n\n";
    float max_variation = args.mean_wait * 0.10f;
    float actual_wait;

    for (int r = 0; r < args.repetitions; ++r) {
        std::uniform_real_distribution<float> dis(-max_variation, max_variation);
        actual_wait = args.mean_wait + dis(gen);
        std::cout << "Pausing " << actual_wait << " seconds before collecting data.\n";
        long long pause_samples = static_cast<long long>(actual_wait * args.sample_rate);
        ctx.pause_frames_remaining = pause_samples;
        ctx.current_phase = MeasurementContext::PAUSING;
        while (ctx.pause_frames_remaining > 0 && Pa_IsStreamActive(stream) == 1) {
            Pa_Sleep(20);
        }

        std::cout << "Begin measurement #" << (r + 1) << " of " << args.repetitions << "...\n";
        ctx.current_rep = r;
        ctx.current_frame = 0;
        ctx.current_phase = MeasurementContext::MEASURING;
        while (ctx.current_phase == MeasurementContext::MEASURING && Pa_IsStreamActive(stream) == 1) {
            Pa_Sleep(20);
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);

    // Return the raw histories wrapped inside DataContainer
    return {ref_history, mic_history};
}



// Computes a single noise-reduced averaged reference signal from all valid aligned runs.
std::vector<float> compute_averaged_stimulus(const DataContainer& alignment_data)
{
    const auto& ref_history = alignment_data.stimulus_data;
    size_t num_runs = ref_history.size();
    if (num_runs == 0) return {};

    size_t signal_size = ref_history[0].size();
    std::vector<float> averaged_ref(signal_size, 0.0f);

    // Sum all aligned reference runs
    for (size_t i = 0; i < num_runs; ++i) {
        for (size_t n = 0; n < signal_size; ++n) {
            averaged_ref[n] += ref_history[i][n];
        }
    }

    // Divide by the number of valid runs to complete the average
    float inv_num_runs = 1.0f / static_cast<float>(num_runs);
    for (size_t n = 0; n < signal_size; ++n) {
        averaged_ref[n] *= inv_num_runs;
    }

    return averaged_ref;
}




// Computes a single noise-reduced averaged microphone signal from all valid aligned runs.
std::vector<float> compute_averaged_measurement(const DataContainer& alignment_data)
{
    const auto& mic_history = alignment_data.measurement_data;
    size_t num_runs = mic_history.size();
    if (num_runs == 0) return {};

    size_t signal_size = mic_history[0].size();
    std::vector<float> averaged_mic(signal_size, 0.0f);

    // Sum all aligned microphone runs
    for (size_t i = 0; i < num_runs; ++i) {
        for (size_t n = 0; n < signal_size; ++n) {
            averaged_mic[n] += mic_history[i][n];
        }
    }

    // Divide by the number of valid runs to complete the average
    float inv_num_runs = 1.0f / static_cast<float>(num_runs);
    for (size_t n = 0; n < signal_size; ++n) {
        averaged_mic[n] *= inv_num_runs;
    }

    return averaged_mic;
}





// Computes the time-domain impulse response using the true H1 estimator
// across multiple pre-aligned, un-averaged measurement datasets.
std::vector<float> compute_true_h1_time_domain(const DataContainer& alignment_data)
{
    // Total number of measurement repetitions/trials
    size_t num_trials = alignment_data.stimulus_data.size();
    if (num_trials == 0) return {};

    // Length of a single time block
    size_t signal_length = alignment_data.stimulus_data[0].size();
    
    // For real-input FFTs, number of unique positive frequency bins is (N / 2) + 1
    size_t num_freq_bins = (signal_length / 2) + 1;

    // Allocate accumulation buffers for ensemble averaging
    std::vector<float> G_xx_sum(num_freq_bins, 0.0f);
    std::vector<std::complex<float>> G_xy_sum(num_freq_bins, 0.0f);

    // PocketFFT configurations for Real-to-Complex (r2c) forward transform
    pocketfft::shape_t shape_fwd = { signal_length };
    pocketfft::stride_t stride_in_fwd = { sizeof(float) };
    pocketfft::stride_t stride_out_fwd = { sizeof(std::complex<float>) };
    pocketfft::shape_t axes = { 0 };

    // Intermediate frequency buffers utilized inside the loop
    std::vector<std::complex<float>> REF_f(num_freq_bins);
    std::vector<std::complex<float>> MIC_f(num_freq_bins);

    // Scale factor to normalize forward PocketFFT values (1 / N)
    float fft_norm = 1.0f / static_cast<float>(signal_length);

    // 1. Core H1 Phase: Cross-correlate each trial BEFORE averaging
    for (size_t i = 0; i < num_trials; ++i) {
        // PocketFFT requires non-const vectors for its internal memory layout
        std::vector<float> ref_trial = alignment_data.stimulus_data[i];
        std::vector<float> mic_trial = alignment_data.measurement_data[i];

        // Execute Real-to-Complex Forward FFT for reference stimulus
        pocketfft::r2c(shape_fwd, stride_in_fwd, stride_out_fwd, axes, pocketfft::FORWARD, 
                       ref_trial.data(), REF_f.data(), fft_norm);

        // Execute Real-to-Complex Forward FFT for microphone response
        pocketfft::r2c(shape_fwd, stride_in_fwd, stride_out_fwd, axes, pocketfft::FORWARD, 
                       mic_trial.data(), MIC_f.data(), fft_norm);

        // Compute and accumulate auto- and cross-spectral densities
        for (size_t f = 0; f < num_freq_bins; ++f) {
            std::complex<float> REF_conj = std::conj(REF_f[f]);
            
            // Gxx = REF* * REF (Auto-spectrum, purely real energy)
            G_xx_sum[f] += std::real(REF_conj * REF_f[f]); 
            
            // Gxy = REF* * MIC (Cross-spectrum)
            // Uncorrelated background noise phase vectors point in random directions 
            // across trials and systematically cancel out inside this accumulator.
            G_xy_sum[f] += REF_conj * MIC_f[f];            
        }
    }

    // 2. Compute the final frequency-domain H1 matrix division
    // (Note: dividing both sums by num_trials cancels out mathematically, so we omit for performance)
    std::vector<std::complex<float>> H1_f(num_freq_bins);
    for (size_t f = 0; f < num_freq_bins; ++f) {
        float Gxx = G_xx_sum[f];
        std::complex<float> Gxy = G_xy_sum[f];

        // Guard against division-by-zero noise amplification in unexcited frequencies
        if (Gxx > 1e-15f) {
            H1_f[f] = Gxy / Gxx;
        } else {
            H1_f[f] = std::complex<float>(0.0f, 0.0f);
        }
    }

    // 3. Inverse FFT (Complex-to-Real) to generate the clean Time-Domain Impulse Response
    std::vector<float> impulse_response(signal_length);
    pocketfft::shape_t shape_bwd = { signal_length };
    pocketfft::stride_t stride_in_bwd = { sizeof(std::complex<float>) };
    pocketfft::stride_t stride_out_bwd = { sizeof(float) };

    pocketfft::c2r(shape_bwd, stride_in_bwd, stride_out_bwd, axes, pocketfft::BACKWARD, 
                   H1_f.data(), impulse_response.data(), 1.0f);

    return impulse_response;
}






int main(int argc, char* argv[]) {
    // Show usage/help info automatically if no arguments are provided
    if (argc == 1) {
        show_help_info(argv[0]);
        return 0;
    }

    AppArgs args = parse_arguments(argc, argv);

    //sanitize the basefilename
    args.basefilename = sanitizeFilename(args.basefilename, "results");

    // Print all program parameters to the screen
    args.print();

    if (!check_host_api_and_device(args)) {
        return -1;
    }

    // Generate a chirp to be played as the system stimulus
    std::vector<float> chirp = generate_log_chirp(args); 

    // Collect raw measurement data from hardware
    DataContainer raw_data = perform_acoustic_measurement(args, chirp);
    if (raw_data.stimulus_data.empty()) {
        std::cerr << "Error: Acoustic measurement session failed.\n";
        return -1;
    }

    // Perform time-alignment, consensus anchor selection, and quality control
    DataContainer aligned_data = perform_QA_and_time_alignment(
        raw_data, 
        args.sample_rate, 
        args.alignment_window
    );

    if (aligned_data.stimulus_data.empty()) {
        std::cerr << "Error: Time alignment and quality control failed (all runs rejected).\n";
        return -1;
    } else {
        std::cout << "\n" << aligned_data.stimulus_data.size() 
                  << " out of " << raw_data.stimulus_data.size() << " reps will now be averaged.\n";
    }

    // Average the stimulus and measurement data
    std::vector<float> averaged_ref = compute_averaged_stimulus(aligned_data);
    std::vector<float> averaged_mic = compute_averaged_measurement(aligned_data);

    // Compute time domain impulse response using H1 estimator 
    std::vector<float> h1_ir = compute_true_h1_time_domain(aligned_data);

    // Generate and store a timestamp
    args.timestamp = get_current_timestamp_32();

    std::cout << "\n--- Saving Data Files ---\n";
    std::string filetype;
    std::string filename;

    //Write measurement data as WAV file
    filetype = "m";
    filename = args.basefilename + filetype + ".wav";
    if (write_wav_file_plus_info(args, filetype, averaged_mic)) {
        std::cout << "Measurement data WAV file and custom xDAT metadata written to the file " << filename << "\n";
    } else {
        std::cerr << "Failure: An error occurred while attempting to write the measurement WAV file.\n";
        return -1; // Optional: exit with error code
    }

    //Write stimulus data as WAV file
    filetype = "s";
    filename = args.basefilename + filetype + ".wav";
    if (write_wav_file_plus_info(args, filetype, chirp)) {
        std::cout << "Stimulus WAV file and custom xDAT metadata written to the file " << filename << "\n";
    } else {
        std::cerr << "Failure: An error occurred while attempting to write the stimulus WAV file.\n";
        return -1; // Optional: exit with error code
    }

    //Write H1 time domain response to WAV file
    filetype = "h1";
    filename = args.basefilename + filetype + ".wav";
    if (write_wav_file_plus_info(args, filetype, h1_ir)) {
        std::cout << "Impulse response WAV file and custom xDAT metadata written to the file " << filename << "\n";
    } else {
        std::cerr << "Failure: An error occurred while attempting to write the impulse response WAV file.\n";
        return -1; // Optional: exit with error code
    }

    //Generate and save the Farina Inverse Filter as WAV file
    std::vector<float> inv_filter = generate_Farina_inverse_filter(args);
    filetype = "Fif";
    filename = args.basefilename + filetype + ".wav";
    if (write_wav_file_plus_info(args, filetype, inv_filter)) {
        std::cout << "Farina inverse filter WAV file and custom xDAT metadata written to the file " << filename << "\n";
    } else {
        std::cerr << "Failure: An error occurred while attempting to write the Farina inverse filter WAV file.\n";
        return -1; // Optional: exit with error code
    }

    std::cout << "\n------------ All tasks completed ------------ \n\n";

    Pa_Terminate();
    return 0;
}



