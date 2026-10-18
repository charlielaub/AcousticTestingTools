cmake_minimum_required(VERSION 3.18)
project(AcousticTestingTools)

# --- Force C++17 globally ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED)

# -------------------------------------------------------------
# PortAudio & ASIO Discovery (Cross-Platform via vcpkg / PkgConfig)
# -------------------------------------------------------------
if(WIN32)
    message(STATUS "Windows detected. Looking for vcpkg-managed PortAudio and ASIO...")
    find_package(portaudio CONFIG REQUIRED)
else()
    # Linux / Non-Windows path via PkgConfig
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
    endif()

    if(NOT PORTAUDIO_FOUND)
        message(FATAL_ERROR "PortAudio dependency not found on this Linux system. Install libportaudio2 / libasound2-dev.")
    endif()
endif()

# -------------------------------------------------------------
# Program 1: DaDooRunRun
# -------------------------------------------------------------
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_link_libraries(DaDooRunRun PRIVATE Eigen3::Eigen)

if(WIN32)
    target_link_libraries(DaDooRunRun PRIVATE portaudio::portaudio)
    target_compile_definitions(DaDooRunRun PRIVATE PA_USE_ASIO=1)
else()
    target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})
endif()

# -------------------------------------------------------------
# Program 2: ProbeAudioDevs
# -------------------------------------------------------------
add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_link_libraries(ProbeAudioDevs PRIVATE Eigen3::Eigen)

if(WIN32)
    target_link_libraries(ProbeAudioDevs PRIVATE portaudio::portaudio)
    target_compile_definitions(ProbeAudioDevs PRIVATE PA_USE_ASIO=1)
else()
    target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})
endif()