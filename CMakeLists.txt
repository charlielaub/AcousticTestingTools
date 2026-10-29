cmake_minimum_required(VERSION 3.18)
project(AcousticTestingTools)

# --- Force C++17 globally ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Include path for PocketFFT (header-only)
target_include_directories(DaDooRunRun PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIN32)
    # vcpkg provides the 'portaudio' or 'portaudio_static' target
    if(TARGET portaudio::portaudio)
        target_link_libraries(DaDooRunRun PRIVATE portaudio::portaudio)
    elseif(TARGET portaudio)
        target_link_libraries(DaDooRunRun PRIVATE portaudio)
    elseif(TARGET portaudio_static)
        target_link_libraries(DaDooRunRun PRIVATE portaudio_static)
    endif()
    target_compile_definitions(DaDooRunRun PRIVATE PA_USE_ASIO=1)
else()
    target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})
endif()

# -------------------------------------------------------------
# Program 2: ProbeAudioDevs
# -------------------------------------------------------------
add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)

if(WIN32)
    if(TARGET portaudio::portaudio)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio::portaudio)
    elseif(TARGET portaudio)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio)
    elseif(TARGET portaudio_static)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio_static)
    endif()
    target_compile_definitions(ProbeAudioDevs PRIVATE PA_USE_ASIO=1)
else()
    target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})
endif()