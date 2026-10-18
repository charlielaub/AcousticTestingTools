cmake_minimum_required(VERSION 3.18) # Upgraded from 3.15 to support FetchContent_MakeAvailable
project(AcousticTestingTools)

# --- Force C++17 globally ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED)

# -------------------------------------------------------------
# PortAudio System Configurations
# -------------------------------------------------------------
if(WIN32)
    # Windows GitHub CI Path: Build PortAudio from source with native ASIO support
    message(STATUS "Windows detected. Automating ASIO SDK and PortAudio source compilation...")

    # 1. Download a direct zip release from Steinberg's official GitHub distribution mirror
    FetchContent_Declare(
        asiosdk
        URL https://github.com
    )
    FetchContent_MakeAvailable(asiosdk)

    # Steinberg's repository nests the actual source code under a subfolder.
    # We point to that exact subfolder so PortAudio knows where to look:
    set(ASIO_SDK_PATH "${asiosdk_SOURCE_DIR}/asiosdk-2.3.3")

    # 2. Configure PortAudio to use the ASIO driver backend
    set(PA_USE_ASIO ON CACHE BOOL "Enable ASIO support" FORCE)
    
    # 3. Pull PortAudio source directly from the official stable branch
    FetchContent_Declare(
        portaudio
        GIT_REPOSITORY https://github.com
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(portaudio)

else()
    # Linux / Non-Windows path
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
    endif()

    if(PORTAUDIO_FOUND)
        message(STATUS "Using PortAudio via PkgConfig (Linux)")
    else()
        message(FATAL_ERROR "PortAudio dependency not found on this system.")
    endif()
endif()

# -------------------------------------------------------------
# Program 1: DaDooRunRun
# -------------------------------------------------------------
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_link_libraries(DaDooRunRun PRIVATE Eigen3::Eigen)

if(WIN32)
    # Bind directly to the local ASIO compiled target and inject definition
    target_link_libraries(DaDooRunRun PRIVATE portaudio)
    target_compile_definitions(DaDooRunRun PRIVATE PA_USE_ASIO=1)
else()
    # Fallback to standard targets/variables discovered on Linux
    if(TARGET portaudio_static)
        target_link_libraries(DaDooRunRun PRIVATE portaudio_static)
    elseif(TARGET portaudio::portaudio)
        target_link_libraries(DaDooRunRun PRIVATE portaudio::portaudio)
    elseif(TARGET portaudio)
        target_link_libraries(DaDooRunRun PRIVATE portaudio)
    else()
        target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
        target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})
    endif()
endif()

# -------------------------------------------------------------
# Program 2: ProbeAudioDevs
# -------------------------------------------------------------
add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_link_libraries(ProbeAudioDevs PRIVATE Eigen3::Eigen)

if(WIN32)
    # Bind directly to the local ASIO compiled target and inject definition
    target_link_libraries(ProbeAudioDevs PRIVATE portaudio)
    target_compile_definitions(ProbeAudioDevs PRIVATE PA_USE_ASIO=1)
else()
    # Fallback to standard targets/variables discovered on Linux
    if(TARGET portaudio_static)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio_static)
    elseif(TARGET portaudio::portaudio)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio::portaudio)
    elseif(TARGET portaudio)
        target_link_libraries(ProbeAudioDevs PRIVATE portaudio)
    else()
        target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
        target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})
    endif()
endif()
