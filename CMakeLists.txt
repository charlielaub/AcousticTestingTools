cmake_minimum_required(VERSION 3.18)
project(AcousticTestingTools)

# --- Force C++17 globally ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED)

# -------------------------------------------------------------
# PortAudio System Configurations
# -------------------------------------------------------------
if(WIN32)
    message(STATUS "Windows detected. Configuring automated audio dependencies...")

    # 1. Look for the local vcpkg-managed ASIO SDK paths automatically
    # When using vcpkg's toolchain, asiosdk is globally available out of the box.
    set(PA_USE_ASIO ON CACHE BOOL "Enable ASIO support" FORCE)
    
    # 2. Add the local PortAudio source directory directly into our compilation targets
    if(DEFINED PORTAUDIO_SOURCE_LOCAL_PATH)
        message(STATUS "Using pre-cloned PortAudio source from: ${PORTAUDIO_SOURCE_LOCAL_PATH}")
        add_subdirectory("${PORTAUDIO_SOURCE_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/portaudio_build")
    else()
        message(FATAL_ERROR "PORTAUDIO_SOURCE_LOCAL_PATH not set! Please pass -DPORTAUDIO_SOURCE_LOCAL_PATH=/path/to/source")
    endif()

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
    # Bind directly to the local project-compiled target and inject definition
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
    # Bind directly to the local project-compiled target and inject definition
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
