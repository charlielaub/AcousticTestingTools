cmake_minimum_required(VERSION 3.15)
project(AcousticTestingTools)

# Find PortAudio package
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

if(NOT PORTAUDIO_FOUND)
    # Fallback finder or manual configuration for Windows/platforms without pkg-config
    find_path(PORTAUDIO_INCLUDE_DIRS portaudio.h)
    find_library(PORTAUDIO_LIBRARIES NAMES portaudio portaudio_x86_64)
endif()

# Define Executables inside measurement_tools/
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})

add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})