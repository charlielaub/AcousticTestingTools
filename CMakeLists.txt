cmake_minimum_required(VERSION 3.15)
project(AcousticTestingTools)

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED)

# Find PortAudio package
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

if(NOT PORTAUDIO_FOUND)
    find_package(portaudio CONFIG REQUIRED)
endif()

# Define Executables inside measurement_tools/
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_link_libraries(DaDooRunRun PRIVATE Eigen3::Eigen)
if(TARGET portaudio::portaudio)
    target_link_libraries(DaDooRunRun PRIVATE portaudio::portaudio)
else()
    target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})
endif()

add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_link_libraries(ProbeAudioDevs PRIVATE Eigen3::Eigen)
if(TARGET portaudio::portaudio)
    target_link_libraries(ProbeAudioDevs PRIVATE portaudio::portaudio)
else()
    target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})
endif()