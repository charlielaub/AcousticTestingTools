cmake_minimum_required(VERSION 3.15)
project(AcousticTestingTools)

find_package(PortAudio REQUIRED)

# Point CMake to the subfolder path for each program
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_link_libraries(DaDooRunRun PRIVATE PortAudio::PortAudio)

add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_link_libraries(ProbeAudioDevs PRIVATE PortAudio::PortAudio)