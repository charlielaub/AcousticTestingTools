cmake_minimum_required(VERSION 3.15)
project(AcousticTestingTools)

# --- Force C++17 globally ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 (Header-only library)
find_package(Eigen3 REQUIRED)

# Find PortAudio (Cross-platform robust finder)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)
endif()

if(PORTAUDIO_FOUND)
    # Linux / pkg-config path
    message(STATUS "Using PortAudio via PkgConfig (Linux)")
else()
    # Windows / vcpkg path
    message(STATUS "Using PortAudio via vcpkg (Windows)")
    find_package(portaudio CONFIG REQUIRED)
endif()

# --- Program 1: DaDooRunRun ---
add_executable(DaDooRunRun measurement_tools/DaDooRunRun.cpp)
target_link_libraries(DaDooRunRun PRIVATE Eigen3::Eigen)

if(TARGET portaudio::portaudio)
    target_link_libraries(DaDooRunRun PRIVATE portaudio::portaudio)
elseif(TARGET portaudio)
    target_link_libraries(DaDooRunRun PRIVATE portaudio)
    # Safety fallback: explicitly include vcpkg headers if the target misses them
    if(DEFINED PORTAUDIO_INCLUDE_DIRS)
        target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    endif()
else()
    target_include_directories(DaDooRunRun PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(DaDooRunRun PRIVATE ${PORTAUDIO_LIBRARIES})
endif()

# --- Program 2: ProbeAudioDevs ---
add_executable(ProbeAudioDevs measurement_tools/ProbeAudio.cpp)
target_link_libraries(ProbeAudioDevs PRIVATE Eigen3::Eigen)

if(TARGET portaudio::portaudio)
    target_link_libraries(ProbeAudioDevs PRIVATE portaudio::portaudio)
elseif(TARGET portaudio)
    target_link_libraries(ProbeAudioDevs PRIVATE portaudio)
    # Safety fallback: explicitly include vcpkg headers if the target misses them
    if(DEFINED PORTAUDIO_INCLUDE_DIRS)
        target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    endif()
else()
    target_include_directories(ProbeAudioDevs PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
    target_link_libraries(ProbeAudioDevs PRIVATE ${PORTAUDIO_LIBRARIES})
endif()