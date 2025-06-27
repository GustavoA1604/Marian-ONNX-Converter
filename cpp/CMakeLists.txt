cmake_minimum_required(VERSION 3.10)
project(MarianONNXInference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_PATH}/lib)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

# Find required libraries
find_package(Threads REQUIRED)

# Try to find SentencePiece using pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SENTENCEPIECE sentencepiece)
endif()

# If pkg-config didn't find it, try to find it manually
if(NOT SENTENCEPIECE_FOUND)
    find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
        PATHS /usr/local/include /usr/include
        PATH_SUFFIXES sentencepiece
    )
    find_library(SENTENCEPIECE_LIBRARY sentencepiece
        PATHS /usr/local/lib /usr/lib
    )
    
    if(SENTENCEPIECE_INCLUDE_DIR AND SENTENCEPIECE_LIBRARY)
        set(SENTENCEPIECE_FOUND TRUE)
        set(SENTENCEPIECE_INCLUDE_DIRS ${SENTENCEPIECE_INCLUDE_DIR})
        set(SENTENCEPIECE_LIBRARIES ${SENTENCEPIECE_LIBRARY})
    endif()
endif()

# Check if SentencePiece was found
if(NOT SENTENCEPIECE_FOUND)
    message(FATAL_ERROR "SentencePiece not found. Please install SentencePiece development package.")
endif()

# Create executable
add_executable(main 
    main.cpp 
    TranslationModel.cpp
)

# Include directories (matching the -I flags from g++ command)
target_include_directories(main PRIVATE 
    /usr/local/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${SENTENCEPIECE_INCLUDE_DIRS}
)

# Compiler definitions (matching the -D flags from g++ command)
target_compile_definitions(main PRIVATE USE_SIMD)

# Compiler options (matching the optimization flags from g++ command)
target_compile_options(main PRIVATE
    -mavx2
    -mfma
    ${SENTENCEPIECE_CFLAGS_OTHER}
)

# Library directories (matching the -L flags from g++ command)
target_link_directories(main PRIVATE
    /usr/local/lib
    ${ONNXRUNTIME_LIB_DIR}
)

# Link libraries (matching the -l flags from g++ command)
target_link_libraries(main 
    ${SENTENCEPIECE_LIBRARIES}
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

# Set rpath to find ONNX Runtime library at runtime (matching the -Wl,-rpath flag)
set_target_properties(main PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    RUNTIME_OUTPUT_NAME main
)

# Print configuration information
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIB}")
message(STATUS "SentencePiece Include: ${SENTENCEPIECE_INCLUDE_DIRS}")
message(STATUS "SentencePiece Libraries: ${SENTENCEPIECE_LIBRARIES}")
message(STATUS "SIMD Support: Enabled (AVX2, FMA)") 