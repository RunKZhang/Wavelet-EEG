cmake_minimum_required(VERSION 2.8)

project(wavelet_EEG)
set(CMAKE_CXX_STANDARD 17)

file(GLOB srclist "src/*.h" )
INCLUDE_DIRECTORIES(eigen)
INCLUDE_DIRECTORIES(neural_network_library)
INCLUDE_DIRECTORIES(autodiff)




add_executable(wavelet_EEG src/main.cpp ${srclist})
