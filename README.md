# Wavelet-EEG
This repository is for my undergraduate thesis. I uses wavelet transform and a neural network library developed from scratch to classify EEG signals.

# Prerequisites
CMake 3.23.0: https://cmake.org/download/

This project is built by using CMake, so it is necessary to install CMake in your computer before compile the project.

Eigen 3.4.0 or above: https://eigen.tuxfamily.org/index.php?title=Main_Page

autodiff 0.6.7 or above: https://github.com/autodiff/autodiff

If you have installed the two libraries mentioned above, please put the folders of them under the root directory of this project.

# Files Explanation
*src* contains files about the implementation of wavelet transform for the analysis of the EEG signals. 
In this folder, *WT.h* is the main implementation of wavelet transform. *process_data.h* contains some preprocessing methods of the analysis.
*main.cpp* is the main entrance of the project.

*neural_network_library* contains the neural network library implemented from scratch and it can also found in my another repository:https://github.com/RunKZhang/CPP-Neural-Network-Library-From-Scratch.

*data* contains the datasets used in this project.
