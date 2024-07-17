#include "kifmm.h"
#include <iostream>
#include <cstdarg>
#include <random>
#include <vector>


// Function to generate random 3D coordinate data
std::vector<float> generate_random_coordinates(size_t num_points, float min_range, float max_range) {
    // Random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> dis(min_range, max_range); // Define the range

    std::vector<float> coordinates;
    coordinates.reserve(num_points * 3); // Reserve space for N*3 elements

    for (size_t i = 0; i < num_points; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        float z = dis(gen);
        coordinates.push_back(x);
        coordinates.push_back(y);
        coordinates.push_back(z);
    }

    return coordinates;
}

// Function to generate random 3D coordinate data
std::vector<float> generate_random_charges(size_t num_points, float min_range, float max_range) {
    // Random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> dis(min_range, max_range); // Define the range

    std::vector<float> coordinates;
    coordinates.reserve(num_points * 3); // Reserve space for N*3 elements

    for (size_t i = 0; i < num_points; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        float z = dis(gen);
        coordinates.push_back(x);
        coordinates.push_back(y);
        coordinates.push_back(z);
    }

    return coordinates;
}

int main(void) {


    std::cout << "Setting FMM Parameters" << std::endl;
    size_t num_points = 1000;
    float min_range = 0.0f;
    float max_range = 1.0f;

    // Generate random coordinates
    std::vector<float> coordinates = generate_random_coordinates(num_points, min_range, max_range);
    float* sources_ptr = coordinates.data();
    float* targets_ptr = coordinates.data();

    // Generate charge data
    size_t nvecs = 1;
    std::vector<float> charges;
    charges.reserve(num_points * nvecs); // Reserve space for N*3 elements

    for (size_t i = 0; i < num_points; ++i) {
        charges.push_back(1.0f);
    }

    float* charges_ptr = charges.data();

    std::cout << "Building FMM" << std::endl;
    LaplaceBlas32* fmm = laplace_blas_f32(sources_ptr, num_points, targets_ptr, num_points, charges_ptr, num_points);

    return 0;
}