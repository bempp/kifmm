// Example usage of FMM from C++
#include "kifmm_pp.hxx"
#include <cstdarg>
#include <iostream>
#include <random>
#include <vector>

// Function to generate random 3D coordinate data
template <typename T>
std::vector<T> generate_random_coordinates(size_t num_points, T min_range,
                                           T max_range) {
  // Random number generation
  std::random_device rd;  // Obtain a random number from hardware
  std::mt19937 gen(rd()); // Seed the generator
  std::uniform_real_distribution<> dis(min_range,
                                       max_range); // Define the range

  std::vector<T> coordinates;
  coordinates.reserve(num_points * 3); // Reserve space for N*3 elements

  for (size_t i = 0; i < num_points; ++i) {
    T x = dis(gen);
    T y = dis(gen);
    T z = dis(gen);
    coordinates.push_back(x);
    coordinates.push_back(y);
    coordinates.push_back(z);
  }

  return coordinates;
}

// Function to generate random 3D coordinate data
template <typename T>
std::vector<T> generate_random_charges(size_t num_points, T min_range,
                                       T max_range) {
  // Random number generation
  std::random_device rd;  // Obtain a random number from hardware
  std::mt19937 gen(rd()); // Seed the generator
  std::uniform_real_distribution<> dis(min_range,
                                       max_range); // Define the range

  std::vector<T> coordinates;
  coordinates.reserve(num_points * 3); // Reserve space for N*3 elements

  for (size_t i = 0; i < num_points; ++i) {
    T x = dis(gen);
    T y = dis(gen);
    T z = dis(gen);
    coordinates.push_back(x);
    coordinates.push_back(y);
    coordinates.push_back(z);
  }

  return coordinates;
}

int main(void) {

  // Create particle data
  size_t num_points = 1000;
  float min_range = 0.0f;
  float max_range = 1.0f;

  // Generate random coordinates
  std::vector<double> coordinates =
      generate_random_coordinates<double>(num_points, min_range, max_range);
  double *sources_ptr = coordinates.data();
  double *targets_ptr = coordinates.data();

  // Generate charge data
  size_t nvecs = 1;
  std::vector<double> charges;
  charges.reserve(num_points * nvecs); // Reserve space for N*3 elements

  for (size_t i = 0; i < num_points; ++i) {
    charges.push_back(1.0f);
  }

  // Choose FMM parameters
  FieldTranslation<double> blas =
      FieldTranslation<double>(FieldTranslationType::Blas, 10, 0.001);

  FieldTranslation<double> fft =
      FieldTranslation<double>(FieldTranslationType::Fft, 10);

  // Generate FMM
  KiFmm<double> fmm(coordinates, coordinates, charges, fft);

  // Run FMM
  fmm.evaluate(false);

  return 0;
}