// Example usage of FMM from C++
#include "kifmm_pp.hxx"
#include <cstdarg>
#include <iostream>
#include <random>
#include <vector>

int main() {
  // Create particle data
  size_t num_points = 1000;
  float min_range = 0.0f;
  float max_range = 1.0f;

  // Generate random coordinates
  std::vector<double> coordinates =
      generate_random_coordinates<double>(num_points, min_range, max_range);

  // Generate charge data
  size_t nvecs = 1;
  std::vector<double> charges;
  charges.reserve(num_points * nvecs); // Reserve space for N*3 elements

  for (size_t i = 0; i < num_points; ++i) {
    charges.push_back(1.0f);
  }

  // Choose FMM parameters

  // BLAS Field Translations
  double singularValueThreshold = 0.001;
  SvdMode<double> svdModeRandom(singularValueThreshold,
                                SvdMode<double>::RandomParams(10, 10, 10));
  SvdMode<double> svdModeDeterministic(singularValueThreshold);

  FieldTranslation<double> blas = FieldTranslation<double>(
      FieldTranslation<double>::Mode::Blas, svdModeRandom);

  // FFT field translations
  FieldTranslation<double> fft = FieldTranslation<double>(
      FieldTranslation<double>::Mode::Fft, static_cast<size_t>(10));

  // Tree parameters
  bool pruneEmpty = true; // prune empty boxes in tree
  u_int64_t nCrit = 150;  // Critical value of points per leaf box
  std::vector<size_t> expansionOrder = {5}; // Expansion order of FMM

  Laplace<double> laplaceKernel;

  // Generate FMM runtime object
  KiFmm<double, double> fmm(expansionOrder, coordinates, coordinates, charges,
                            laplaceKernel, fft, pruneEmpty, nCrit);

  // Run FMM
  fmm.evaluate(false);

  return 0;
}