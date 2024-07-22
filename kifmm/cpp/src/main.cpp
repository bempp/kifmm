// Example usage of FMM from C++
#include "kifmm_pp.hxx"

int main(void) {

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
  size_t targetRank = 10;
  FmmSvdMode svdModeRandom(targetRank, FmmSvdMode::RandomParams(10, 10, 10));
  FmmSvdMode svdModeDeterministic(targetRank);
  FieldTranslation<double> blas = FieldTranslation<double>(
      FieldTranslationType::Blas, static_cast<double>(0.001));

  // FFT field translations
  FieldTranslation<double> fft = FieldTranslation<double>(
      FieldTranslationType::Fft, static_cast<size_t>(10));

  // Tree parameters
  bool pruneEmpty = true; // prune empty boxes in tree
  u_int64_t nCrit = 150;  // Critical value of points per leaf box
  std::vector<size_t> expansion_order = {5}; // Expansion order of FMM

  // Generate FMM runtime object
  KiFmm<double> fmm(expansion_order, coordinates, coordinates, charges, fft,
                    pruneEmpty, nCrit);

  // Run FMM
  fmm.evaluate(false);

  return 0;
}