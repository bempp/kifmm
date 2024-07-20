#ifndef KIFMM_HXX
#define KIFMM_HXX

#include <cstdarg>
#include <iostream>
#include <kifmm_rs.h>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

class FmmPointer {
public:
  // enum class Type { None, Blas32, Blas64 Fft32, };

  FmmPointer();
  ~FmmPointer();

  void set(LaplaceBlas32 *ptr);
  void set(LaplaceBlas64 *ptr);
  void set(LaplaceFft32 *ptr);
  void set(LaplaceFft64 *ptr);

  void *get() const;
  // Type getType() const;

private:
  void clear();

  void *ptr;
  // Type type;
};

template <typename T> struct BlasFieldTranslation {
  size_t target_rank;
  T singular_value_threshold;

  BlasFieldTranslation(
      size_t target_rank,
      T singular_value_threshold = std::numeric_limits<T>::epsilon())
      : target_rank(target_rank),
        singular_value_threshold(singular_value_threshold) {}
};

struct FftFieldTranslation {
  size_t block_size;

  FftFieldTranslation(size_t block_size) : block_size(block_size) {}
};

// Define a union to hold the data
template <typename T> union FieldTranslationData {
  BlasFieldTranslation<T> blas;
  FftFieldTranslation fft;

  FieldTranslationData() {}
  ~FieldTranslationData() {}
};

// Define the enum
enum class FieldTranslationType { Blas, Fft };

// Define a struct to encapsulate the enum and data
template <typename T> struct FieldTranslation {

public:
  FieldTranslationType type;
  FieldTranslationData<T> data;
  T singularValueThreshold;
  size_t blockSize;
  FieldTranslation<T>(FieldTranslationType type, size_t targetRank,
                      T singularValueThreshold);
  FieldTranslation<T>(FieldTranslationType type, size_t blockSize);
  ~FieldTranslation<T>();
};

template <typename T> class KiFmm {
public:
  // Constructor
  KiFmm(const std::vector<size_t> &expansionOrder,
        const std::vector<T> &sources, const std::vector<T> &targets,
        const std::vector<T> &charges,
        const FieldTranslation<T> &fieldTranslation, bool pruneEmpty,
        std::optional<uint64_t> nCrit = std::nullopt,
        std::optional<uint64_t> depth = std::nullopt);

  void evaluate(bool timed);

private:
  // Member variables to store source and target coordinates
  std::vector<T> sourceCoordinates;
  std::vector<T> targetCoordinates;
  std::vector<T> sourceCharges;
  FmmPointer fmmInstance;
  FieldTranslation<T> fieldTranslation;
  bool pruneEmpty;
  uint64_t nCrit;
  uint64_t depth;
  std::vector<size_t> expansionOrder;
};

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

// Include the template implementation file
#include "kifmm_pp.txx"

#endif // COORDINATEDATA_HXX
