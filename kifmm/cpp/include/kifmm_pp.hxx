#ifndef KIFMM_HXX
#define KIFMM_HXX

#include <iostream>
#include <kifmm_rs.h>
#include <vector>

class FmmPointer {
public:
  enum class Type { None, Blas32, Blas64 };

  FmmPointer();
  ~FmmPointer();

  void set(LaplaceBlas32 *ptr);
  void set(LaplaceBlas64 *ptr);

  void *get() const;
  Type getType() const;

private:
  void clear();

  void *ptr;
  Type type;
};

template <typename T> struct BlasFieldTranslation {
  size_t target_rank;
  T singular_value_threshold;

  BlasFieldTranslation(size_t target_rank, T singular_value_threshold)
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
  FieldTranslationType type;
  FieldTranslationData<T> data;

  FieldTranslation<T>(FieldTranslationType type, size_t target_rank,
                      T singular_value_threshold);
  FieldTranslation<T>(FieldTranslationType type, size_t block_size);
  ~FieldTranslation<T>();
};

template <typename T> class KiFmm {
public:
  // Constructor
  KiFmm(const std::vector<T> &sources, const std::vector<T> &targets,
        const std::vector<T> &charges,
        const FieldTranslation<T> &fieldTranslation);

  void evaluate(bool timed);

private:
  // Member variables to store source and target coordinates
  std::vector<T> sourceCoordinates;
  std::vector<T> targetCoordinates;
  std::vector<T> sourceCharges;
  FmmPointer fmmInstance;
  FieldTranslation<T> fieldTranslation;
};

// Include the template implementation file
#include "kifmm_pp.txx"

#endif // COORDINATEDATA_HXX
