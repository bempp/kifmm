#ifndef KIFMM_HXX
#define KIFMM_HXX

#include <iostream>
#include <kifmm_bindings.h>
#include <vector>

class FmmPointer {
public:
  enum class Type { None, Blas32, Blas64 };

  FmmPointer();
  ~FmmPointer();

  void set(LaplaceBlas32 *ptr);
  // void set(LaplaceBlas64* ptr);

  void *get() const;
  Type getType() const;

private:
  void clear();

  void *ptr;
  Type type;
};

template <typename T> class KiFmm {
public:
  // Constructor
  KiFmm(const std::vector<T> &sources, const std::vector<T> &targets,
        const std::vector<T> &charges);

  // Member functions (example)
  void printCoordinates() const;

private:
  // Member variables to store source and target coordinates
  std::vector<T> sourceCoordinates;
  std::vector<T> targetCoordinates;
  std::vector<T> sourceCharges;
  FmmPointer fmmInstance;
};

// Include the template implementation file
#include "kifmm_interface.txx"

#endif // COORDINATEDATA_HXX
